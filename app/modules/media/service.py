from datetime import timedelta
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from app.api.errors import fail
from app.core.security import utcnow
from app.db.models import MediaAsset, PhotoSearch
from app.modules.listings import service as listings
from app.modules.listings.schemas import PhotoOut
from app.modules.media.images import InvalidImage, normalize_image


async def upload(db, user, file: UploadFile, settings, storage, listing_id: UUID | None = None):
    # Reserve before IO; pending rows also count toward the limit.
    uid, user_id = uuid4(), user.id
    if listing_id:
        row = await listings.load(db, listing_id, user, edit=True, lock=True)
        if row.status not in ("draft", "active"):
            fail(409, "listing_not_editable")
        count, maximum = (
            await db.execute(
                select(
                    func.count(MediaAsset.id),
                    func.max(MediaAsset.position),
                ).where(
                    MediaAsset.listing_id == listing_id, MediaAsset.status.in_(["pending", "ready"])
                )
            )
        ).one()
        if count >= 5:
            fail(409, "photo_limit_reached")
        position = (maximum + 1) if maximum is not None else 0
    else:
        position = None
    media = MediaAsset(
        id=uid,
        owner_id=user_id,
        listing_id=listing_id,
        purpose="listing" if listing_id else "search",
        status="pending",
        storage_key=f"{uid}.jpg",
        position=position,
        expires_at=None
        if listing_id
        else utcnow() + timedelta(seconds=settings.search_ttl_seconds),
    )
    db.add(media)
    await db.commit()
    try:
        chunks, size = [], 0
        while chunk := await file.read(64 * 1024):
            size += len(chunk)
            if size > settings.upload_max_bytes:
                fail(413, "upload_too_large")
            chunks.append(chunk)
        normalized = await run_in_threadpool(
            normalize_image,
            b"".join(chunks),
            settings.image_max_pixels,
        )
        await run_in_threadpool(storage.write, media.storage_key, normalized.content)
        if listing_id:
            # Same lock order as delete/publish: listing, then media.
            row = await listings.load(db, listing_id, user, edit=True, lock=True)
            if row.status not in ("draft", "active"):
                fail(409, "listing_not_editable")
        media = await db.scalar(
            select(MediaAsset)
            .where(MediaAsset.id == uid)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if media.status != "pending":
            fail(409, "upload_no_longer_pending")
        media.sha256, media.width, media.height = (
            normalized.sha256,
            normalized.width,
            normalized.height,
        )
        media.content_type, media.size_bytes = "image/jpeg", len(normalized.content)
        media.status = "ready"
        await db.commit()
    except Exception as exc:
        await db.rollback()
        media = await db.get(MediaAsset, uid)
        media.status = "failed"
        await db.commit()
        await run_in_threadpool(storage.delete, f"{uid}.jpg")
        if isinstance(exc, InvalidImage):
            fail(422, "invalid_image")
        raise
    finally:
        await file.close()
    return PhotoOut.model_validate(media)


async def delete_photo(db, user, listing_id, photo_id, storage):
    row = await listings.load(db, listing_id, user, edit=True, lock=True)
    photo = await db.scalar(
        select(MediaAsset)
        .where(
            MediaAsset.id == photo_id,
            MediaAsset.listing_id == listing_id,
        )
        .with_for_update()
    )
    if photo is None or photo.status == "deleted":
        fail(404, "photo_not_found")
    if row.status == "active" and photo.status == "ready":
        count = await db.scalar(
            select(func.count())
            .select_from(MediaAsset)
            .where(
                MediaAsset.listing_id == listing_id,
                MediaAsset.status == "ready",
            )
        )
        if count <= 1:
            fail(409, "last_active_photo")
    photo.status = "deleted"
    key = photo.storage_key
    await db.commit()
    await run_in_threadpool(storage.delete, key)


async def readable(db, uid, user):
    media = await db.get(MediaAsset, uid)
    if media is None:
        fail(404, "media_not_found")
    if media.purpose == "search":
        if user is None or user.id != media.owner_id:
            fail(404, "media_not_found")
        if media.expires_at <= utcnow():
            fail(410, "media_expired")
    else:
        await listings.load(db, media.listing_id, user)
    if media.status != "ready":
        fail(404, "media_not_found")
    return media


async def cleanup(sessions, storage, stale_seconds=3600):
    """Idempotent recovery; preserve all query images referenced by queued/running jobs."""
    from sqlalchemy import and_, or_

    cutoff, now = utcnow() - timedelta(seconds=stale_seconds), utcnow()
    async with sessions() as db:
        live = (
            select(PhotoSearch.id)
            .where(
                PhotoSearch.query_media_id == MediaAsset.id,
                PhotoSearch.status.in_(["queued", "running"]),
            )
            .exists()
        )
        rows = list(
            (
                await db.scalars(
                    select(MediaAsset)
                    .where(
                        ~live,
                        or_(
                            and_(MediaAsset.status == "pending", MediaAsset.created_at < cutoff),
                            MediaAsset.status.in_(["failed", "deleted"]),
                            MediaAsset.expires_at <= now,
                        ),
                    )
                    .with_for_update(skip_locked=True)
                )
            ).all()
        )
        keys = [row.storage_key for row in rows]
        for row in rows:
            row.status = "deleted"
        await db.commit()
    for key in keys:
        await run_in_threadpool(storage.delete, key)
    files = await run_in_threadpool(storage.old_files, cutoff.timestamp())
    orphans = 0
    # Re-check each aged file against current DB state, rather than a stale inventory.
    for path in files:
        key = path.stem + ".jpg"
        try:
            storage.path(key)
        except ValueError:
            continue
        async with sessions() as db:
            exists = await db.scalar(select(MediaAsset.id).where(MediaAsset.storage_key == key))
        if exists is None:
            await run_in_threadpool(storage.delete, key)
            orphans += 1
    return {"cleaned_media": len(keys), "orphans": orphans}
