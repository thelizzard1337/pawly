import hashlib
import json
from datetime import timedelta

from sqlalchemy import select

from app.api.errors import fail
from app.core.security import utcnow
from app.db.models import Listing, MediaAsset, PhotoSearch, PhotoSearchResult, User
from app.modules.listings import repository as listings_repo
from app.modules.listings import service as listings
from app.modules.photo_search.schemas import SearchAccepted, SearchOut, SearchResultOut


def fingerprint(data):
    canonical = json.dumps(data.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def create(db, user, data, settings, idempotency_key):
    if idempotency_key is not None:
        if not idempotency_key.strip() or len(idempotency_key) > 200:
            fail(422, "invalid_idempotency_key")
    digest = fingerprint(data)
    # Serialize creation per owner; the partial unique index is a second line of defense.
    await db.scalar(select(User).where(User.id == user.id).with_for_update())
    if idempotency_key:
        existing = await db.scalar(
            select(PhotoSearch).where(
                PhotoSearch.owner_id == user.id,
                PhotoSearch.idempotency_key == idempotency_key,
            )
        )
        if existing:
            if existing.request_fingerprint != digest:
                fail(409, "idempotency_payload_conflict")
            if existing.expires_at <= utcnow():
                fail(410, "search_expired")
            return SearchAccepted(id=existing.id, status=existing.status)
    media = await db.scalar(
        select(MediaAsset)
        .where(
            MediaAsset.id == data.query_media_id,
        )
        .with_for_update()
    )
    if media is None or media.owner_id != user.id or media.purpose != "search":
        fail(404, "query_media_not_found")
    if media.expires_at <= utcnow():
        fail(410, "query_media_expired")
    if media.status != "ready":
        fail(409, "query_media_not_ready")
    expiry = utcnow() + timedelta(seconds=settings.search_ttl_seconds)
    media.expires_at = max(media.expires_at, expiry)
    row = PhotoSearch(
        owner_id=user.id,
        query_media_id=media.id,
        scope=data.filters().model_dump(mode="json"),
        limit=data.limit,
        status="queued",
        provider="mock-v1",
        expires_at=expiry,
        idempotency_key=idempotency_key,
        request_fingerprint=digest,
    )
    db.add(row)
    await db.commit()
    return SearchAccepted(id=row.id, status=row.status)


async def detail(db, user, uid):
    row = await db.get(PhotoSearch, uid)
    if row is None or row.owner_id != user.id:
        fail(404, "search_not_found")
    if row.expires_at <= utcnow():
        fail(410, "search_expired")
    results = (
        await db.execute(
            select(PhotoSearchResult, Listing)
            .join(
                Listing,
                Listing.id == PhotoSearchResult.listing_id,
            )
            .join(MediaAsset, MediaAsset.id == PhotoSearchResult.matched_media_id)
            .where(
                PhotoSearchResult.search_id == uid,
                *listings_repo.public_conditions(),
                MediaAsset.listing_id == Listing.id,
                MediaAsset.status == "ready",
            )
            .order_by(PhotoSearchResult.rank)
        )
    ).all()
    dtos = await listings.outputs(db, [listing for _, listing in results])
    return SearchOut(
        id=row.id,
        status=row.status,
        provider=row.provider,
        model_version=row.model_version,
        is_mock=True,
        error=row.error_code,
        created_at=row.created_at,
        expires_at=row.expires_at,
        results=[
            SearchResultOut(
                listing=dto,
                matched_media_id=result.matched_media_id,
                rank=result.rank,
                similarity_score=None,
            )
            for (result, _), dto in zip(results, dtos, strict=True)
        ],
    )
