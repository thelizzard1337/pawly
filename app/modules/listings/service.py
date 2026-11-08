from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import fail
from app.api.pagination import Page, page_parts, paginate
from app.db.models import Listing, MediaAsset, Shelter, ShelterMember, User
from app.modules.listings import repository as repo
from app.modules.listings.schemas import (
    ListingCreate,
    ListingOut,
    ListingPatch,
    Location,
    MapOut,
    MapPoint,
    PhotoOut,
)


async def is_manager(db, user, shelter_id):
    return bool(user and shelter_id and await db.get(ShelterMember, (shelter_id, user.id)))


async def may_edit(db, user, listing):
    if user is None:
        return False
    if listing.kind == "adoption":
        return user.is_admin or await is_manager(db, user, listing.shelter_id)
    return user.is_admin or user.id == listing.author_id


async def require_edit(db, user, listing):
    if not await may_edit(db, user, listing):
        fail(404, "listing_not_found")


async def require_manager(db, user, shelter_id, verified=False):
    shelter = await db.get(Shelter, shelter_id)
    if shelter is None or not await is_manager(db, user, shelter_id):
        fail(403, "shelter_manager_required")
    if verified and shelter.verification_status != "verified":
        fail(409, "shelter_not_verified")
    return shelter


async def load(db, uid, user=None, edit=False, lock=False):
    row = await repo.get_listing(db, uid, lock)
    if row is None:
        fail(404, "listing_not_found")
    if edit or row.status != "active" or row.is_hidden:
        await require_edit(db, user, row)
    return row


def apply_fields(row, fields):
    for key, value in fields.items():
        if key == "location" and value is not None:
            value = f"SRID=4326;POINT({value['longitude']} {value['latitude']})"
        setattr(row, key, value)


async def outputs(db, rows):
    locations, photos = await repo.listing_details(db, rows)
    result = []
    for row in rows:
        lat, lon = locations[row.id]
        data = {
            key: getattr(row, key)
            for key in ListingOut.model_fields
            if key not in ("location", "photos")
        }
        data["location"] = Location(latitude=lat, longitude=lon) if lat is not None else None
        data["photos"] = [PhotoOut.model_validate(x) for x in photos.get(row.id, [])]
        result.append(ListingOut(**data))
    return result


async def output(db, row):
    return (await outputs(db, [row]))[0]


async def create(db: AsyncSession, user: User, data: ListingCreate):
    if data.kind == "adoption":
        await require_manager(db, user, data.shelter_id)
    row = Listing(author_id=user.id, kind=data.kind, shelter_id=data.shelter_id)
    apply_fields(row, data.model_dump(exclude={"kind", "shelter_id"}))
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return await output(db, row)


async def validate_publish(db, row, user):
    if row.kind == "adoption":
        await require_manager(db, user, row.shelter_id, verified=True)
    if not row.species or not row.location or not row.description or not row.description.strip():
        fail(409, "listing_incomplete")
    if row.kind != "adoption" and row.event_at is None:
        fail(409, "event_at_required")
    count = await db.scalar(
        select(func.count())
        .select_from(MediaAsset)
        .where(
            MediaAsset.listing_id == row.id,
            MediaAsset.status == "ready",
        )
    )
    if count == 0:
        fail(409, "ready_photo_required")


async def patch(db, user, uid, data: ListingPatch):
    row = await load(db, uid, user, edit=True, lock=True)
    if row.status not in ("draft", "active"):
        fail(409, "listing_not_editable")
    apply_fields(row, data.model_dump(exclude_unset=True))
    if row.status == "active":
        # Disable autoflush until service checks produce a useful domain error.
        with db.no_autoflush:
            await validate_publish(db, row, user)
    await db.commit()
    await db.refresh(row)
    return await output(db, row)


async def transition(db, user, uid, action):
    row = await load(db, uid, user, edit=True, lock=True)
    if action == "publish":
        if row.status != "draft" or row.is_hidden:
            fail(409, "cannot_publish")
        await validate_publish(db, row, user)
        row.status = "active"
    elif action == "resolve":
        if row.status != "active":
            fail(409, "cannot_resolve")
        row.status = "resolved"
    elif action == "archive":
        row.status = "archived"
    await db.commit()
    await db.refresh(row)
    return await output(db, row)


async def public_page(db, query, shelter_id=None):
    rows = await repo.list_public(db, query, query.limit, query.cursor, shelter_id)
    rows, cursor = page_parts(rows, query.limit)
    return Page[ListingOut](items=await outputs(db, rows), next_cursor=cursor)


async def my_page(db, user, limit, cursor):
    stmt = select(Listing).where(Listing.author_id == user.id)
    rows = list((await db.scalars(paginate(stmt, Listing, cursor, limit))).all())
    rows, cursor = page_parts(rows, limit)
    return Page[ListingOut](items=await outputs(db, rows), next_cursor=cursor)


async def map_points(db, bbox: str, limit: int):
    """bbox=west,south,east,north; no antimeridian wrapping. Limit+1 detects truncation."""
    import math

    from sqlalchemy import cast

    try:
        west, south, east, north = [float(v) for v in bbox.split(",")]
        if not all(math.isfinite(v) for v in (west, south, east, north)):
            raise ValueError
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError
    except ValueError:
        fail(422, "invalid_bbox")
    geometry = cast(Listing.location, repo.Geometry())
    rows = list(
        (
            await db.scalars(
                select(Listing)
                .where(
                    *repo.public_conditions(),
                    func.ST_Intersects(
                        geometry, func.ST_MakeEnvelope(west, south, east, north, 4326)
                    ),
                )
                .order_by(Listing.created_at.desc(), Listing.id.desc())
                .limit(limit + 1)
            )
        ).all()
    )
    selected = rows[:limit]
    locations, _ = await repo.listing_details(db, selected)
    return MapOut(
        items=[
            MapPoint(
                id=row.id,
                kind=row.kind,
                species=row.species,
                title=row.title,
                location=Location(latitude=locations[row.id][0], longitude=locations[row.id][1]),
            )
            for row in selected
        ],
        truncated=len(rows) > limit,
        limit=limit,
    )
