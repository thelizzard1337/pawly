from uuid import UUID

from sqlalchemy import cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import UserDefinedType

from app.api.pagination import paginate
from app.db.models import Listing, MediaAsset
from app.modules.listings.schemas import Filters


class Geometry(UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **kw):
        return "geometry"


def public_conditions():
    return Listing.status == "active", Listing.is_hidden.is_(False)


def apply_filters(statement, filters: Filters):
    for field in ("kind", "species", "sex", "color"):
        value = getattr(filters, field)
        if value is not None:
            statement = statement.where(getattr(Listing, field) == value)
    if filters.event_from is not None:
        statement = statement.where(Listing.event_at >= filters.event_from)
    if filters.event_to is not None:
        statement = statement.where(Listing.event_at <= filters.event_to)
    if filters.latitude is not None:
        point = func.ST_SetSRID(func.ST_MakePoint(filters.longitude, filters.latitude), 4326)
        from app.db.models import Geography

        statement = statement.where(
            func.ST_DWithin(
                Listing.location,
                cast(point, Geography()),
                filters.radius_m,
            )
        )
    return statement


async def get_listing(db: AsyncSession, uid: UUID, lock=False):
    stmt = select(Listing).where(Listing.id == uid)
    if lock:
        stmt = stmt.with_for_update().execution_options(populate_existing=True)
    return await db.scalar(stmt)


async def list_public(db, filters, limit, cursor, shelter_id=None):
    stmt = apply_filters(select(Listing).where(*public_conditions()), filters)
    if shelter_id:
        stmt = stmt.where(Listing.shelter_id == shelter_id)
    return list((await db.scalars(paginate(stmt, Listing, cursor, limit))).all())


async def listing_details(db: AsyncSession, listings: list[Listing]):
    if not listings:
        return {}, {}
    ids = [row.id for row in listings]
    geometry = cast(Listing.location, Geometry())
    locations = {
        row.id: (row.lat, row.lon)
        for row in (
            await db.execute(
                select(
                    Listing.id, func.ST_Y(geometry).label("lat"), func.ST_X(geometry).label("lon")
                ).where(Listing.id.in_(ids)),
            )
        )
    }
    photos = {}
    for media in await db.scalars(
        select(MediaAsset)
        .where(
            MediaAsset.listing_id.in_(ids),
            MediaAsset.status == "ready",
        )
        .order_by(MediaAsset.position)
    ):
        photos.setdefault(media.listing_id, []).append(media)
    return locations, photos
