from sqlalchemy import select

from app.api.errors import fail
from app.api.pagination import Page, page_parts, paginate
from app.db.models import Listing, Shelter, ShelterMember
from app.modules.listings.service import require_manager
from app.modules.shelters.schemas import ShelterOut


async def create(db, user, data):
    shelter = Shelter(**data.model_dump(), verification_status="pending")
    db.add(shelter)
    await db.flush()
    db.add(ShelterMember(shelter_id=shelter.id, user_id=user.id, role="manager"))
    await db.commit()
    await db.refresh(shelter)
    return shelter


async def detail(db, uid):
    row = await db.get(Shelter, uid)
    if row is None:
        fail(404, "shelter_not_found")
    return row


async def page(db, limit, cursor):
    rows = list((await db.scalars(paginate(select(Shelter), Shelter, cursor, limit))).all())
    rows, cursor = page_parts(rows, limit)
    return Page[ShelterOut](items=[ShelterOut.model_validate(r) for r in rows], next_cursor=cursor)


async def patch(db, user, uid, data):
    await require_manager(db, user, uid)
    row = await db.scalar(select(Shelter).where(Shelter.id == uid).with_for_update())
    for key, value in data.model_dump(exclude_unset=True).items():
        if value is None:
            fail(422, "shelter_field_required")
        setattr(row, key, value)
    await db.commit()
    await db.refresh(row)
    return row


async def verify(db, uid, status):
    row = await db.scalar(select(Shelter).where(Shelter.id == uid).with_for_update())
    if row is None:
        fail(404, "shelter_not_found")
    row.verification_status = status
    await db.commit()
    await db.refresh(row)
    return row


async def hide(db, uid, hidden):
    row = await db.scalar(select(Listing).where(Listing.id == uid).with_for_update())
    if row is None:
        fail(404, "listing_not_found")
    row.is_hidden = hidden
    await db.commit()
    await db.refresh(row)
    from app.modules.listings.service import output

    return await output(db, row)
