from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.api.errors import fail
from app.api.pagination import Page, page_parts, paginate
from app.core.security import utcnow
from app.db.models import AdoptionRequest, ContactRequest, Listing
from app.modules.adoption.schemas import AdoptionOut, ContactOut
from app.modules.listings import service as listings


async def create(db, user, uid, data):
    listing = await listings.load(db, uid, user, lock=True)
    if listing.kind != "adoption" or listing.status != "active" or listing.is_hidden:
        fail(409, "adoption_unavailable")
    if listing.author_id == user.id or await listings.is_manager(db, user, listing.shelter_id):
        fail(409, "cannot_apply_to_own_shelter")
    request = AdoptionRequest(listing_id=uid, applicant_id=user.id, message=data.message)
    db.add(request)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        fail(409, "pending_request_exists")
    await db.refresh(request)
    return request


async def decide(db, user, uid, action):
    request = await db.get(AdoptionRequest, uid)
    if request is None:
        fail(404, "request_not_found")
    # Listing lock is always first, including withdraw/reject; serialize all decisions.
    listing = await db.scalar(
        select(Listing)
        .where(Listing.id == request.listing_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if action == "withdraw":
        if request.applicant_id != user.id:
            fail(404, "request_not_found")
    else:
        await listings.require_manager(db, user, listing.shelter_id, verified=True)
    request = await db.scalar(
        select(AdoptionRequest)
        .where(AdoptionRequest.id == uid)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if request.status != "pending":
        fail(409, "request_not_pending")
    if action == "approve":
        if listing.status != "active" or listing.is_hidden:
            fail(409, "adoption_unavailable")
        request.status = "approved"
        request.reviewed_by, request.reviewed_at = user.id, utcnow()
        listing.status = "resolved"
        await db.execute(
            update(AdoptionRequest)
            .where(
                AdoptionRequest.listing_id == listing.id,
                AdoptionRequest.id != uid,
                AdoptionRequest.status == "pending",
            )
            .values(status="rejected", reviewed_by=user.id, reviewed_at=request.reviewed_at)
        )
    elif action == "reject":
        request.status = "rejected"
        request.reviewed_by, request.reviewed_at = user.id, utcnow()
    else:
        request.status = "withdrawn"
    await db.commit()
    await db.refresh(request)
    return request


async def page(db, user, limit, cursor, shelter_id=None):
    if shelter_id:
        await listings.require_manager(db, user, shelter_id)
        stmt = select(AdoptionRequest).join(Listing).where(Listing.shelter_id == shelter_id)
    else:
        stmt = select(AdoptionRequest).where(AdoptionRequest.applicant_id == user.id)
    rows = list((await db.scalars(paginate(stmt, AdoptionRequest, cursor, limit))).all())
    rows, cursor = page_parts(rows, limit)
    return Page[AdoptionOut](
        items=[AdoptionOut.model_validate(r) for r in rows], next_cursor=cursor
    )


async def contact(db, user, uid, data):
    listing = await listings.load(db, uid, user, lock=True)
    if listing.kind not in ("lost", "found") or listing.status != "active" or listing.is_hidden:
        fail(409, "contact_unavailable")
    if listing.author_id == user.id:
        fail(409, "cannot_contact_self")
    row = ContactRequest(listing_id=uid, sender_id=user.id, message=data.message)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def inbox(db, user, limit, cursor):
    stmt = select(ContactRequest).join(Listing).where(Listing.author_id == user.id)
    rows = list((await db.scalars(paginate(stmt, ContactRequest, cursor, limit))).all())
    rows, cursor = page_parts(rows, limit)
    return Page[ContactOut](items=[ContactOut.model_validate(r) for r in rows], next_cursor=cursor)
