from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from app.api.dependencies import DB, AdminUser, CurrentUser
from app.api.errors import fail
from app.api.pagination import Page
from app.modules.listings import service as listings
from app.modules.listings.schemas import ListingCreate, ListingOut, ListingQuery
from app.modules.shelters import service
from app.modules.shelters.schemas import ShelterCreate, ShelterOut, ShelterPatch, VerifyIn

router = APIRouter(tags=["shelters"])


@router.get("/shelters", response_model=Page[ShelterOut])
async def page(
    db: DB, limit: int = Query(20, ge=1, le=100), cursor: str | None = Query(None, max_length=300)
):
    return await service.page(db, limit, cursor)


@router.post("/shelters", response_model=ShelterOut, status_code=201)
async def create(data: ShelterCreate, db: DB, user: CurrentUser):
    return await service.create(db, user, data)


@router.get("/shelters/{uid}", response_model=ShelterOut)
async def detail(uid: UUID, db: DB):
    return await service.detail(db, uid)


@router.patch("/shelters/{uid}", response_model=ShelterOut)
async def patch(uid: UUID, data: ShelterPatch, db: DB, user: CurrentUser):
    return await service.patch(db, user, uid, data)


@router.get("/shelters/{uid}/listings", response_model=Page[ListingOut])
async def shelter_listings(uid: UUID, db: DB, query: Annotated[ListingQuery, Query()]):
    await service.detail(db, uid)
    return await listings.public_page(db, query, shelter_id=uid)


@router.post("/shelters/{uid}/listings", response_model=ListingOut, status_code=201)
async def create_listing(uid: UUID, data: ListingCreate, db: DB, user: CurrentUser):
    if data.kind != "adoption" or data.shelter_id != uid:
        fail(422, "shelter_listing_mismatch")
    return await listings.create(db, user, data)


@router.post("/admin/shelters/{uid}/verify", response_model=ShelterOut, tags=["admin"])
async def verify(uid: UUID, db: DB, user: AdminUser, data: VerifyIn = VerifyIn()):
    return await service.verify(db, uid, data.verification_status)


@router.post("/admin/listings/{uid}/hide", response_model=ListingOut, tags=["admin"])
async def hide(uid: UUID, db: DB, user: AdminUser):
    return await service.hide(db, uid, True)


@router.post("/admin/listings/{uid}/unhide", response_model=ListingOut, tags=["admin"])
async def unhide(uid: UUID, db: DB, user: AdminUser):
    return await service.hide(db, uid, False)
