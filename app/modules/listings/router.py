from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from app.api.dependencies import DB, CurrentUser, OptionalUser
from app.api.pagination import Page
from app.modules.listings import service
from app.modules.listings.schemas import (
    ListingCreate,
    ListingOut,
    ListingPatch,
    ListingQuery,
    MapOut,
)

router = APIRouter(tags=["listings"])


@router.get("/listings", response_model=Page[ListingOut])
async def listings(db: DB, query: Annotated[ListingQuery, Query()]):
    return await service.public_page(db, query)


@router.get("/listings/map", response_model=MapOut)
async def map_points(
    db: DB, bbox: str = Query(max_length=100), limit: int = Query(500, ge=1, le=1000)
):
    """bbox: west,south,east,north (degrees). At most 1000 points; truncated=true on overflow."""
    return await service.map_points(db, bbox, limit)


@router.post("/listings", response_model=ListingOut, status_code=201)
async def create(data: ListingCreate, db: DB, user: CurrentUser):
    return await service.create(db, user, data)


@router.get("/users/me/listings", response_model=Page[ListingOut])
async def mine(
    db: DB,
    user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None, max_length=300),
):
    return await service.my_page(db, user, limit, cursor)


@router.get("/listings/{uid}", response_model=ListingOut)
async def detail(uid: UUID, db: DB, user: OptionalUser):
    return await service.output(db, await service.load(db, uid, user))


@router.patch("/listings/{uid}", response_model=ListingOut)
async def patch(uid: UUID, data: ListingPatch, db: DB, user: CurrentUser):
    return await service.patch(db, user, uid, data)


@router.delete("/listings/{uid}", response_model=ListingOut)
async def archive(uid: UUID, db: DB, user: CurrentUser):
    return await service.transition(db, user, uid, "archive")


@router.post("/listings/{uid}/publish", response_model=ListingOut)
async def publish(uid: UUID, db: DB, user: CurrentUser):
    return await service.transition(db, user, uid, "publish")


@router.post("/listings/{uid}/resolve", response_model=ListingOut)
async def resolve(uid: UUID, db: DB, user: CurrentUser):
    return await service.transition(db, user, uid, "resolve")
