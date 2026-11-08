from uuid import UUID

from fastapi import APIRouter, Query

from app.api.dependencies import DB, CurrentUser
from app.api.pagination import Page
from app.modules.adoption import service
from app.modules.adoption.schemas import AdoptionOut, ContactOut, MessageIn

router = APIRouter(tags=["adoption and contact"])


@router.post("/listings/{uid}/adoption-requests", response_model=AdoptionOut, status_code=201)
async def create(uid: UUID, data: MessageIn, db: DB, user: CurrentUser):
    return await service.create(db, user, uid, data)


@router.get("/users/me/adoption-requests", response_model=Page[AdoptionOut])
async def mine(
    db: DB,
    user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None, max_length=300),
):
    return await service.page(db, user, limit, cursor)


@router.get("/shelters/{uid}/adoption-requests", response_model=Page[AdoptionOut])
async def shelter_requests(
    uid: UUID,
    db: DB,
    user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None, max_length=300),
):
    return await service.page(db, user, limit, cursor, shelter_id=uid)


@router.post("/adoption-requests/{uid}/approve", response_model=AdoptionOut)
async def approve(uid: UUID, db: DB, user: CurrentUser):
    return await service.decide(db, user, uid, "approve")


@router.post("/adoption-requests/{uid}/reject", response_model=AdoptionOut)
async def reject(uid: UUID, db: DB, user: CurrentUser):
    return await service.decide(db, user, uid, "reject")


@router.post("/adoption-requests/{uid}/withdraw", response_model=AdoptionOut)
async def withdraw(uid: UUID, db: DB, user: CurrentUser):
    return await service.decide(db, user, uid, "withdraw")


@router.post("/listings/{uid}/contact-requests", response_model=ContactOut, status_code=201)
async def contact(uid: UUID, data: MessageIn, db: DB, user: CurrentUser):
    return await service.contact(db, user, uid, data)


@router.get("/users/me/contact-requests", response_model=Page[ContactOut])
async def inbox(
    db: DB,
    user: CurrentUser,
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None, max_length=300),
):
    return await service.inbox(db, user, limit, cursor)
