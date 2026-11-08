from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Request, Response, UploadFile

from app.api.dependencies import DB, CurrentUser
from app.modules.listings.schemas import PhotoOut
from app.modules.media import service as media
from app.modules.photo_search import service
from app.modules.photo_search.schemas import (
    SearchAccepted,
    SearchCreate,
    SearchCreateV2,
    SearchOut,
    SearchOutV2,
)

router = APIRouter(prefix="/photo-searches", tags=["photo search"])
router_v2 = APIRouter(prefix="/photo-searches", tags=["photo search v2"])


@router.post("/images", response_model=PhotoOut, status_code=201)
async def image(file: UploadFile, db: DB, user: CurrentUser, request: Request):
    """Upload a private query image (shared by v1 and v2 searches)."""
    return await media.upload(db, user, file, request.app.state.settings, request.app.state.storage)


@router.post("", response_model=SearchAccepted, status_code=202, summary="Create search (legacy)")
async def create(
    data: SearchCreate,
    db: DB,
    user: CurrentUser,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
):
    """Legacy v1 contract: no breed/colour text, at most 5 candidates. Prefer POST /api/v2."""
    result = await service.create(db, user, data, request.app.state.settings, idempotency_key)
    response.headers["Location"] = f"/api/v1/photo-searches/{result.id}"
    return result


@router.get("/{uid}", response_model=SearchOut)
async def detail(uid: UUID, db: DB, user: CurrentUser):
    return await service.detail(db, user, uid)


@router_v2.post("", response_model=SearchAccepted, status_code=202)
async def create_v2(
    data: SearchCreateV2,
    db: DB,
    user: CurrentUser,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
):
    """Dog photo search. Upload the image first via POST /api/v1/photo-searches/images.

    Results are candidates in review order; no similarity/confidence values are returned.
    """
    result = await service.create(db, user, data, request.app.state.settings, idempotency_key)
    response.headers["Location"] = f"/api/v2/photo-searches/{result.id}"
    return result


@router_v2.get("/{uid}", response_model=SearchOutV2)
async def detail_v2(uid: UUID, db: DB, user: CurrentUser):
    return await service.detail_v2(db, user, uid)
