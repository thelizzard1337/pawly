from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Request, Response, UploadFile

from app.api.dependencies import DB, CurrentUser
from app.modules.listings.schemas import PhotoOut
from app.modules.media import service as media
from app.modules.photo_search import service
from app.modules.photo_search.schemas import SearchAccepted, SearchCreate, SearchOut

router = APIRouter(prefix="/photo-searches", tags=["photo search"])


@router.post("/images", response_model=PhotoOut, status_code=201)
async def image(file: UploadFile, db: DB, user: CurrentUser, request: Request):
    return await media.upload(db, user, file, request.app.state.settings, request.app.state.storage)


@router.post("", response_model=SearchAccepted, status_code=202)
async def create(
    data: SearchCreate,
    db: DB,
    user: CurrentUser,
    request: Request,
    response: Response,
    idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
):
    result = await service.create(db, user, data, request.app.state.settings, idempotency_key)
    response.headers["Location"] = f"/api/v1/photo-searches/{result.id}"
    return result


@router.get("/{uid}", response_model=SearchOut)
async def detail(uid: UUID, db: DB, user: CurrentUser):
    return await service.detail(db, user, uid)
