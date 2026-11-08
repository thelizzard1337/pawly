from uuid import UUID

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import DB, CurrentUser, OptionalUser
from app.api.errors import fail
from app.modules.listings.schemas import PhotoOut
from app.modules.media import service

router = APIRouter(tags=["media"])


@router.post("/listings/{uid}/photos", response_model=PhotoOut, status_code=201)
async def upload_photo(uid: UUID, file: UploadFile, db: DB, user: CurrentUser, request: Request):
    return await service.upload(
        db, user, file, request.app.state.settings, request.app.state.storage, listing_id=uid
    )


@router.delete("/listings/{uid}/photos/{photo_id}", status_code=204)
async def delete_photo(uid: UUID, photo_id: UUID, db: DB, user: CurrentUser, request: Request):
    await service.delete_photo(db, user, uid, photo_id, request.app.state.storage)


@router.get("/media/{uid}", responses={200: {"content": {"image/jpeg": {}}}})
async def get_media(uid: UUID, db: DB, user: OptionalUser, request: Request):
    media = await service.readable(db, uid, user)
    path = request.app.state.storage.path(media.storage_key)
    if not await run_in_threadpool(path.is_file):
        fail(404, "media_file_missing")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
