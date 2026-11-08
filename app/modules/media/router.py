from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Form, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import DB, CurrentUser, OptionalUser
from app.api.errors import fail
from app.modules.listings.schemas import PhotoOut
from app.modules.media import service
from app.modules.photo_search.schemas import BBoxIn

router = APIRouter(tags=["media"])


class CropIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox: BBoxIn | None


@router.post("/listings/{uid}/photos", response_model=PhotoOut, status_code=201)
async def upload_photo(
    uid: UUID,
    file: UploadFile,
    db: DB,
    user: CurrentUser,
    request: Request,
    bbox: Annotated[str | None, Form(max_length=200)] = None,
):
    """Optional ``bbox``: "x_min,y_min,x_max,y_max" in [0,1] on the EXIF-oriented image.

    Without bbox the whole image is treated as the user's crop of the dog.
    """
    return await service.upload(
        db,
        user,
        file,
        request.app.state.settings,
        request.app.state.storage,
        listing_id=uid,
        bbox=bbox,
    )


@router.put("/listings/{uid}/photos/{photo_id}/crop", response_model=PhotoOut)
async def set_crop(uid: UUID, photo_id: UUID, data: CropIn, db: DB, user: CurrentUser):
    """Change the user crop; previous CV features are kept as superseded history."""
    return await service.set_crop(db, user, uid, photo_id, data.bbox)


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
