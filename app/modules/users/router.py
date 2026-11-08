from fastapi import APIRouter

from app.api.dependencies import DB, CurrentUser
from app.modules.users import service
from app.modules.users.schemas import ProfileOut, ProfilePatch

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=ProfileOut)
async def me(db: DB, user: CurrentUser):
    return await service.profile(db, user)


@router.patch("/me", response_model=ProfileOut)
async def patch_me(data: ProfilePatch, db: DB, user: CurrentUser):
    return await service.update_profile(db, user, data)
