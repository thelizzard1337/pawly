from fastapi import APIRouter

from app.modules.adoption.router import router as adoption
from app.modules.auth.router import router as auth
from app.modules.cv_index.router import router as cv_status
from app.modules.listings.router import router as listings
from app.modules.media.router import router as media
from app.modules.photo_search.router import router as photo_search
from app.modules.photo_search.router import router_v2 as photo_search_v2
from app.modules.shelters.router import router as shelters
from app.modules.users.router import router as users

router = APIRouter(prefix="/api/v1")
for module in (auth, users, listings, media, shelters, adoption, photo_search, cv_status):
    router.include_router(module)

router_v2 = APIRouter(prefix="/api/v2")
router_v2.include_router(photo_search_v2)
