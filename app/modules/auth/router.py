from fastapi import APIRouter, Request, Response

from app.api.dependencies import DB
from app.core.security import new_token
from app.modules.auth import service
from app.modules.auth.schemas import CsrfOut, Login, Register, TokenOut
from app.modules.users.schemas import UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/csrf", response_model=CsrfOut)
async def csrf(request: Request, response: Response):
    """Get double-submit CSRF token; send it as X-CSRF-Token on refresh/logout."""
    token = new_token()
    response.set_cookie(
        "pawly_csrf",
        token,
        secure=request.app.state.settings.cookie_secure,
        httponly=False,
        samesite="lax",
        path=service.COOKIE_PATH,
    )
    response.headers["Cache-Control"] = "no-store"
    return CsrfOut(csrf_token=token)


@router.post("/register", response_model=UserOut, status_code=201)
async def register(data: Register, db: DB):
    return await service.register(db, data)


@router.post("/login", response_model=TokenOut)
async def login(data: Login, db: DB, request: Request, response: Response):
    return await service.login(db, data, response, request.app.state.settings)


@router.post("/refresh", response_model=TokenOut)
async def refresh(db: DB, request: Request, response: Response):
    return await service.refresh(db, request, response)


@router.post("/logout", status_code=204)
async def logout(db: DB, request: Request, response: Response):
    await service.logout(db, request, response)
