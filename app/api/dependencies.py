from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import fail
from app.core.security import decode_access
from app.db.models import User

bearer = HTTPBearer(auto_error=False)


async def session(request: Request):
    async with request.app.state.sessions() as db:
        yield db


DB = Annotated[AsyncSession, Depends(session)]


async def optional_user(
    request: Request,
    db: DB,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> User | None:
    if credentials is None:
        return None
    try:
        uid = decode_access(credentials.credentials, request.app.state.settings)
    except (jwt.PyJWTError, ValueError, TypeError):
        fail(401, "invalid_access_token")
    user = await db.get(User, uid)
    if user is None or not user.is_active:
        fail(401, "inactive_or_missing_user")
    return user


OptionalUser = Annotated[User | None, Depends(optional_user)]


async def current_user(user: OptionalUser) -> User:
    if user is None:
        fail(401, "authentication_required")
    return user


CurrentUser = Annotated[User, Depends(current_user)]


async def admin_user(user: CurrentUser) -> User:
    if not user.is_admin:
        fail(403, "admin_required")
    return user


AdminUser = Annotated[User, Depends(admin_user)]
