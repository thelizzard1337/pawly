import secrets
from datetime import timedelta
from uuid import uuid4

from fastapi import Request, Response
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.errors import fail
from app.core.config import Settings
from app.core.security import access_token, new_token, password_hasher, token_hash, utcnow
from app.db.models import RefreshSession, User
from app.modules.auth.schemas import Login, Register, TokenOut

COOKIE_PATH = "/api/v1/auth"
# Used to equalize the expensive password verification path for unknown emails.
DUMMY_HASH = password_hasher.hash("not-a-user-password")


def check_csrf(request: Request):
    settings = request.app.state.settings
    origin = request.headers.get("origin")
    if origin is not None and origin not in settings.cors_origins:
        fail(403, "origin_not_allowed")
    cookie = request.cookies.get("pawly_csrf", "")
    header = request.headers.get("x-csrf-token", "")
    if not cookie or not header or not secrets.compare_digest(cookie, header):
        fail(403, "csrf_required")


def set_cookies(response: Response, token: str, settings: Settings):
    response.set_cookie(
        "pawly_refresh",
        token,
        max_age=settings.refresh_ttl_seconds,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path=COOKIE_PATH,
    )


async def register(db: AsyncSession, data: Register) -> User:
    hashed = await run_in_threadpool(password_hasher.hash, data.password)
    user = User(email=data.email, display_name=data.display_name, password_hash=hashed)
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        fail(409, "email_already_registered")
    await db.refresh(user)
    return user


async def login(db: AsyncSession, data: Login, response: Response, settings: Settings) -> TokenOut:
    user = await db.scalar(select(User).where(User.email == data.email))
    valid = await run_in_threadpool(
        password_hasher.verify,
        data.password,
        user.password_hash if user else DUMMY_HASH,
    )
    if not valid or user is None or not user.is_active:
        fail(401, "invalid_credentials")
    token = new_token()
    db.add(
        RefreshSession(
            user_id=user.id,
            family_id=uuid4(),
            token_hash=token_hash(token),
            expires_at=utcnow() + timedelta(seconds=settings.refresh_ttl_seconds),
        )
    )
    await db.commit()
    set_cookies(response, token, settings)
    return TokenOut(
        access_token=access_token(user.id, settings), expires_in=settings.access_ttl_seconds
    )


async def locked_refresh(db: AsyncSession, raw_token: str | None):
    if not raw_token:
        fail(401, "refresh_required")
    # Serialize the entire family, including logout and simultaneous reuse/rotation.
    # An advisory lock avoids locking an old token while another request inserts its successor.
    row = await db.scalar(
        select(RefreshSession).where(
            RefreshSession.token_hash == token_hash(raw_token),
        )
    )
    if row is None:
        fail(401, "invalid_refresh")
    family_id = row.family_id
    await db.rollback()
    from sqlalchemy import text

    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key,0))"), {"key": str(family_id)}
    )
    row = await db.scalar(
        select(RefreshSession)
        .where(
            RefreshSession.token_hash == token_hash(raw_token),
        )
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    return row


async def refresh(db: AsyncSession, request: Request, response: Response) -> TokenOut:
    check_csrf(request)
    settings = request.app.state.settings
    row = await locked_refresh(db, request.cookies.get("pawly_refresh"))
    now = utcnow()
    if row.revoked_at is not None:
        await db.execute(
            update(RefreshSession)
            .where(
                RefreshSession.family_id == row.family_id,
                RefreshSession.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        await db.commit()  # Must survive the HTTP error below.
        fail(401, "refresh_reuse_detected")
    if row.expires_at <= now:
        row.revoked_at = now
        await db.commit()
        fail(401, "refresh_expired")
    user = await db.get(User, row.user_id)
    if user is None or not user.is_active:
        fail(401, "inactive_user")
    token = new_token()
    successor = RefreshSession(
        id=uuid4(),
        user_id=user.id,
        family_id=row.family_id,
        token_hash=token_hash(token),
        expires_at=now + timedelta(seconds=settings.refresh_ttl_seconds),
    )
    db.add(successor)
    await db.flush()
    row.revoked_at, row.replaced_by_id = now, successor.id
    await db.commit()
    set_cookies(response, token, settings)
    return TokenOut(
        access_token=access_token(user.id, settings), expires_in=settings.access_ttl_seconds
    )


async def logout(db: AsyncSession, request: Request, response: Response):
    check_csrf(request)
    raw = request.cookies.get("pawly_refresh")
    if raw:
        row = await locked_refresh(db, raw)
        await db.execute(
            update(RefreshSession)
            .where(
                RefreshSession.family_id == row.family_id,
                RefreshSession.revoked_at.is_(None),
            )
            .values(revoked_at=utcnow())
        )
        await db.commit()
    response.delete_cookie(
        "pawly_refresh",
        path=COOKIE_PATH,
        secure=request.app.state.settings.cookie_secure,
        httponly=True,
        samesite="lax",
    )
