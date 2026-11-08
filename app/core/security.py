import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from pwdlib import PasswordHash

from app.core.config import Settings

password_hasher = PasswordHash.recommended()


def utcnow() -> datetime:
    return datetime.now(UTC)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(48)


def access_token(user_id: UUID, settings: Settings) -> str:
    now = utcnow()
    return jwt.encode(
        {
            "sub": str(user_id),
            "iat": now,
            "exp": now + timedelta(seconds=settings.access_ttl_seconds),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


def decode_access(token: str, settings: Settings) -> UUID:
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=["HS256"],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        options={"require": ["exp", "sub", "iss", "aud", "iat"]},
    )
    return UUID(payload["sub"])
