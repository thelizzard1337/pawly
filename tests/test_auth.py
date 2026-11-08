import asyncio
from datetime import timedelta
from uuid import UUID

import jwt
import pytest
from sqlalchemy import select

from app.core.security import utcnow
from app.db.models import RefreshSession, User

pytestmark = pytest.mark.integration


async def test_registration_login_and_private_profile(client, app, actor):
    credentials = {
        "email": "Test.User@EXAMPLE.com",
        "password": "Password-12345",
        "display_name": "Test",
    }
    bad = await client.post("/api/v1/auth/register", json={**credentials, "is_admin": True})
    assert bad.status_code == 422
    response = await client.post("/api/v1/auth/register", json=credentials)
    assert response.status_code == 201
    assert response.json()["email"] == "test.user@example.com"
    assert response.json()["is_admin"] is False
    assert "password_hash" not in response.json()
    response = await client.post("/api/v1/auth/register", json=credentials)
    assert response.status_code == 409
    owner = await actor()
    response = await client.patch(
        "/api/v1/users/me",
        headers=owner["headers"],
        json={"phone": "+77001234567", "preferred_channel": "phone"},
    )
    assert response.status_code == 200
    assert response.json()["phone"] == "+77001234567"
    response = await client.patch(
        "/api/v1/users/me", headers=owner["headers"], json={"is_admin": True}
    )
    assert response.status_code == 422
    assert (await client.get("/api/v1/users/me")).status_code == 401
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": owner["credentials"]["email"],
            "password": "wrong-password",
        },
    )
    assert response.status_code == 401
    async with app.state.sessions() as db:
        user = await db.get(User, UUID(owner["id"]))
        assert user.password_hash.startswith("$argon2")
        user.is_active = False
        await db.commit()
    assert (await client.get("/api/v1/users/me", headers=owner["headers"])).status_code == 401


async def test_access_expiry_and_current_admin_role(client, app, actor):
    owner = await actor()
    settings = app.state.settings
    expired = jwt.encode(
        {
            "sub": owner["id"],
            "iat": utcnow() - timedelta(hours=1),
            "exp": utcnow() - timedelta(seconds=1),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    response = await client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


async def test_refresh_rotation_reuse_revokes_family_and_csrf(client, app, actor):
    owner = await actor()
    assert (await client.post("/api/v1/auth/refresh")).status_code == 403
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    headers = {"X-CSRF-Token": csrf, "Origin": "https://evil.example.com"}
    assert (await client.post("/api/v1/auth/refresh", headers=headers)).status_code == 403
    headers["Origin"] = "http://localhost:8000"
    response = await client.post("/api/v1/auth/refresh", headers=headers)
    assert response.status_code == 200
    successor = client.cookies.get("pawly_refresh")
    assert successor != owner["refresh"]
    refresh_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in refresh_cookie and "SameSite=lax" in refresh_cookie
    replay = {**headers, "Cookie": f"pawly_refresh={owner['refresh']}; pawly_csrf={csrf}"}
    response = await client.post("/api/v1/auth/refresh", headers=replay)
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "refresh_reuse_detected"
    response = await client.post(
        "/api/v1/auth/refresh",
        headers={
            **headers,
            "Cookie": f"pawly_refresh={successor}; pawly_csrf={csrf}",
        },
    )
    assert response.status_code == 401
    async with app.state.sessions() as db:
        rows = list(
            await db.scalars(
                select(RefreshSession).where(
                    RefreshSession.user_id == UUID(owner["id"]),
                )
            )
        )
        assert rows and all(row.revoked_at is not None for row in rows)
        assert all(row.token_hash != owner["refresh"] for row in rows)


async def test_concurrent_refresh_and_logout(client, actor):
    owner = await actor()
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    headers = {
        "X-CSRF-Token": csrf,
        "Cookie": f"pawly_refresh={owner['refresh']}; pawly_csrf={csrf}",
    }
    responses = await asyncio.gather(
        *[client.post("/api/v1/auth/refresh", headers=headers) for _ in range(2)]
    )
    assert sorted(r.status_code for r in responses) == [200, 401]
    owner = await actor("logout")
    response = await client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 204
    response = await client.post(
        "/api/v1/auth/refresh",
        headers={
            "X-CSRF-Token": csrf,
            "Cookie": f"pawly_refresh={owner['refresh']}; pawly_csrf={csrf}",
        },
    )
    assert response.status_code == 401


async def test_refresh_expiry(client, app, actor):
    owner = await actor()
    async with app.state.sessions() as db:
        row = await db.scalar(
            select(RefreshSession).where(
                RefreshSession.user_id == UUID(owner["id"]),
            )
        )
        row.expires_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    csrf = (await client.get("/api/v1/auth/csrf")).json()["csrf_token"]
    response = await client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "refresh_expired"
