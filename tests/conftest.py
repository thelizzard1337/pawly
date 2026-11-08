import io
import os
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from PIL import Image
from sqlalchemy import text

from app.core.config import Settings
from app.main import create_app


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PAWLY_INTEGRATION") != "1":
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use the isolated Docker test service"))


@pytest.fixture
def image():
    output = io.BytesIO()
    Image.new("RGB", (80, 60), "orange").save(output, "PNG")
    return output.getvalue()


@pytest_asyncio.fixture
async def app(tmp_path):
    settings = Settings(media_root=tmp_path / "media")
    if settings.app_env != "test" or not settings.db_name.endswith("_test"):
        pytest.fail("Integration tests require APP_ENV=test and an isolated *_test database")
    instance = create_app(settings)
    async with LifespanManager(instance):
        async with instance.state.sessions() as db:
            actual = await db.scalar(text("SELECT current_database()"))
            assert actual == settings.db_name and actual.endswith("_test")
            await db.execute(
                text(
                    "TRUNCATE photo_search_results, photo_searches, contact_requests, "
                    "adoption_requests, "
                    "media_assets, listings, shelter_members, shelters, refresh_sessions, "
                    "user_contacts, users CASCADE"
                )
            )
            await db.commit()
        yield instance


@pytest_asyncio.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as value:
        yield value


@pytest.fixture
def actor(client):
    async def create(name="user"):
        credentials = {"email": f"{name}-{uuid4().hex}@example.com", "password": "Password-12345"}
        response = await client.post(
            "/api/v1/auth/register",
            json={
                **credentials,
                "display_name": name,
            },
        )
        assert response.status_code == 201, response.text
        uid = response.json()["id"]
        response = await client.post("/api/v1/auth/login", json=credentials)
        assert response.status_code == 200, response.text
        return {
            "id": uid,
            "headers": {"Authorization": f"Bearer {response.json()['access_token']}"},
            "credentials": credentials,
            "refresh": client.cookies.get("pawly_refresh"),
        }

    return create
