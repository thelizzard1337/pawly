from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.core.config import SCHEMA_REVISION
from app.db.models import Listing, MediaAsset
from tests.helpers import listing

pytestmark = pytest.mark.integration


async def test_database_rejects_invalid_states(client, app, actor, image):
    owner = await actor()
    uid, _ = await listing(client, owner, image, publish=False, photo=False)
    async with app.state.sessions() as db:
        invalid = [
            Listing(author_id=UUID(owner["id"]), kind="adoption", status="draft"),
            Listing(
                author_id=UUID(owner["id"]),
                kind="found",
                status="active",
                species="cat",
                location="SRID=4326;POINT(76 43)",
                description=None,
            ),
            MediaAsset(
                owner_id=UUID(owner["id"]),
                listing_id=UUID(uid),
                purpose="listing",
                status="pending",
                position=None,
                storage_key=f"{uuid4()}.jpg",
            ),
            MediaAsset(
                owner_id=UUID(owner["id"]),
                purpose="search",
                status="ready",
                sha256="a" * 64,
                content_type="image/jpeg",
                size_bytes=1,
                width=None,
                height=1,
                storage_key=f"{uuid4()}.jpg",
            ),
        ]
        for row in invalid:
            with pytest.raises(IntegrityError):
                async with db.begin_nested():
                    db.add(row)
                    await db.flush()


async def test_readiness_schema_and_storage_permissions(client, app):
    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/openapi.json")).status_code == 200
    assert (await client.get("/health/ready")).status_code == 200
    async with app.state.sessions() as db:
        await db.execute(text("UPDATE alembic_version SET version_num='wrong_revision'"))
        await db.commit()
    try:
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 503
    finally:
        async with app.state.sessions() as db:
            await db.execute(
                text("UPDATE alembic_version SET version_num=:revision"),
                {"revision": SCHEMA_REVISION},
            )
            await db.commit()


async def test_failed_storage_write_is_recoverable(client, app, actor, image, monkeypatch):
    owner = await actor()
    uid, _ = await listing(client, owner, image, publish=False, photo=False)

    def broken_write(*args):
        raise OSError("Injected storage failure")

    monkeypatch.setattr(app.state.storage, "write", broken_write)
    with pytest.raises(OSError, match="Injected storage failure"):
        await client.post(
            f"/api/v1/listings/{uid}/photos",
            headers=owner["headers"],
            files={"file": ("cat.png", image, "image/png")},
        )
    async with app.state.sessions() as db:
        photo = await db.scalar(select(MediaAsset).where(MediaAsset.listing_id == UUID(uid)))
        assert photo.status == "failed"
        assert not app.state.storage.path(photo.storage_key).exists()
