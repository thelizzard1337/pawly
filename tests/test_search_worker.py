import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from app.core.security import utcnow
from app.db.models import MediaAsset, PhotoSearch
from app.integrations.cv.matcher import MockPhotoMatcher
from app.modules.media.service import cleanup
from app.modules.photo_search import jobs
from tests.helpers import grant_admin, listing, search

pytestmark = pytest.mark.integration


async def complete(app):
    job = await jobs.claim(app.state.sessions, app.state.settings)
    assert job is not None
    matches = await jobs.execute(app.state.sessions, MockPhotoMatcher(), job)
    assert await jobs.finalize(app.state.sessions, job, matches)
    return job


async def test_real_candidates_scope_and_post_search_hiding(client, app, actor, image):
    owner, searcher, admin = await actor(), await actor("searcher"), await actor("admin")
    await grant_admin(app, admin)
    uid, mid = await listing(client, owner, image)
    await listing(client, owner, image, kind="lost")
    await listing(client, owner, image, species="dog")
    await listing(client, owner, image, publish=False)
    sid, qid = await search(
        client, searcher, image, latitude=43.2389, longitude=76.8897, radius_m=1000
    )
    assert (await client.get(f"/api/v1/media/{qid}")).status_code == 404
    assert (await client.get(f"/api/v1/media/{qid}", headers=owner["headers"])).status_code == 404
    assert (
        await client.get(f"/api/v1/photo-searches/{sid}", headers=owner["headers"])
    ).status_code == 404
    assert (
        await client.post(
            "/api/v1/photo-searches",
            headers=owner["headers"],
            json={
                "query_media_id": qid,
                "target_kind": "found",
            },
        )
    ).status_code == 404
    await complete(app)
    response = await client.get(f"/api/v1/photo-searches/{sid}", headers=searcher["headers"])
    result = response.json()
    assert result["status"] == "completed" and result["is_mock"] is True
    assert result["provider"] == "mock-v1" and result["model_version"] is None
    assert len(result["results"]) == 1
    assert result["results"][0]["listing"]["id"] == uid
    assert result["results"][0]["matched_media_id"] == mid
    assert result["results"][0]["similarity_score"] is None
    await client.post(f"/api/v1/admin/listings/{uid}/hide", headers=admin["headers"])
    response = await client.get(f"/api/v1/photo-searches/{sid}", headers=searcher["headers"])
    assert response.json()["results"] == []
    sid2, _ = await search(client, searcher, image)
    await complete(app)
    response = await client.get(f"/api/v1/photo-searches/{sid2}", headers=searcher["headers"])
    assert response.json()["status"] == "completed" and response.json()["results"] == []


async def test_idempotency_concurrency_conflict_and_expiry(client, app, actor, image):
    owner = await actor()
    response = await client.post(
        "/api/v1/photo-searches/images",
        headers=owner["headers"],
        files={"file": ("query.png", image, "image/png")},
    )
    qid = response.json()["id"]
    headers = {**owner["headers"], "Idempotency-Key": "same-key"}
    payload = {"query_media_id": qid, "target_kind": "found", "species": "cat", "limit": 5}
    responses = await asyncio.gather(
        *[client.post("/api/v1/photo-searches", headers=headers, json=payload) for _ in range(4)]
    )
    assert all(r.status_code == 202 for r in responses), [r.text for r in responses]
    assert len({r.json()["id"] for r in responses}) == 1
    sid = responses[0].json()["id"]
    assert responses[0].headers["location"] == f"/api/v1/photo-searches/{sid}"
    response = await client.post(
        "/api/v1/photo-searches", headers=headers, json={**payload, "limit": 1}
    )
    assert response.status_code == 409
    async with app.state.sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        row.expires_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    assert (
        await client.get(f"/api/v1/photo-searches/{sid}", headers=owner["headers"])
    ).status_code == 410


async def test_concurrent_claim_lease_fencing_recovery_and_retry(client, app, actor, image):
    owner = await actor()
    sid, _ = await search(client, owner, image)
    sessions, settings = app.state.sessions, app.state.settings
    claimed = await asyncio.gather(*[jobs.claim(sessions, settings) for _ in range(4)])
    assert sum(job is not None for job in claimed) == 1
    original = next(job for job in claimed if job)
    async with sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        assert row.attempts == 1
        row.lease_until = utcnow() - timedelta(seconds=1)
        await db.commit()
    assert await jobs.finalize(sessions, original, []) is False
    await jobs.recover(sessions, settings)
    async with sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        assert row.status == "queued"
        row.available_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    replacement = await jobs.claim(sessions, settings)
    assert replacement.lease_token != original.lease_token
    assert await jobs.finalize(sessions, original, []) is False
    assert await jobs.failed(sessions, settings, replacement, "test_transient")
    async with sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        assert row.status == "queued" and row.attempts == 2
        assert row.available_at > utcnow()
        row.available_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    last = await jobs.claim(sessions, settings)
    assert await jobs.failed(sessions, settings, last, "test_transient")
    async with sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        assert row.status == "failed" and row.attempts == 3
    assert await jobs.claim(sessions, settings) is None


async def test_cleanup_preserves_live_search_and_recovers_orphans(client, app, actor, image):
    owner = await actor()
    sid, qid = await search(client, owner, image)
    storage = app.state.storage
    async with app.state.sessions() as db:
        media = await db.get(MediaAsset, UUID(qid))
        media.expires_at = utcnow() - timedelta(seconds=1)
        key = media.storage_key
        await db.commit()
    await cleanup(app.state.sessions, storage)
    assert storage.path(key).exists()  # queued job protects even an expired query image.
    await complete(app)
    await cleanup(app.state.sessions, storage)
    await cleanup(app.state.sessions, storage)
    assert not storage.path(key).exists()
    assert (await client.get(f"/api/v1/media/{qid}", headers=owner["headers"])).status_code == 410
    import os

    orphan = f"{uuid4()}.jpg"
    storage.write(orphan, b"unreferenced")
    before = (utcnow() - timedelta(hours=2)).timestamp()
    os.utime(storage.path(orphan), (before, before))
    result = await cleanup(app.state.sessions, storage)
    assert result["orphans"] == 1 and not storage.path(orphan).exists()


async def test_recovery_exhausted_and_expired_jobs(client, app, actor, image):
    owner = await actor()
    sid, _ = await search(client, owner, image)
    await jobs.claim(app.state.sessions, app.state.settings)
    async with app.state.sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        row.attempts = app.state.settings.worker_max_attempts
        row.lease_until = utcnow() - timedelta(seconds=1)
        await db.commit()
    await jobs.recover(app.state.sessions, app.state.settings)
    async with app.state.sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        assert row.status == "failed" and row.error_code == "attempts_exhausted"


async def test_worker_timeout_is_retried(client, app, actor, image):
    from app.worker import process_job

    class SlowMatcher:
        async def match(self, query_hash, candidates, limit):
            await asyncio.sleep(1)
            return []

    owner = await actor()
    sid, _ = await search(client, owner, image)
    settings = app.state.settings.model_copy(update={"worker_timeout_seconds": 0.01})
    job = await jobs.claim(app.state.sessions, settings)
    assert await process_job(app.state.sessions, settings, SlowMatcher(), job)
    async with app.state.sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        assert row.status == "queued" and row.error_code == "matcher_timeout"
        assert row.lease_token is None
