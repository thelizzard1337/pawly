"""Indexing queue + real VectorPhotoMatcher on PostgreSQL/PostGIS/pgvector.

The extractor is a deterministic test double (embedding = mean crop colour), so ordering can be
asserted. It replaces only the neural network; SQL, pgvector, queue, fencing and API are real.
"""

import io
from datetime import timedelta
from uuid import UUID

import pytest
from PIL import Image
from sqlalchemy import func, select, text

from app.core.security import utcnow
from app.cv import spec
from app.cv.pipeline import PipelineSpec
from app.db.models import CVWorkerStatus, MediaAsset, MediaFeature, PhotoSearch, PhotoSearchResult
from app.integrations.cv.runtime import CVRuntime
from app.integrations.cv.vector_matcher import VectorPhotoMatcher
from app.modules.cv_index import indexing
from app.modules.cv_index import repository as index_repo
from app.modules.photo_search import jobs
from tests.helpers import grant_admin, listing

np = pytest.importorskip("numpy")
image_ops = pytest.importorskip("app.cv.image_ops")
pytestmark = pytest.mark.integration

ORANGE, RED, GREEN, BLUE = (255, 165, 0), (200, 30, 30), (0, 160, 0), (20, 40, 230)


def png(color, size=(96, 96), second=None):
    image = Image.new("RGB", size, color)
    if second is not None:
        image.paste(Image.new("RGB", (size[0] // 2, size[1]), second), (size[0] // 2, 0))
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def json_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in json_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in json_keys(v)}
    return set()


class ColorExtractor:
    """Test double: blue crops are "no dog"; the embedding encodes the mean crop colour."""

    def __init__(self, pipeline=None, fail=False):
        self.pipeline = pipeline or PipelineSpec(
            "pretrained_features", spec.BACKBONES["efficientnet_b0"], 0.5, None, "embedding-v1"
        )
        self.fail, self.calls = fail, 0

    def extract(self, raw, bbox):
        from app.cv.extractor import Extraction

        self.calls += 1
        if self.fail:
            raise RuntimeError("injected extraction failure")
        try:
            image = image_ops.load_canonical(raw)
            crop, crop_spec = image_ops.apply_crop(image, bbox)
        except image_ops.ImageRejected as exc:
            return Extraction(None, rejected=exc.reason)
        mean = np.asarray(crop, dtype=np.float64).reshape(-1, 3).mean(0) / 255
        if mean[2] > 0.6 and mean[0] < 0.3:
            return Extraction(crop_spec, rejected=spec.NO_DOG)
        vector = np.zeros(self.pipeline.backbone.dimension)
        vector[:3], vector[3] = mean + 0.05, 0.05
        vector /= np.linalg.norm(vector)
        descriptor = image_ops.rgb_descriptor(crop)
        return Extraction(
            crop_spec,
            embedding=vector.tolist(),
            rgb_histogram=descriptor.histogram,
            dominant_colors=descriptor.dominant,
            timings_ms={"total": 1.0},
        )


@pytest.fixture
def runtime():
    value = CVRuntime(ColorExtractor())
    yield value
    value.close()


async def drain_index(app, runtime):
    sessions, settings = app.state.sessions, app.state.settings
    await index_repo.schedule_missing(sessions, runtime.pipeline, 100)
    while job := await index_repo.claim(sessions, settings, runtime.pipeline.pipeline_id):
        await indexing.process(sessions, settings, app.state.storage, runtime, job)


async def run_search(app, runtime):
    job = await jobs.claim(app.state.sessions, app.state.settings)
    assert job is not None
    matcher = VectorPhotoMatcher(runtime, app.state.storage, app.state.settings)
    outcome = await jobs.execute(app.state.sessions, matcher, job)
    assert await jobs.finalize(app.state.sessions, job, outcome)
    return outcome


async def dog_listing(client, actor, paint, **fields):
    data = {"species": "dog", "breed": "unknown", "color": "рыжий", **fields}
    return await listing(client, actor, png(paint), **data)


async def v2_search(client, actor, color, **overrides):
    response = await client.post(
        "/api/v1/photo-searches/images",
        headers=actor["headers"],
        files={"file": ("q.png", png(color), "image/png")},
    )
    assert response.status_code == 201, response.text
    query_id = response.json()["id"]
    payload = {
        "query_media_id": query_id,
        "breed_text": "Немецкая овчарка + терьер",
        "coat_color_text": "рыжий с белым",
        "target_kind": "found",
        "limit": 10,
        **overrides,
    }
    response = await client.post("/api/v2/photo-searches", headers=actor["headers"], json=payload)
    return response, query_id


async def test_extensions_are_installed(app):
    async with app.state.sessions() as db:
        rows = dict((await db.execute(text("SELECT extname, extversion FROM pg_extension"))).all())
    assert "postgis" in rows and "vector" in rows


async def test_index_and_v2_search_end_to_end(client, app, actor, runtime):
    owner, searcher = await actor("owner"), await actor("searcher")
    mixed, _ = await dog_listing(client, owner, ORANGE, breed="немецкая овчарка + терьер")
    unknown, _ = await dog_listing(client, owner, RED, breed=None, color=None)
    far, _ = await dog_listing(client, owner, GREEN, breed="лабрадор", color="чёрный")
    await drain_index(app, runtime)
    async with app.state.sessions() as db:
        rows = (await db.scalars(select(MediaFeature))).all()
        assert {row.status for row in rows} == {"indexed"} and len(rows) == 3
        snapshot = {row.user_breed_text for row in rows}
        assert snapshot == {"немецкая овчарка + терьер", None, "лабрадор"}  # raw user text
        assert all(row.breed_prediction is None for row in rows)  # no trained heads
        assert all(row.embedding_space_id == runtime.pipeline.embedding_space_id for row in rows)
    response, _ = await v2_search(client, searcher, ORANGE)
    assert response.status_code == 202, response.text
    sid = response.json()["id"]
    assert response.headers["location"] == f"/api/v2/photo-searches/{sid}"
    await run_search(app, runtime)
    body = (await client.get(f"/api/v2/photo-searches/{sid}", headers=searcher["headers"])).json()
    assert body["status"] == "completed" and body["outcome"] == "candidates"
    assert body["provider"] == "pawly-cv" and body["is_mock"] is False
    assert body["cv_mode"] == "pretrained_features"
    assert body["embedding_space_id"] == runtime.pipeline.embedding_space_id
    assert body["query"]["breed_text"] == "Немецкая овчарка + терьер"  # verbatim
    # Unknown/mixed breeds are part of the corpus; text never filters the visual ranking.
    assert [r["listing"]["id"] for r in body["results"]] == [mixed, unknown, far]
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    keys = json_keys(body)
    assert not {k for k in keys if "score" in k or "similar" in k or "confidence" in k}
    legacy = await client.get(f"/api/v1/photo-searches/{sid}", headers=searcher["headers"])
    assert all(r["similarity_score"] is None for r in legacy.json()["results"])
    assert legacy.json()["is_mock"] is False
    async with app.state.sessions() as db:
        row = await db.get(PhotoSearch, UUID(sid))
        assert row.breed_text_norm == "немецкая овчарка + терьер"
        assert row.coat_color_text == "рыжий с белым" and row.api_version == "v2"
        scores = (await db.scalars(select(PhotoSearchResult.ranking_score))).all()
        assert all(score is not None for score in scores)  # internal only


async def test_multiple_photos_unique_listing_and_best_photo(client, app, actor, runtime):
    owner, searcher = await actor(), await actor()
    uid, green_photo = await dog_listing(client, owner, GREEN, publish=False)
    response = await client.post(
        f"/api/v1/listings/{uid}/photos",
        headers=owner["headers"],
        files={"file": ("o.png", png(ORANGE), "image/png")},
    )
    orange_photo = response.json()["id"]
    assert (
        await client.post(f"/api/v1/listings/{uid}/publish", headers=owner["headers"])
    ).status_code == 200
    other, _ = await dog_listing(client, owner, RED)
    await drain_index(app, runtime)
    await v2_search(client, searcher, ORANGE)
    outcome = await run_search(app, runtime)
    assert [m.listing_id for m in outcome.matches] == [UUID(uid), UUID(other)]
    assert outcome.matches[0].media_id == UUID(orange_photo) != UUID(green_photo)
    assert outcome.meta["eligible_listings"] == 2


async def test_visibility_rechecked_and_scope_filters(client, app, actor, runtime):
    owner, searcher, admin = await actor(), await actor(), await actor("admin")
    await grant_admin(app, admin)
    hidden, _ = await dog_listing(client, owner, ORANGE)
    archived, _ = await dog_listing(client, owner, ORANGE)
    lost, _ = await dog_listing(client, owner, ORANGE, kind="lost")
    visible, _ = await dog_listing(client, owner, RED)
    deleted, deleted_photo = await dog_listing(client, owner, ORANGE, publish=False)
    await drain_index(app, runtime)
    response, _ = await v2_search(client, searcher, ORANGE)
    sid = response.json()["id"]
    await client.post(f"/api/v1/admin/listings/{hidden}/hide", headers=admin["headers"])
    await client.delete(f"/api/v1/listings/{archived}", headers=owner["headers"])
    outcome = await run_search(app, runtime)
    ids = [str(m.listing_id) for m in outcome.matches]
    assert ids == [visible]  # hidden, archived, draft and other-kind listings excluded
    assert lost not in ids and deleted not in ids
    # Visibility is rechecked on read even after results were stored.
    await client.post(f"/api/v1/admin/listings/{visible}/hide", headers=admin["headers"])
    body = (await client.get(f"/api/v2/photo-searches/{sid}", headers=searcher["headers"])).json()
    assert body["results"] == []
    assert deleted_photo


async def test_embedding_spaces_and_pipelines_are_never_mixed(client, app, actor, runtime):
    owner, searcher = await actor(), await actor()
    uid, _ = await dog_listing(client, owner, ORANGE)
    resnet = CVRuntime(
        ColorExtractor(
            PipelineSpec(
                "pretrained_features", spec.BACKBONES["resnet50"], 0.5, None, "embedding-v1"
            )
        )
    )
    try:
        await drain_index(app, resnet)  # 2048-d rows in another embedding space
    finally:
        resnet.close()
    async with app.state.sessions() as db:
        dims = (await db.scalars(select(MediaFeature.dimension))).all()
        assert dims == [2048]
    await v2_search(client, searcher, ORANGE)
    outcome = await run_search(app, runtime)  # efficientnet space: corpus is empty
    assert outcome.outcome == "candidates" and outcome.matches == []
    assert outcome.meta["eligible_listings"] == 0
    await drain_index(app, runtime)
    await v2_search(client, searcher, ORANGE)
    outcome = await run_search(app, runtime)
    assert [str(m.listing_id) for m in outcome.matches] == [uid]
    async with app.state.sessions() as db:
        assert await db.scalar(select(func.count()).select_from(MediaFeature)) == 4  # 2 + 2 query


async def test_empty_corpus_and_no_dog_rejection(client, app, actor, runtime):
    searcher = await actor()
    response, _ = await v2_search(client, searcher, ORANGE)
    sid = response.json()["id"]
    await run_search(app, runtime)
    body = (await client.get(f"/api/v2/photo-searches/{sid}", headers=searcher["headers"])).json()
    assert body["status"] == "completed" and body["outcome"] == "candidates"
    assert body["results"] == []
    response, _ = await v2_search(client, searcher, BLUE)
    sid = response.json()["id"]
    await run_search(app, runtime)
    body = (await client.get(f"/api/v2/photo-searches/{sid}", headers=searcher["headers"])).json()
    assert body["status"] == "completed" and body["outcome"] == "rejected"
    assert body["reason_code"] == "no_dog" and body["results"] == []
    async with app.state.sessions() as db:
        assert (await db.get(PhotoSearch, UUID(sid))).attempts == 1  # domain reject: no retry


async def test_query_features_are_reused_and_crop_is_versioned(client, app, actor, runtime):
    owner, searcher = await actor(), await actor()
    await dog_listing(client, owner, ORANGE)
    await drain_index(app, runtime)
    calls = runtime.extractor.calls
    response, query_id = await v2_search(client, searcher, ORANGE)
    await run_search(app, runtime)
    response = await client.post(
        "/api/v2/photo-searches",
        headers=searcher["headers"],
        json={
            "query_media_id": query_id,
            "breed_text": "x",
            "coat_color_text": "y",
            "target_kind": "found",
        },
    )
    await run_search(app, runtime)
    assert runtime.extractor.calls == calls + 1  # same media + crop + pipeline -> reused
    response, _ = await v2_search(client, searcher, ORANGE)
    bbox = {"x_min": 0, "y_min": 0, "x_max": 0.5, "y_max": 1}
    await client.post(
        "/api/v2/photo-searches",
        headers=searcher["headers"],
        json={
            "query_media_id": query_id,
            "breed_text": "x",
            "coat_color_text": "y",
            "target_kind": "found",
            "bbox": bbox,
        },
    )
    async with app.state.sessions() as db:
        assert (
            await db.scalar(select(PhotoSearch.crop_bbox).where(PhotoSearch.crop_bbox.is_not(None)))
        ) == [0.0, 0.0, 0.5, 1.0]


async def test_v2_validation_ownership_and_crop(client, app, actor, image):
    owner, stranger = await actor(), await actor()
    response = await client.post(
        "/api/v1/photo-searches/images",
        headers=owner["headers"],
        files={"file": ("q.png", png(ORANGE), "image/png")},
    )
    qid = response.json()["id"]
    base = {
        "query_media_id": qid,
        "breed_text": "метис",
        "coat_color_text": "рыжий",
        "target_kind": "found",
    }
    url = "/api/v2/photo-searches"
    assert (await client.post(url, headers=stranger["headers"], json=base)).status_code == 404
    for patch in (
        {"breed_text": "   "},
        {"coat_color_text": ""},
        {"breed_text": None},
        {"limit": 51},
        {"bbox": {"x_min": 0.5, "y_min": 0, "x_max": 0.4, "y_max": 1}},
        {"bbox": {"x_min": 0, "y_min": 0, "x_max": 1.5, "y_max": 1}},
    ):
        response = await client.post(url, headers=owner["headers"], json={**base, **patch})
        assert response.status_code == 422, (patch, response.text)
    tiny = {"x_min": 0, "y_min": 0, "x_max": 0.1, "y_max": 0.1}  # < 32 px on a 96 px image
    response = await client.post(url, headers=owner["headers"], json={**base, "bbox": tiny})
    assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_crop"
    ok = await client.post(url, headers=owner["headers"], json={**base, "limit": 50})
    assert ok.status_code == 202
    # Legacy v1 contract is unchanged: limit 1..5, no breed/colour fields.
    legacy = {"query_media_id": qid, "target_kind": "found"}
    assert (
        await client.post(
            "/api/v1/photo-searches", headers=owner["headers"], json={**legacy, "limit": 6}
        )
    ).status_code == 422
    assert (
        await client.post(
            "/api/v1/photo-searches", headers=owner["headers"], json={**legacy, "breed_text": "x"}
        )
    ).status_code == 422
    assert (
        await client.post("/api/v1/photo-searches", headers=owner["headers"], json=legacy)
    ).status_code == 202


async def test_indexing_idempotency_fencing_retry_and_rejection(client, app, actor, runtime):
    sessions, settings = app.state.sessions, app.state.settings
    pipeline = runtime.pipeline
    owner = await actor()
    await dog_listing(client, owner, ORANGE)
    await dog_listing(client, owner, BLUE)  # domain rejection: no dog
    assert await index_repo.schedule_missing(sessions, pipeline, 100) == 2
    assert await index_repo.schedule_missing(sessions, pipeline, 100) == 0  # idempotent
    first = await index_repo.claim(sessions, settings, pipeline.pipeline_id)
    async with sessions() as db:
        row = await db.get(MediaFeature, first.id)
        row.lease_until = utcnow() - timedelta(seconds=1)
        await db.commit()
    stale = runtime.extractor.extract(b"", None)
    assert await index_repo.complete(sessions, first, stale, (None, None)) is False  # fenced
    await index_repo.recover(sessions, settings, pipeline.pipeline_id)
    async with sessions() as db:
        row = await db.get(MediaFeature, first.id)
        assert row.status == "pending" and row.attempts == 1
        row.available_at = utcnow()
        await db.commit()
    failing = CVRuntime(ColorExtractor(fail=True))
    try:
        job = await index_repo.claim(sessions, settings, pipeline.pipeline_id)
        await indexing.process(sessions, settings, app.state.storage, failing, job)
    finally:
        failing.close()
    async with sessions() as db:
        row = await db.get(MediaFeature, job.id)
        assert row.status == "pending" and row.error_code == "extraction_failed"
        assert row.available_at > utcnow()
        await db.execute(
            text("UPDATE media_features SET available_at = now() - interval '1 second'")
        )
        await db.commit()
    while job := await index_repo.claim(sessions, settings, pipeline.pipeline_id):
        await indexing.process(sessions, settings, app.state.storage, runtime, job)
    async with sessions() as db:
        rows = (await db.scalars(select(MediaFeature).order_by(MediaFeature.created_at))).all()
        statuses = sorted((row.status, row.reason_code) for row in rows)
        assert statuses == [("indexed", None), ("rejected", "no_dog")]
    # Rejected rows are final: neither recovery nor the scheduler re-queues them.
    await index_repo.recover(sessions, settings, pipeline.pipeline_id)
    assert await index_repo.schedule_missing(sessions, pipeline, 100) == 0
    assert await index_repo.claim(sessions, settings, pipeline.pipeline_id) is None


async def test_crop_change_supersedes_and_revert_revives(client, app, actor, runtime):
    owner = await actor()
    uid, photo = await listing(
        client, owner, png(ORANGE, size=(200, 100), second=GREEN), species="dog"
    )
    await drain_index(app, runtime)
    url = f"/api/v1/listings/{uid}/photos/{photo}/crop"
    right = {"bbox": {"x_min": 0.5, "y_min": 0, "x_max": 1, "y_max": 1}}
    response = await client.put(url, headers=owner["headers"], json=right)
    assert response.status_code == 200 and response.json()["crop_bbox"] == [0.5, 0, 1, 1]
    bad = {"bbox": {"x_min": 0, "y_min": 0, "x_max": 0.05, "y_max": 1}}
    assert (await client.put(url, headers=owner["headers"], json=bad)).status_code == 422
    await drain_index(app, runtime)
    async with app.state.sessions() as db:
        rows = (await db.scalars(select(MediaFeature).order_by(MediaFeature.created_at))).all()
        assert len(rows) == 2 and rows[0].superseded_at is not None
        assert rows[1].superseded_at is None and rows[1].crop_box == [100, 0, 200, 100]
        assert rows[1].crop_hash != rows[0].crop_hash
    assert (await client.put(url, headers=owner["headers"], json={"bbox": None})).status_code == 200
    calls = runtime.extractor.calls
    await drain_index(app, runtime)
    assert runtime.extractor.calls == calls  # original crop revived, not recomputed
    async with app.state.sessions() as db:
        current = (
            await db.scalars(select(MediaFeature).where(MediaFeature.superseded_at.is_(None)))
        ).all()
        assert len(current) == 1 and current[0].crop_bbox is None
    stranger = await actor()
    assert (await client.put(url, headers=stranger["headers"], json=right)).status_code == 404


async def test_upload_with_bbox_and_cleanup_removes_features(client, app, actor, runtime):
    from app.modules.media.service import cleanup

    owner = await actor()
    uid, _ = await listing(client, owner, png(ORANGE), species="dog", photo=False, publish=False)
    response = await client.post(
        f"/api/v1/listings/{uid}/photos",
        headers=owner["headers"],
        data={"bbox": "0,0,0.1,0.1"},
        files={"file": ("o.png", png(ORANGE), "image/png")},
    )
    assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_crop"
    response = await client.post(
        f"/api/v1/listings/{uid}/photos",
        headers=owner["headers"],
        data={"bbox": "0.1,0.1,0.9,0.9"},
        files={"file": ("o.png", png(ORANGE), "image/png")},
    )
    assert response.status_code == 201 and response.json()["crop_bbox"] == [0.1, 0.1, 0.9, 0.9]
    photo = response.json()["id"]
    await drain_index(app, runtime)
    async with app.state.sessions() as db:
        row = await db.scalar(select(MediaFeature).where(MediaFeature.media_id == UUID(photo)))
        assert row.crop_box == [9, 9, 87, 87] and row.status == "indexed"
    assert (
        await client.delete(f"/api/v1/listings/{uid}/photos/{photo}", headers=owner["headers"])
    ).status_code == 204
    result = await cleanup(app.state.sessions, app.state.storage)
    assert result["deleted_features"] == 1
    async with app.state.sessions() as db:
        assert await db.scalar(select(func.count()).select_from(MediaFeature)) == 0
        assert (await db.get(MediaAsset, UUID(photo))).status == "deleted"


async def test_cv_status_and_ml_health(client, app, runtime):
    assert (await client.get("/health/ml")).status_code == 503  # no worker heartbeat
    assert (await client.get("/health/ready")).status_code == 200  # CRUD readiness unaffected
    async with app.state.sessions() as db:
        db.add(
            CVWorkerStatus(
                worker_id="test-worker",
                cv_mode=app.state.settings.cv_mode,
                provider="mock-v1",
                state="ready",
                started_at=utcnow(),
                heartbeat_at=utcnow(),
            )
        )
        await db.commit()
    assert (await client.get("/health/ml")).status_code == 200
    body = (await client.get("/api/v1/cv/status")).json()
    assert body["ready"] is True and body["trained_heads"]["available"] is False
    assert "test-worker" not in str(body)  # no host/process identifiers
