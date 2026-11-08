"""HTTP-only smoke flow against an actual API and an independently running worker.

In a real CV mode it is a plumbing smoke: the gallery listing and the query use the SAME
public-domain dog photo, so finding it checks wiring (upload -> crop -> gate -> embedding ->
pgvector -> API), not generalisation or re-identification.
"""

import asyncio
import io
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx
from PIL import Image

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "dog_beagle_pd.jpg"


def solid_png(color=(120, 80, 40), size=(64, 48)) -> bytes:
    raw = io.BytesIO()
    Image.new("RGB", size, color).save(raw, "PNG")
    return raw.getvalue()


def json_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in json_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in json_keys(v)}
    return set()


async def wait_for(client, path, ready, seconds=180, headers=None):
    deadline = time.monotonic() + seconds
    while True:
        response = await client.get(path, headers=headers)
        if response.status_code < 500 or response.status_code == 503:
            body = response.json()
            if ready(response, body):
                return body
        if time.monotonic() > deadline:
            raise AssertionError(f"timeout waiting for {path}: {response.text[:500]}")
        await asyncio.sleep(0.5)


async def accounts(client, suffix):
    result = {}
    for role in ("author", "searcher"):
        credentials = {"email": f"{role}-{suffix}@example.com", "password": "Smoke-password-42"}
        response = await client.post(
            "/api/v1/auth/register", json={**credentials, "display_name": role}
        )
        response.raise_for_status()
        response = await client.post("/api/v1/auth/login", json=credentials)
        response.raise_for_status()
        token = response.json()["access_token"]
        result[role] = ({"Authorization": f"Bearer {token}"}, credentials)
    return result


async def poll_search(client, url, headers):
    body = await wait_for(
        client, url, lambda r, b: b.get("status") in ("completed", "failed"), 90, headers
    )
    assert body["status"] == "completed", body
    return body


async def run(base_url="http://localhost:8000"):
    suffix = uuid4().hex
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        for path in ("/docs", "/openapi.json", "/health/live", "/health/ready"):
            response = await client.get(path)
            response.raise_for_status()
        status = (await client.get("/api/v1/cv/status")).json()
        real = status["configured_mode"] != "mock"
        if real:
            print(f"Waiting for CV worker readiness ({status['configured_mode']})...")
        status = (
            await wait_for(client, "/api/v1/cv/status", lambda r, b: b["ready"], 300)
            if real
            else status
        )
        users = await accounts(client, suffix)
        author_headers, author_credentials = users["author"]
        search_headers, _ = users["searcher"]
        image = FIXTURE.read_bytes() if real else solid_png()
        response = await client.post(
            "/api/v1/listings",
            headers=author_headers,
            json={
                "kind": "found",
                "species": "dog" if real else "cat",
                "breed": "бигль + терьер",
                "color": "трёхцветный",
                "description": f"Smoke {suffix}",
                "location": {"latitude": 43.2389, "longitude": 76.8897},
                "event_at": "2026-11-06T10:00:00Z",
            },
        )
        response.raise_for_status()
        listing_id = response.json()["id"]
        response = await client.post(
            f"/api/v1/listings/{listing_id}/photos",
            headers=author_headers,
            data={"bbox": "0.05,0.2,0.95,1"} if real else None,
            files={"file": ("fixture.jpg", image, "image/jpeg")},
        )
        response.raise_for_status()
        media_id = response.json()["id"]
        response = await client.post(
            f"/api/v1/listings/{listing_id}/publish", headers=author_headers
        )
        response.raise_for_status()
        if real:
            # Wait until every ready listing photo has a processed row in the active index (the
            # worker's scheduler enqueues new photos every CV_INDEX_SWEEP_SECONDS).
            def indexed(response, body):
                index = body["index"]
                done = index and index["indexed"] + index["rejected"] + index["failed"]
                return (
                    bool(index)
                    and index["pending"] + index["running"] == 0
                    and (done >= index["listing_photos_ready"])
                )

            await wait_for(client, "/api/v1/cv/status", indexed, 180)
            result = await real_search(client, search_headers, image, suffix, status)
            ids = [row["listing"]["id"] for row in result["results"]]
            match = next(row for row in result["results"] if row["listing"]["id"] == listing_id)
            assert match["matched_media_id"] == media_id, match
            print(f"Same-image plumbing search: listing rank {ids.index(listing_id) + 1}")
            rejected = await real_search(
                client, search_headers, solid_png(size=(320, 320)), suffix + "-nodog", status
            )
            assert rejected["outcome"] == "rejected" and rejected["reason_code"] == "no_dog"
            assert rejected["results"] == []
            print("No-dog query rejected with reason_code=no_dog")
        else:
            await legacy_mock_search(client, search_headers, image, suffix)
        response = await client.get(f"/api/v1/media/{media_id}")
        response.raise_for_status()
        mode = status["configured_mode"]
        print(f"Smoke passed: listing={listing_id} media={media_id} mode={mode}")
        return {
            "listing_id": listing_id,
            "media_id": media_id,
            "media_bytes": response.content.hex(),
            "credentials": author_credentials,
        }


async def real_search(client, headers, image, key, status):
    response = await client.post(
        "/api/v1/photo-searches/images",
        headers=headers,
        files={"file": ("query.jpg", image, "image/jpeg")},
    )
    response.raise_for_status()
    response = await client.post(
        "/api/v2/photo-searches",
        headers={**headers, "Idempotency-Key": key},
        json={
            "query_media_id": response.json()["id"],
            "breed_text": "Бигль + терьер",
            "coat_color_text": "трёхцветный: чёрный, рыжий, белый",
            "target_kind": "found",
            "limit": 20,
        },
    )
    assert response.status_code == 202, response.text
    result = await poll_search(client, response.headers["location"], headers)
    assert result["provider"] == "pawly-cv" and result["is_mock"] is False, result
    assert result["cv_mode"] == status["model_stage"], result
    assert result["embedding_space_id"] == status["embedding_space_id"], result
    assert result["pipeline_id"] == status["pipeline_id"], result
    assert result["query"]["breed_text"] == "Бигль + терьер"
    keys = json_keys(result)
    assert not {k for k in keys if "score" in k or "similar" in k or "confidence" in k}, keys
    return result


async def legacy_mock_search(client, headers, image, suffix):
    response = await client.post(
        "/api/v1/photo-searches/images",
        headers=headers,
        files={"file": ("query.png", image, "image/png")},
    )
    response.raise_for_status()
    response = await client.post(
        "/api/v1/photo-searches",
        headers={**headers, "Idempotency-Key": suffix},
        json={
            "query_media_id": response.json()["id"],
            "target_kind": "found",
            "species": "cat",
            "limit": 5,
        },
    )
    assert response.status_code == 202, response.text
    result = await poll_search(client, f"/api/v1/photo-searches/{response.json()['id']}", headers)
    assert result["is_mock"] is True and result["provider"] == "mock-v1"
    assert 1 <= len(result["results"]) <= 5
    assert all(row["similarity_score"] is None for row in result["results"])
    print("Mock mode: plumbing only; no visual similarity is measured.")


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
