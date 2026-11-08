"""HTTP-only smoke flow against an actual API and independently running worker."""

import asyncio
import io
import sys
import time
from uuid import uuid4

import httpx
from PIL import Image


async def run(base_url="http://localhost:8000"):
    raw = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 80, 40)).save(raw, "PNG")
    image = raw.getvalue()
    suffix = uuid4().hex
    async with httpx.AsyncClient(base_url=base_url, timeout=20) as client:
        for path in ("/docs", "/openapi.json", "/health/live", "/health/ready"):
            response = await client.get(path)
            response.raise_for_status()
        for role in ("author", "searcher"):
            credentials = {"email": f"{role}-{suffix}@example.com", "password": "Smoke-password-42"}
            response = await client.post(
                "/api/v1/auth/register",
                json={
                    **credentials,
                    "display_name": role,
                },
            )
            response.raise_for_status()
            response = await client.post("/api/v1/auth/login", json=credentials)
            response.raise_for_status()
            token = response.json()["access_token"]
            if role == "author":
                author_credentials = credentials
                author_headers = {"Authorization": f"Bearer {token}"}
            else:
                search_headers = {"Authorization": f"Bearer {token}"}
        response = await client.post(
            "/api/v1/listings",
            headers=author_headers,
            json={
                "kind": "found",
                "species": "cat",
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
            files={"file": ("fixture.png", image, "image/png")},
        )
        response.raise_for_status()
        media_id = response.json()["id"]
        response = await client.post(
            f"/api/v1/listings/{listing_id}/publish", headers=author_headers
        )
        response.raise_for_status()
        response = await client.get(
            "/api/v1/listings",
            params={
                "kind": "found",
                "latitude": 43.2389,
                "longitude": 76.8897,
                "radius_m": 1000,
            },
        )
        response.raise_for_status()
        assert any(row["id"] == listing_id for row in response.json()["items"])
        response = await client.post(
            "/api/v1/photo-searches/images",
            headers=search_headers,
            files={"file": ("query.png", image, "image/png")},
        )
        response.raise_for_status()
        query_id = response.json()["id"]
        response = await client.post(
            "/api/v1/photo-searches",
            headers={
                **search_headers,
                "Idempotency-Key": suffix,
            },
            json={"query_media_id": query_id, "target_kind": "found", "species": "cat", "limit": 5},
        )
        assert response.status_code == 202, response.text
        search_id = response.json()["id"]
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            response = await client.get(
                f"/api/v1/photo-searches/{search_id}", headers=search_headers
            )
            response.raise_for_status()
            result = response.json()
            if result["status"] == "completed":
                break
            assert result["status"] != "failed", result
            await asyncio.sleep(0.25)
        else:
            raise AssertionError("Worker did not complete the search")
        assert result["is_mock"] is True and result["provider"] == "mock-v1"
        assert result["model_version"] is None
        assert 1 <= len(result["results"]) <= 5
        assert all(row["similarity_score"] is None for row in result["results"])
        response = await client.get(f"/api/v1/media/{media_id}")
        response.raise_for_status()
        print(f"Smoke passed: listing={listing_id} media={media_id} search={search_id}")
        print("Synthetic image validates HTTP/media plumbing only; no CV quality is measured.")
        return {
            "listing_id": listing_id,
            "media_id": media_id,
            "media_bytes": response.content.hex(),
            "credentials": author_credentials,
        }


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
