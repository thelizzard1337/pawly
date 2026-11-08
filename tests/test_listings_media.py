import asyncio
import io

import pytest
from PIL import Image

from tests.helpers import grant_admin, listing

pytestmark = pytest.mark.integration


async def test_lifecycle_privacy_and_publish_invariants(client, actor, image):
    owner, stranger = await actor("owner"), await actor("stranger")
    uid, media_id = await listing(client, owner, image, publish=False)
    assert (await client.get(f"/api/v1/listings/{uid}")).status_code == 404
    assert (await client.get(f"/api/v1/media/{media_id}")).status_code == 404
    assert (
        await client.patch(
            f"/api/v1/listings/{uid}", headers=stranger["headers"], json={"title": "stolen"}
        )
    ).status_code == 404
    assert (
        await client.post(f"/api/v1/listings/{uid}/publish", headers=owner["headers"])
    ).status_code == 200
    assert (await client.get(f"/api/v1/listings/{uid}")).status_code == 200
    assert (await client.get(f"/api/v1/media/{media_id}")).status_code == 200
    for payload in (
        {"species": None},
        {"location": None},
        {"description": " "},
        {"event_at": None},
    ):
        response = await client.patch(
            f"/api/v1/listings/{uid}", headers=owner["headers"], json=payload
        )
        assert response.status_code == 409, response.text
    response = await client.delete(
        f"/api/v1/listings/{uid}/photos/{media_id}", headers=owner["headers"]
    )
    assert response.status_code == 409
    assert (await client.post(f"/api/v1/listings/{uid}/resolve", headers=owner["headers"])).json()[
        "status"
    ] == "resolved"
    assert (
        await client.post(f"/api/v1/listings/{uid}/publish", headers=owner["headers"])
    ).status_code == 409
    assert (await client.delete(f"/api/v1/listings/{uid}", headers=owner["headers"])).json()[
        "status"
    ] == "archived"
    assert (await client.get(f"/api/v1/media/{media_id}")).status_code == 404
    assert (
        await client.get(f"/api/v1/media/{media_id}", headers=owner["headers"])
    ).status_code == 200


async def test_publish_missing_photo_and_incomplete_draft(client, actor, image):
    owner = await actor()
    uid, _ = await listing(client, owner, image, photo=False, publish=False)
    response = await client.post(f"/api/v1/listings/{uid}/publish", headers=owner["headers"])
    assert response.status_code == 409
    response = await client.post(
        "/api/v1/listings", headers=owner["headers"], json={"kind": "lost"}
    )
    assert response.status_code == 201
    uid = response.json()["id"]
    assert (
        await client.post(f"/api/v1/listings/{uid}/publish", headers=owner["headers"])
    ).status_code == 409


async def test_filters_map_cursor_and_hidden(client, app, actor, image):
    owner, admin = await actor(), await actor("admin")
    await grant_admin(app, admin)
    ids = [(await listing(client, owner, image))[0] for _ in range(3)]
    near = {"latitude": 43.2389, "longitude": 76.8897, "radius_m": 500}
    assert len((await client.get("/api/v1/listings", params=near)).json()["items"]) == 3
    far = {"latitude": 76.8897, "longitude": 43.2389, "radius_m": 500}
    assert (await client.get("/api/v1/listings", params=far)).json()["items"] == []
    for params in ({"latitude": 1}, {"radius_m": 0}, {"limit": 101}, {"cursor": "garbage"}):
        assert (await client.get("/api/v1/listings", params=params)).status_code == 422
    assert (await client.get("/api/v1/listings", params={"species": "dog"})).json()["items"] == []
    seen, cursor = [], None
    for _ in range(4):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        response = (await client.get("/api/v1/listings", params=params)).json()
        seen += [row["id"] for row in response["items"]]
        cursor = response["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == len(set(seen)) == 3 and set(seen) == set(ids)
    response = await client.get("/api/v1/listings/map", params={"bbox": "76,43,77,44", "limit": 1})
    assert response.status_code == 200 and response.json()["truncated"] is True
    assert len(response.json()["items"]) == 1
    for bbox in ("nan,0,10,20", "0,0,181,90", "0,1,2", "5,0,0,1"):
        assert (await client.get("/api/v1/listings/map", params={"bbox": bbox})).status_code == 422
    assert (
        await client.post(f"/api/v1/admin/listings/{ids[0]}/hide", headers=owner["headers"])
    ).status_code == 403
    response = await client.post(f"/api/v1/admin/listings/{ids[0]}/hide", headers=admin["headers"])
    assert response.status_code == 200
    assert ids[0] not in [
        row["id"] for row in (await client.get("/api/v1/listings")).json()["items"]
    ]
    points = (await client.get("/api/v1/listings/map", params={"bbox": "76,43,77,44"})).json()
    assert ids[0] not in [row["id"] for row in points["items"]]


async def test_upload_validation_and_metadata(client, actor, image):
    owner = await actor()
    uid, _ = await listing(client, owner, image, publish=False, photo=False)
    endpoint = f"/api/v1/listings/{uid}/photos"
    response = await client.post(
        endpoint,
        headers=owner["headers"],
        files={"file": ("fake.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 422
    response = await client.post(
        endpoint,
        headers=owner["headers"],
        files={"file": ("big.jpg", b"x" * (10 * 1024 * 1024 + 1))},
    )
    assert response.status_code == 413
    # The actual bytes determine type; neither the filename nor MIME is trusted.
    response = await client.post(
        endpoint, headers=owner["headers"], files={"file": ("../../fake.exe", image, "text/plain")}
    )
    assert response.status_code == 201
    raw = (
        await client.get(f"/api/v1/media/{response.json()['id']}", headers=owner["headers"])
    ).content
    with Image.open(io.BytesIO(raw)) as decoded:
        assert decoded.format == "JPEG" and not decoded.getexif()


async def test_concurrent_upload_limit(client, actor, image):
    owner = await actor()
    uid, _ = await listing(client, owner, image, photo=False, publish=False)
    responses = await asyncio.gather(
        *[
            client.post(
                f"/api/v1/listings/{uid}/photos",
                headers=owner["headers"],
                files={"file": ("cat.png", image, "image/png")},
            )
            for _ in range(8)
        ]
    )
    assert sorted(r.status_code for r in responses) == [201] * 5 + [409] * 3
    response = await client.get(f"/api/v1/listings/{uid}", headers=owner["headers"])
    photos = response.json()["photos"]
    assert len(photos) == 5 and len({p["position"] for p in photos}) == 5
