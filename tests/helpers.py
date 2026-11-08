from uuid import UUID

from app.db.models import User


async def grant_admin(app, actor):
    async with app.state.sessions() as db:
        user = await db.get(User, UUID(actor["id"]))
        user.is_admin = True
        await db.commit()


async def listing(client, actor, image, *, publish=True, photo=True, **overrides):
    data = {
        "kind": "found",
        "species": "cat",
        "description": "Cat near the park",
        "location": {"latitude": 43.2389, "longitude": 76.8897},
        "event_at": "2026-11-06T10:00:00Z",
        **overrides,
    }
    response = await client.post("/api/v1/listings", json=data, headers=actor["headers"])
    assert response.status_code == 201, response.text
    uid = response.json()["id"]
    media_id = None
    if photo:
        response = await client.post(
            f"/api/v1/listings/{uid}/photos",
            headers=actor["headers"],
            files={"file": ("cat.png", image, "image/png")},
        )
        assert response.status_code == 201, response.text
        media_id = response.json()["id"]
    if publish:
        response = await client.post(f"/api/v1/listings/{uid}/publish", headers=actor["headers"])
        assert response.status_code == 200, response.text
    return uid, media_id


async def search(client, actor, image, **overrides):
    response = await client.post(
        "/api/v1/photo-searches/images",
        headers=actor["headers"],
        files={"file": ("query.png", image, "image/png")},
    )
    assert response.status_code == 201, response.text
    query_id = response.json()["id"]
    response = await client.post(
        "/api/v1/photo-searches",
        headers=actor["headers"],
        json={
            "query_media_id": query_id,
            "target_kind": "found",
            "species": "cat",
            **overrides,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()["id"], query_id
