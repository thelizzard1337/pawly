import asyncio
from uuid import UUID

import pytest
from sqlalchemy import func, select

from app.db.models import AdoptionRequest, Listing
from tests.helpers import grant_admin, listing

pytestmark = pytest.mark.integration


async def test_cli_role_commands_are_idempotent(client, app, actor):
    from types import SimpleNamespace

    from app.cli import execute
    from app.db.models import ShelterMember

    owner, manager = await actor("owner"), await actor("new-manager")
    sid = await shelter(client, owner)
    for _ in range(2):
        await execute(SimpleNamespace(command="grant-admin", email=owner["credentials"]["email"]))
        await execute(
            SimpleNamespace(
                command="add-manager", shelter_id=UUID(sid), email=manager["credentials"]["email"]
            )
        )
    response = await client.get("/api/v1/users/me", headers=owner["headers"])
    assert response.json()["is_admin"] is True
    async with app.state.sessions() as db:
        memberships = list(
            await db.scalars(
                select(ShelterMember).where(
                    ShelterMember.shelter_id == UUID(sid),
                    ShelterMember.user_id == UUID(manager["id"]),
                )
            )
        )
        assert len(memberships) == 1


async def shelter(client, owner):
    response = await client.post(
        "/api/v1/shelters",
        headers=owner["headers"],
        json={
            "name": "Kind shelter",
            "description": "For cats",
            "city": "Almaty",
        },
    )
    assert response.status_code == 201
    assert response.json()["verification_status"] == "pending"
    return response.json()["id"]


async def test_verification_manager_and_approval_race(client, app, actor, image):
    manager, stranger, admin = await actor("manager"), await actor("stranger"), await actor("admin")
    await grant_admin(app, admin)
    sid = await shelter(client, manager)
    response = await client.patch(
        f"/api/v1/shelters/{sid}",
        headers=manager["headers"],
        json={"verification_status": "verified"},
    )
    assert response.status_code == 422
    response = await client.patch(
        f"/api/v1/shelters/{sid}", headers=stranger["headers"], json={"name": "stolen"}
    )
    assert response.status_code == 403
    uid, _ = await listing(client, manager, image, publish=False, kind="adoption", shelter_id=sid)
    response = await client.post(f"/api/v1/listings/{uid}/publish", headers=manager["headers"])
    assert response.status_code == 409
    assert (
        await client.post(f"/api/v1/admin/shelters/{sid}/verify", headers=manager["headers"])
    ).status_code == 403
    assert (
        await client.post(f"/api/v1/admin/shelters/{sid}/verify", headers=admin["headers"])
    ).status_code == 200
    assert (
        await client.post(f"/api/v1/listings/{uid}/publish", headers=manager["headers"])
    ).status_code == 200
    request_ids = []
    for applicant in (stranger, await actor("applicant")):
        response = await client.post(
            f"/api/v1/listings/{uid}/adoption-requests",
            headers=applicant["headers"],
            json={"message": "A loving home"},
        )
        assert response.status_code == 201, response.text
        request_ids.append(response.json()["id"])
        duplicate = await client.post(
            f"/api/v1/listings/{uid}/adoption-requests",
            headers=applicant["headers"],
            json={"message": "Duplicate"},
        )
        assert duplicate.status_code == 409
    assert (
        await client.get(f"/api/v1/shelters/{sid}/adoption-requests", headers=stranger["headers"])
    ).status_code == 403
    assert (
        await client.post(
            f"/api/v1/adoption-requests/{request_ids[0]}/approve", headers=stranger["headers"]
        )
    ).status_code == 403
    responses = await asyncio.gather(
        *[
            client.post(f"/api/v1/adoption-requests/{rid}/approve", headers=manager["headers"])
            for rid in request_ids
        ]
    )
    assert sorted(r.status_code for r in responses) == [200, 409]
    async with app.state.sessions() as db:
        approved = await db.scalar(
            select(func.count())
            .select_from(AdoptionRequest)
            .where(
                AdoptionRequest.listing_id == UUID(uid),
                AdoptionRequest.status == "approved",
            )
        )
        rejected = await db.scalar(
            select(func.count())
            .select_from(AdoptionRequest)
            .where(
                AdoptionRequest.listing_id == UUID(uid),
                AdoptionRequest.status == "rejected",
            )
        )
        assert approved == rejected == 1
        assert (await db.get(Listing, UUID(uid))).status == "resolved"


async def test_withdraw_reject_and_contacts(client, app, actor, image):
    manager, applicant, admin = (
        await actor("manager"),
        await actor("applicant"),
        await actor("admin"),
    )
    await grant_admin(app, admin)
    sid = await shelter(client, manager)
    await client.post(f"/api/v1/admin/shelters/{sid}/verify", headers=admin["headers"])
    uid, _ = await listing(client, manager, image, kind="adoption", shelter_id=sid)
    response = await client.post(
        f"/api/v1/listings/{uid}/adoption-requests",
        headers=applicant["headers"],
        json={"message": "Please consider me"},
    )
    rid = response.json()["id"]
    response = await client.post(
        f"/api/v1/adoption-requests/{rid}/withdraw", headers=applicant["headers"]
    )
    assert response.status_code == 200 and response.json()["status"] == "withdrawn"
    response = await client.post(
        f"/api/v1/listings/{uid}/adoption-requests",
        headers=applicant["headers"],
        json={"message": "Apply again"},
    )
    rid = response.json()["id"]
    response = await client.post(
        f"/api/v1/adoption-requests/{rid}/reject", headers=manager["headers"]
    )
    assert response.status_code == 200 and response.json()["status"] == "rejected"
    found, _ = await listing(client, manager, image)
    response = await client.post(
        f"/api/v1/listings/{found}/contact-requests",
        headers=applicant["headers"],
        json={"message": "I recognize this cat"},
    )
    assert response.status_code == 201
    response = await client.get("/api/v1/users/me/contact-requests", headers=manager["headers"])
    assert len(response.json()["items"]) == 1
    assert "phone" not in response.text and "email" not in response.text
    assert (
        await client.get("/api/v1/users/me/contact-requests", headers=applicant["headers"])
    ).json()["items"] == []
