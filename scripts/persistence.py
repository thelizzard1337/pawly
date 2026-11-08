"""Used by verify-compose.sh; state travels over stdin/stdout without host Python."""

import asyncio
import contextlib
import json
import sys

import httpx

from scripts.smoke import run


async def main():
    action, base_url = sys.argv[1:3]
    if action == "capture":
        with contextlib.redirect_stdout(sys.stderr):
            state = await run(base_url)
        print(json.dumps(state))
    else:
        state = json.load(sys.stdin)
        async with httpx.AsyncClient(base_url=base_url) as client:
            response = await client.post("/api/v1/auth/login", json=state["credentials"])
            response.raise_for_status()
            response = await client.get(f"/api/v1/listings/{state['listing_id']}")
            response.raise_for_status()
            response = await client.get(f"/api/v1/media/{state['media_id']}")
            response.raise_for_status()
            assert response.content.hex() == state["media_bytes"]
        print("Persistence passed: user login, listing and identical media bytes survived recreate")


if __name__ == "__main__":
    asyncio.run(main())
