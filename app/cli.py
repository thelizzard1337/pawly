import argparse
import asyncio
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.db.models import Shelter, ShelterMember, User
from app.db.session import make_database
from app.integrations.storage.local import LocalFileStorage
from app.modules.media.service import cleanup


async def execute(args):
    settings = Settings()
    engine, sessions = make_database(settings)
    try:
        if args.command == "cleanup-media":
            storage = LocalFileStorage(settings.media_root)
            await run_in_threadpool(storage.check)
            print(json.dumps(await cleanup(sessions, storage)))
            return
        async with sessions() as db:
            user = await db.scalar(select(User).where(User.email == args.email.strip().lower()))
            if user is None or not user.is_active:
                raise SystemExit("Active registered user not found")
            if args.command == "grant-admin":
                user.is_admin = True
            else:
                shelter = await db.get(Shelter, args.shelter_id)
                if shelter is None:
                    raise SystemExit("Shelter not found")
                await db.execute(
                    insert(ShelterMember)
                    .values(
                        shelter_id=shelter.id,
                        user_id=user.id,
                        role="manager",
                    )
                    .on_conflict_do_nothing()
                )
            await db.commit()
            print("Role granted (idempotent)")
    finally:
        await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Explicit PawLy administration")
    commands = parser.add_subparsers(dest="command", required=True)
    admin = commands.add_parser("grant-admin")
    admin.add_argument("email")
    manager = commands.add_parser("add-manager")
    manager.add_argument("shelter_id", type=UUID)
    manager.add_argument("email")
    commands.add_parser("cleanup-media")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
