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


async def cv_command(args, settings, sessions):
    """CV index maintenance. Run inside the worker container (it has the bundle mount)."""
    from app.cv.pipeline import PipelineConfigError, resolve
    from app.modules.cv_index import repository as index_repo
    from app.modules.cv_index.router import status_payload

    if args.command == "model-status":
        report = {"configured_mode": settings.cv_mode}
        if settings.cv_mode != "mock":
            try:
                pipeline = resolve(settings)
                report["configured_pipeline"] = pipeline.describe()
            except PipelineConfigError as exc:
                report["configuration_error"] = str(exc)
            if settings.cv_mode == "trained_heads" and "configuration_error" not in report:
                try:  # Full check (checksums, shapes, taxonomy) needs the CV image.
                    from app.cv.bundle import BundleError, load_bundle

                    bundle = load_bundle(settings.cv_bundle_dir, settings.cv_backbone)
                    report["bundle"] = {"valid": True, "bundle_id": bundle.bundle_id}
                except ImportError:
                    report["bundle"] = {"valid": None, "note": "torch not installed here"}
                except BundleError as exc:
                    report["bundle"] = {"valid": False, "error": str(exc)}
        async with sessions() as db:
            report["runtime"] = await status_payload(db, settings)
        print(json.dumps(report, indent=2, default=str))
        return
    if settings.cv_mode == "mock":
        raise SystemExit("CV_MODE=mock has no feature index")
    pipeline = resolve(settings)
    reset = 0
    if args.command == "reindex":
        reset = await index_repo.reindex(sessions, pipeline.pipeline_id, args.failed_only)
    scheduled = 0
    while created := await index_repo.schedule_missing(sessions, pipeline, args.batch):
        scheduled += created
    async with sessions() as db:
        counts = await index_repo.counts(db, pipeline.pipeline_id)
    print(
        json.dumps(
            {
                "pipeline_id": pipeline.pipeline_id,
                "reset_to_pending": reset,
                "newly_scheduled": scheduled,
                "counts": counts,
                "note": "The CV worker processes pending rows in the background.",
            },
            indent=2,
        )
    )


async def execute(args):
    settings = Settings()
    engine, sessions = make_database(settings)
    try:
        if args.command == "cleanup-media":
            storage = LocalFileStorage(settings.media_root)
            await run_in_threadpool(storage.check)
            print(json.dumps(await cleanup(sessions, storage)))
            return
        if args.command in ("model-status", "index-existing-media", "reindex"):
            await cv_command(args, settings, sessions)
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
    commands.add_parser("model-status", help="CV mode, pipeline, bundle validity, index counts")
    existing = commands.add_parser(
        "index-existing-media", help="Enqueue ready listing photos missing from the active index"
    )
    existing.add_argument("--batch", type=int, default=500)
    reindex = commands.add_parser(
        "reindex", help="Recompute features of the active pipeline (keeps provenance history)"
    )
    reindex.add_argument("--failed-only", action="store_true")
    reindex.add_argument("--batch", type=int, default=500)
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
