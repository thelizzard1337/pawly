"""Durable worker: photo-search jobs first, then listing-photo indexing (CV modes only).

The model lives only in this process (loaded once, run in a dedicated thread). API processes
never import torch.
"""

import asyncio
import logging
import os
import signal
import socket

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert

from app.core.config import Settings
from app.core.security import utcnow
from app.db.models import CVWorkerStatus
from app.db.session import make_database
from app.integrations.cv.matcher import MockPhotoMatcher
from app.integrations.cv.runtime import CVUnavailable
from app.integrations.storage.local import LocalFileStorage
from app.modules.cv_index import indexing
from app.modules.cv_index import repository as index_repo
from app.modules.photo_search import jobs

logger = logging.getLogger("pawly.worker")


async def process_job(sessions, settings, matcher, job):
    try:
        async with asyncio.timeout(settings.worker_timeout_seconds):
            outcome = await jobs.execute(sessions, matcher, job)
        return await jobs.finalize(sessions, job, outcome)
    except ValueError:
        return await jobs.failed(sessions, settings, job, "query_media_unavailable", False)
    except TimeoutError:
        return await jobs.failed(sessions, settings, job, "matcher_timeout")
    except CVUnavailable:
        return await jobs.failed(sessions, settings, job, "cv_unavailable")
    except Exception:
        logger.warning("Photo search attempt failed; retry scheduled")
        return await jobs.failed(sessions, settings, job, "matcher_temporarily_unavailable")


class Heartbeat:
    def __init__(self, sessions, settings):
        self.sessions, self.settings = sessions, settings
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"[:100]
        self.started_at = utcnow()

    async def beat(self, state, pipeline=None, error=None, details=None):
        values = {
            "cv_mode": self.settings.cv_mode,
            "provider": pipeline.provider
            if pipeline
            else ("mock-v1" if self.settings.cv_mode == "mock" else "pawly-cv"),
            "state": state,
            "pipeline_id": pipeline.pipeline_id if pipeline else None,
            "embedding_space_id": pipeline.embedding_space_id if pipeline else None,
            "bundle_id": pipeline.bundle_id if pipeline else None,
            "error_code": error,
            "details": details or (pipeline.describe() if pipeline else None),
            "heartbeat_at": utcnow(),
        }
        async with self.sessions() as db:
            await db.execute(
                insert(CVWorkerStatus)
                .values(worker_id=self.worker_id, started_at=self.started_at, **values)
                .on_conflict_do_update(index_elements=["worker_id"], set_=values)
            )
            await db.commit()

    async def remove(self):
        """Graceful stop: a replaced worker must not keep reporting readiness."""
        async with self.sessions() as db:
            await db.execute(
                delete(CVWorkerStatus).where(CVWorkerStatus.worker_id == self.worker_id)
            )
            await db.commit()


async def build_matcher(settings, storage):
    """Returns (matcher, runtime, error). Never falls back from a real mode to mock."""
    if settings.cv_mode == "mock":
        return MockPhotoMatcher(), None, None
    from app.integrations.cv.runtime import load_runtime
    from app.integrations.cv.vector_matcher import VectorPhotoMatcher

    try:
        runtime = await load_runtime(settings)
    except Exception as exc:  # weights missing, bundle invalid, incompatible taxonomy...
        code = str(exc).splitlines()[0][:200] or type(exc).__name__
        logger.error("CV pipeline failed to load (%s): %s", settings.cv_mode, code)
        return VectorPhotoMatcher(None, storage, settings, load_error=code), None, code
    logger.info("CV pipeline ready: %s", runtime.pipeline.pipeline_id)
    return VectorPhotoMatcher(runtime, storage, settings), runtime, None


async def run():
    settings = Settings()
    engine, sessions = make_database(settings)
    storage = LocalFileStorage(settings.media_root)
    heartbeat, stop = Heartbeat(sessions, settings), asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    runtime = None
    try:
        try:
            await heartbeat.beat("loading")
        except Exception:
            logger.warning("Heartbeat failed; database not reachable yet")
        matcher, runtime, load_error = await build_matcher(settings, storage)
        pipeline = runtime.pipeline if runtime else None
        last_beat = last_sweep = last_reload = 0.0
        while not stop.is_set():
            job = index_job = None
            now = loop.time()
            try:
                if settings.cv_mode != "mock" and runtime is None and now - last_reload > 60:
                    last_reload = now
                    matcher, runtime, load_error = await build_matcher(settings, storage)
                    pipeline = runtime.pipeline if runtime else None
                if now - last_beat >= settings.cv_heartbeat_seconds:
                    state = "error" if load_error else "ready"
                    await heartbeat.beat(state, pipeline, load_error)
                    last_beat = now
                await jobs.recover(sessions, settings)
                if stop.is_set():
                    break
                job = await jobs.claim(sessions, settings)
                if job:
                    await process_job(sessions, settings, matcher, job)
                elif pipeline is not None:
                    await index_repo.recover(sessions, settings, pipeline.pipeline_id)
                    if now - last_sweep >= settings.cv_index_sweep_seconds:
                        await index_repo.schedule_missing(
                            sessions, pipeline, settings.cv_index_batch
                        )
                        last_sweep = now
                    index_job = await index_repo.claim(sessions, settings, pipeline.pipeline_id)
                    if index_job:
                        await indexing.process(sessions, settings, storage, runtime, index_job)
            except Exception:
                # Connection pools pre-ping; retries recover after DB restarts.
                logger.warning("Worker database operation failed; will retry")
            if job is None and index_job is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_seconds)
                except TimeoutError:
                    continue
    finally:
        try:
            await heartbeat.remove()
        except Exception:
            logger.warning("Could not remove worker heartbeat; it expires on its own")
        if runtime is not None:
            runtime.close()
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run())
