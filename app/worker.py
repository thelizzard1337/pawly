import asyncio
import logging
import signal

from app.core.config import Settings
from app.db.session import make_database
from app.integrations.cv.matcher import MockPhotoMatcher
from app.modules.photo_search import jobs

logger = logging.getLogger("pawly.worker")


async def process_job(sessions, settings, matcher, job):
    try:
        async with asyncio.timeout(settings.worker_timeout_seconds):
            matches = await jobs.execute(sessions, matcher, job)
        return await jobs.finalize(sessions, job, matches)
    except ValueError:
        return await jobs.failed(sessions, settings, job, "query_media_unavailable", False)
    except TimeoutError:
        return await jobs.failed(sessions, settings, job, "matcher_timeout")
    except Exception:
        logger.warning("Photo search attempt failed; retry scheduled")
        return await jobs.failed(sessions, settings, job, "matcher_temporarily_unavailable")


async def run():
    settings = Settings()
    engine, sessions = make_database(settings)
    matcher, stop = MockPhotoMatcher(), asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        while not stop.is_set():
            job = None
            try:
                await jobs.recover(sessions, settings)
                if stop.is_set():
                    break
                job = await jobs.claim(sessions, settings)
                if job:
                    await process_job(sessions, settings, matcher, job)
            except Exception:
                # Connection pools pre-ping; retries recover after DB restarts.
                logger.warning("Worker database operation failed; will retry")
            if job is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_seconds)
                except TimeoutError:
                    continue
    finally:
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run())
