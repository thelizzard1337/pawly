"""Process one claimed listing-photo indexing job (no DB transaction is held during inference)."""

import asyncio
import hashlib
import logging

from app.cv import spec
from app.db.models import Listing, MediaAsset
from app.modules.cv_index import repository as repo

logger = logging.getLogger("pawly.cv.index")


async def read_canonical(storage, key: str, expected_sha256: str) -> bytes:
    raw = await asyncio.to_thread(storage.path(key).read_bytes)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("input_hash_mismatch")
    return raw


async def process(sessions, settings, storage, runtime, job: repo.IndexJob) -> bool:
    async with sessions() as db:
        media = await db.get(MediaAsset, job.media_id)
        if media is None or media.status != "ready" or media.listing_id is None:
            return await repo.fail(sessions, settings, job, "media_unavailable", temporary=False)
        listing = await db.get(Listing, media.listing_id)
        texts = (listing.breed, listing.color) if listing else (None, None)
        key = media.storage_key
    try:
        raw = await read_canonical(storage, key, job.input_sha256)
    except FileNotFoundError:
        return await repo.fail(sessions, settings, job, "media_file_missing", temporary=False)
    except ValueError:
        return await repo.fail(sessions, settings, job, "input_hash_mismatch", temporary=False)
    except OSError:
        return await repo.fail(sessions, settings, job, "storage_unavailable")
    try:
        async with asyncio.timeout(settings.worker_timeout_seconds):
            extraction = await runtime.extract(raw, spec.bbox_or_none(job.crop_bbox))
    except TimeoutError:
        return await repo.fail(sessions, settings, job, "extraction_timeout")
    except Exception:
        logger.warning("Feature extraction failed; retry scheduled")
        return await repo.fail(sessions, settings, job, "extraction_failed")
    # Domain rejection (no dog, several dogs, bad crop) is a final state, not retried.
    return await repo.complete(sessions, job, extraction, texts)
