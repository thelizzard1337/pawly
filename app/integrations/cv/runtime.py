"""Runs CPU-heavy extraction off the worker event loop, in one dedicated model thread."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from app.cv.pipeline import PipelineSpec

logger = logging.getLogger("pawly.cv")


class CVUnavailable(RuntimeError):
    """The configured real pipeline is not loaded; jobs are retried, never mocked."""


class CVRuntime:
    def __init__(self, extractor):
        self.extractor = extractor
        self.pipeline: PipelineSpec = extractor.pipeline
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pawly-cv")

    async def extract(self, raw: bytes, bbox):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.extractor.extract, raw, bbox)

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


async def load_runtime(settings) -> CVRuntime:
    """Load models in a thread. Errors propagate: trained_heads never degrades silently."""
    from app.cv.extractor import FeatureExtractor

    extractor = await asyncio.to_thread(FeatureExtractor.from_settings, settings)
    return CVRuntime(extractor)
