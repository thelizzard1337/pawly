"""PhotoMatcher adapter boundary.

The worker owns the queue/lease/fencing; a matcher turns one claimed search job into an outcome.
``MockPhotoMatcher`` is plumbing only and is always labelled ``mock``; ``VectorPhotoMatcher``
(app.integrations.cv.vector_matcher) is the real pretrained-feature implementation.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from app.cv.pipeline import MOCK_PROVIDER


@dataclass(frozen=True)
class Candidate:
    listing_id: UUID
    media_id: UUID


@dataclass(frozen=True)
class RankedCandidate:
    listing_id: UUID
    media_id: UUID
    score: float | None = None  # internal ranking score; never returned to users


@dataclass
class MatchOutcome:
    provider: str
    cv_mode: str
    outcome: str  # "candidates" | "rejected"
    matches: list[RankedCandidate] = field(default_factory=list)
    reason_code: str | None = None
    model_version: str | None = None
    pipeline_id: str | None = None
    embedding_space_id: str | None = None
    fusion_version: str | None = None
    query_feature_id: UUID | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_mock(self) -> bool:
        return self.provider == MOCK_PROVIDER

    @classmethod
    def mock(cls, candidates: list[Candidate]) -> "MatchOutcome":
        return cls(
            provider=MOCK_PROVIDER,
            cv_mode="mock",
            outcome="candidates",
            matches=[RankedCandidate(c.listing_id, c.media_id) for c in candidates],
            meta={"note": "mock order is a SHA-256 digest, not visual similarity"},
        )


class PhotoMatcher(Protocol):
    provider: str

    async def search(self, sessions, job) -> MatchOutcome: ...


class MockPhotoMatcher:
    """Stable plumbing test. The digest does not measure visual similarity."""

    provider = MOCK_PROVIDER

    async def search(self, sessions, job) -> MatchOutcome:
        from app.modules.photo_search.jobs import legacy_candidates

        query_hash, candidates = await legacy_candidates(sessions, job)
        return MatchOutcome.mock(await self.match(query_hash, candidates, job.limit))

    async def match(
        self, query_hash: str, candidates: list[Candidate], limit: int
    ) -> list[Candidate]:
        unique = {}
        for candidate in candidates:
            previous = unique.get(candidate.listing_id)
            if previous is None or candidate.media_id.int < previous.media_id.int:
                unique[candidate.listing_id] = candidate
        return sorted(
            unique.values(),
            key=lambda candidate: (
                hashlib.sha256(
                    f"{query_hash}:{candidate.listing_id}:{candidate.media_id}".encode(),
                ).digest(),
                str(candidate.listing_id),
            ),
        )[:limit]
