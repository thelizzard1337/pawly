import hashlib
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class Candidate:
    listing_id: UUID
    media_id: UUID


class PhotoMatcher(Protocol):
    async def match(
        self, query_hash: str, candidates: list[Candidate], limit: int
    ) -> list[Candidate]: ...


class MockPhotoMatcher:
    """Stable plumbing test. The digest does not measure visual similarity."""

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
        )[: min(limit, 5)]
