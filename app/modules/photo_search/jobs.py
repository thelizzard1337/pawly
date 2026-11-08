from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import distinct_on

from app.core.security import utcnow
from app.db.models import Listing, MediaAsset, PhotoSearch, PhotoSearchResult
from app.integrations.cv.matcher import Candidate, PhotoMatcher
from app.modules.listings.repository import apply_filters, public_conditions
from app.modules.listings.schemas import Filters


@dataclass(frozen=True)
class Job:
    id: UUID
    lease_token: UUID
    query_media_id: UUID
    scope: dict
    limit: int


async def recover(sessions, settings):
    now = utcnow()
    async with sessions() as db:
        rows = await db.scalars(
            select(PhotoSearch)
            .where(
                ((PhotoSearch.status == "running") & (PhotoSearch.lease_until <= func.now()))
                | (
                    PhotoSearch.status.in_(["queued", "running"])
                    & (PhotoSearch.expires_at <= func.now())
                ),
            )
            .with_for_update(skip_locked=True)
        )
        for row in rows:
            if row.expires_at <= now or row.attempts >= settings.worker_max_attempts:
                row.status = "failed"
                row.error_code = "expired" if row.expires_at <= now else "attempts_exhausted"
                row.finished_at = now
            else:
                row.status = "queued"
                row.available_at = now + timedelta(seconds=2**row.attempts)
            row.lease_token = row.lease_until = None
        await db.commit()


async def claim(sessions, settings) -> Job | None:
    async with sessions() as db:
        row = await db.scalar(
            select(PhotoSearch)
            .where(
                PhotoSearch.status == "queued",
                PhotoSearch.available_at <= func.now(),
                PhotoSearch.expires_at > func.now(),
                PhotoSearch.attempts < settings.worker_max_attempts,
            )
            .order_by(PhotoSearch.available_at, PhotoSearch.created_at, PhotoSearch.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if row is None:
            return None
        row.status, row.lease_token = "running", uuid4()
        row.lease_until = utcnow() + timedelta(seconds=settings.worker_lease_seconds)
        row.attempts += 1
        row.started_at, row.error_code = utcnow(), None
        job = Job(row.id, row.lease_token, row.query_media_id, row.scope, row.limit)
        await db.commit()  # Release the claim transaction before matching.
        return job


async def execute(sessions, matcher: PhotoMatcher, job: Job):
    async with sessions() as db:
        query = await db.get(MediaAsset, job.query_media_id)
        if query is None or query.status != "ready":
            raise ValueError("query_media_unavailable")
        query_hash = query.sha256
        stmt = (
            select(Listing.id, MediaAsset.id)
            .join(
                MediaAsset,
                MediaAsset.listing_id == Listing.id,
            )
            .where(*public_conditions(), MediaAsset.status == "ready")
        )
        stmt = apply_filters(stmt, Filters.model_validate(job.scope))
        # One real cover photo per listing, selected by its position.
        stmt = stmt.ext(distinct_on(Listing.id)).order_by(
            Listing.id,
            MediaAsset.position,
            MediaAsset.id,
        )
        candidates = [Candidate(*row) for row in await db.execute(stmt)]
    return await matcher.match(query_hash, candidates, job.limit)


async def finalize(sessions, job, matches):
    async with sessions() as db:
        row = await db.scalar(
            select(PhotoSearch)
            .where(
                PhotoSearch.id == job.id,
                PhotoSearch.status == "running",
                PhotoSearch.lease_token == job.lease_token,
                PhotoSearch.lease_until > func.now(),
                PhotoSearch.expires_at > func.now(),
            )
            .with_for_update()
        )
        if row is None:
            return False
        await db.execute(delete(PhotoSearchResult).where(PhotoSearchResult.search_id == row.id))
        for rank, match in enumerate(matches, 1):
            db.add(
                PhotoSearchResult(
                    search_id=row.id,
                    listing_id=match.listing_id,
                    matched_media_id=match.media_id,
                    rank=rank,
                    similarity_score=None,
                )
            )
        row.status, row.finished_at = "completed", utcnow()
        row.lease_token = row.lease_until = None
        row.error_code = None
        await db.commit()
        return True


async def failed(sessions, settings, job, code, temporary=True):
    async with sessions() as db:
        row = await db.scalar(
            select(PhotoSearch)
            .where(
                PhotoSearch.id == job.id,
                PhotoSearch.status == "running",
                PhotoSearch.lease_token == job.lease_token,
                PhotoSearch.lease_until > func.now(),
            )
            .with_for_update()
        )
        if row is None:
            return False
        row.error_code = code
        if temporary and row.attempts < settings.worker_max_attempts and row.expires_at > utcnow():
            row.status = "queued"
            row.available_at = utcnow() + timedelta(seconds=2**row.attempts)
        else:
            row.status, row.finished_at = "failed", utcnow()
        row.lease_token = row.lease_until = None
        await db.commit()
        return True
