"""media_features persistence: provenance rows, indexing queue, supersede/revive, status counts.

Listing photo rows are the indexing queue (pending -> running -> indexed/rejected/failed), using
the same short-transaction claim, lease and fencing rules as photo_searches.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID, uuid4

from sqlalchemy import Text, cast, func, select, update
from sqlalchemy.exc import IntegrityError

from app.core.security import utcnow
from app.cv import spec
from app.cv.pipeline import PipelineSpec
from app.db.models import MediaAsset, MediaFeature


@dataclass(frozen=True)
class IndexJob:
    id: UUID
    lease_token: UUID
    media_id: UUID
    crop_bbox: list[float] | None
    input_sha256: str


def provenance(pipeline: PipelineSpec) -> dict:
    return {
        "pipeline_id": pipeline.pipeline_id,
        "embedding_space_id": pipeline.embedding_space_id,
        "cv_mode": pipeline.mode,
        "backbone": pipeline.backbone.name,
        "weights_id": pipeline.backbone.weights_id,
        "preprocessing_id": pipeline.backbone.preprocessing_id,
        "descriptor_version": spec.RGB_DESCRIPTOR_VERSION,
        "gate_id": pipeline.gate_id,
        "bundle_id": pipeline.bundle_id,
        "dimension": pipeline.backbone.dimension,
        "crop_rule": spec.CROP_RULE,
    }


def crop_fields(bbox_values, width, height) -> tuple[dict, str | None]:
    """Crop columns for a row, or a rejection reason when the stored crop is unusable."""
    try:
        crop = spec.crop_spec(spec.bbox_or_none(bbox_values), width, height)
    except spec.InvalidCrop:
        payload = spec.canonical_json({"rule": spec.CROP_RULE, "invalid": bbox_values})
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return {"crop_bbox": bbox_values, "crop_box": None, "crop_hash": digest}, spec.INVALID_CROP
    return {"crop_bbox": crop.bbox, "crop_box": list(crop.box), "crop_hash": crop.hash}, None


def current_feature_exists(pipeline_id: str):
    return (
        select(MediaFeature.id)
        .where(
            MediaFeature.media_id == MediaAsset.id,
            MediaFeature.pipeline_id == pipeline_id,
            MediaFeature.purpose == "listing",
            MediaFeature.superseded_at.is_(None),
        )
        .exists()
    )


async def schedule_missing(sessions, pipeline: PipelineSpec, batch: int) -> int:
    """Enqueue ready listing photos without a current row for this pipeline. Idempotent."""
    created = 0
    async with sessions() as db:
        rows = (
            await db.execute(
                select(
                    MediaAsset.id,
                    MediaAsset.width,
                    MediaAsset.height,
                    MediaAsset.sha256,
                    MediaAsset.crop_bbox,
                )
                .where(
                    MediaAsset.purpose == "listing",
                    MediaAsset.status == "ready",
                    ~current_feature_exists(pipeline.pipeline_id),
                )
                .order_by(MediaAsset.created_at, MediaAsset.id)
                .limit(batch)
            )
        ).all()
        for media_id, width, height, sha256, bbox in rows:
            fields, reason = crop_fields(bbox, width, height)
            try:
                async with db.begin_nested():
                    revived = await db.execute(
                        update(MediaFeature)
                        .where(
                            MediaFeature.media_id == media_id,
                            MediaFeature.pipeline_id == pipeline.pipeline_id,
                            MediaFeature.crop_hash == fields["crop_hash"],
                            MediaFeature.purpose == "listing",
                            MediaFeature.input_sha256 == sha256,
                        )
                        .values(superseded_at=None)
                    )
                    if revived.rowcount == 0:
                        db.add(
                            MediaFeature(
                                id=uuid4(),
                                media_id=media_id,
                                purpose="listing",
                                status="rejected" if reason else "pending",
                                reason_code=reason,
                                input_sha256=sha256,
                                attempts=0,
                                **fields,
                                **provenance(pipeline),
                            )
                        )
                        await db.flush()
                    created += 1
            except IntegrityError:
                continue  # Another scheduler won the race; the current row already exists.
        await db.commit()
    return created


async def supersede_media(db, media_id: UUID) -> int:
    """Called inside the caller's transaction when the crop changes or a photo is deleted."""
    result = await db.execute(
        update(MediaFeature)
        .where(
            MediaFeature.media_id == media_id,
            MediaFeature.purpose == "listing",
            MediaFeature.superseded_at.is_(None),
        )
        .values(superseded_at=func.now())
    )
    return result.rowcount


async def recover(sessions, settings, pipeline_id: str) -> None:
    async with sessions() as db:
        rows = await db.scalars(
            select(MediaFeature)
            .where(
                MediaFeature.pipeline_id == pipeline_id,
                MediaFeature.status == "running",
                MediaFeature.lease_until <= func.now(),
            )
            .with_for_update(skip_locked=True)
        )
        now = utcnow()
        for row in rows:
            if row.attempts >= settings.worker_max_attempts:
                row.status, row.error_code = "failed", "attempts_exhausted"
            else:
                row.status = "pending"
                row.available_at = now + timedelta(seconds=2**row.attempts)
            row.lease_token = row.lease_until = None
        await db.commit()


async def claim(sessions, settings, pipeline_id: str) -> IndexJob | None:
    async with sessions() as db:
        row = await db.scalar(
            select(MediaFeature)
            .where(
                MediaFeature.pipeline_id == pipeline_id,
                MediaFeature.purpose == "listing",
                MediaFeature.status == "pending",
                MediaFeature.superseded_at.is_(None),
                MediaFeature.available_at <= func.now(),
                MediaFeature.attempts < settings.worker_max_attempts,
            )
            .order_by(MediaFeature.available_at, MediaFeature.created_at, MediaFeature.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if row is None:
            return None
        row.status, row.lease_token = "running", uuid4()
        row.lease_until = utcnow() + timedelta(seconds=settings.worker_lease_seconds)
        row.attempts += 1
        row.error_code = None
        job = IndexJob(row.id, row.lease_token, row.media_id, row.crop_bbox, row.input_sha256)
        await db.commit()  # No transaction is held during inference.
        return job


def _fenced(job: IndexJob):
    return (
        select(MediaFeature)
        .where(
            MediaFeature.id == job.id,
            MediaFeature.status == "running",
            MediaFeature.lease_token == job.lease_token,
            MediaFeature.lease_until > func.now(),
        )
        .with_for_update()
    )


def apply_extraction(row: MediaFeature, extraction) -> None:
    if extraction.rejected:
        row.status, row.reason_code = "rejected", extraction.rejected
        row.embedding = None
        return
    row.status, row.reason_code = "indexed", None
    row.embedding = extraction.embedding
    row.rgb_histogram = extraction.rgb_histogram
    row.dominant_colors = extraction.dominant_colors
    row.breed_prediction = extraction.breed_prediction
    row.color_prediction = extraction.color_prediction
    row.extraction_ms = extraction.timings_ms.get("total")
    row.indexed_at = utcnow()


async def complete(sessions, job: IndexJob, extraction, user_texts) -> bool:
    async with sessions() as db:
        row = await db.scalar(_fenced(job))
        if row is None:
            return False  # Lease expired or another attempt owns the row now.
        apply_extraction(row, extraction)
        row.user_breed_text, row.user_color_text = user_texts
        row.error_code = None
        row.lease_token = row.lease_until = None
        await db.commit()
        return True


async def fail(sessions, settings, job: IndexJob, code: str, temporary=True) -> bool:
    async with sessions() as db:
        row = await db.scalar(_fenced(job))
        if row is None:
            return False
        row.error_code = code
        if temporary and row.attempts < settings.worker_max_attempts:
            row.status = "pending"
            row.available_at = utcnow() + timedelta(seconds=2**row.attempts)
        else:
            row.status = "failed"
        row.lease_token = row.lease_until = None
        await db.commit()
        return True


async def load_embedding(db, feature_id: UUID) -> list[float] | None:
    raw = await db.scalar(
        select(cast(MediaFeature.embedding, Text)).where(MediaFeature.id == feature_id)
    )
    return None if raw is None else [float(v) for v in json.loads(raw)]


async def counts(db, pipeline_id: str) -> dict:
    rows = await db.execute(
        select(MediaFeature.status, func.count())
        .join(MediaAsset, MediaAsset.id == MediaFeature.media_id)
        .where(
            MediaFeature.pipeline_id == pipeline_id,
            MediaFeature.purpose == "listing",
            MediaFeature.superseded_at.is_(None),
            MediaAsset.status == "ready",
        )
        .group_by(MediaFeature.status)
    )
    result = {status: 0 for status in ("pending", "running", "indexed", "rejected", "failed")}
    result.update({status: count for status, count in rows})
    result["listing_photos_ready"] = await db.scalar(
        select(func.count())
        .select_from(MediaAsset)
        .where(MediaAsset.purpose == "listing", MediaAsset.status == "ready")
    )
    return result


async def reindex(sessions, pipeline_id: str, failed_only: bool = False) -> int:
    """Force recomputation for the current pipeline (e.g. after a bug fix, same version)."""
    async with sessions() as db:
        conditions = [
            MediaFeature.pipeline_id == pipeline_id,
            MediaFeature.purpose == "listing",
            MediaFeature.superseded_at.is_(None),
            MediaFeature.status != "running",
        ]
        if failed_only:
            conditions.append(MediaFeature.status == "failed")
        result = await db.execute(
            update(MediaFeature)
            .where(*conditions)
            .values(
                status="pending",
                reason_code=None,
                error_code=None,
                attempts=0,
                available_at=func.now(),
            )
        )
        await db.commit()
        return result.rowcount
