"""Real PhotoMatcher: pretrained embeddings + exact cosine search in PostgreSQL/pgvector.

Ranking (fusion-versioned):
  embedding-v1            score = cos(q, c)                         (baseline)
  embedding-rgb-v1        score = (1-w)*(1+cos)/2 + w*HI(rgb_q, rgb_c)
  embedding-rgb-heads-v1  score = (1-w-v)*(1+cos)/2 + w*HI(rgb) + v*(BC(breed)+HI(color_pred))/2
where HI is histogram intersection and BC the Bhattacharyya coefficient (both in [0, 1]).
Listing score = maximum over its indexed photos ("max image similarity"); matched_media_id is
the photo that achieved it. Exact search: every eligible row is compared, no ANN index.
"""

import math
from uuid import uuid4

from sqlalchemy import Float, bindparam, func, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.exc import IntegrityError

from app.cv import spec
from app.db.models import Listing, MediaAsset, MediaFeature, Vector
from app.integrations.cv.matcher import MatchOutcome, RankedCandidate
from app.integrations.cv.runtime import CVRuntime, CVUnavailable
from app.modules.cv_index import repository as feature_repo
from app.modules.cv_index.indexing import read_canonical
from app.modules.listings.repository import apply_filters, public_conditions
from app.modules.listings.schemas import Filters


def histogram_intersection(a, b) -> float:
    return float(sum(min(x, y) for x, y in zip(a, b, strict=True)))


def bhattacharyya(a, b) -> float:
    return float(sum(math.sqrt(max(x, 0.0) * max(y, 0.0)) for x, y in zip(a, b, strict=True)))


def fuse(fusion: str, settings, cosine: float, query, candidate) -> float:
    if fusion == "embedding-v1":
        return cosine
    visual = (1.0 + cosine) / 2.0
    rgb = histogram_intersection(query["rgb"], candidate["rgb"])
    w_rgb = settings.cv_fusion_rgb_weight
    if fusion == "embedding-rgb-v1":
        return (1.0 - w_rgb) * visual + w_rgb * rgb
    if fusion == "embedding-rgb-heads-v1":
        w_heads = settings.cv_fusion_heads_weight
        breed = bhattacharyya(query["breed"], candidate["breed"])
        color = histogram_intersection(query["color"], candidate["color"])
        return (1.0 - w_rgb - w_heads) * visual + w_rgb * rgb + w_heads * (breed + color) / 2.0
    raise ValueError(f"unknown fusion {fusion}")


class VectorPhotoMatcher:
    def __init__(self, runtime: CVRuntime | None, storage, settings, load_error: str | None = None):
        self.runtime, self.storage, self.settings = runtime, storage, settings
        self.load_error = load_error
        self.provider = runtime.pipeline.provider if runtime else "pawly-cv"

    async def query_features(self, sessions, job, pipeline):
        """Reuse or compute features of the query crop; returns (feature_id, row dict|reason)."""
        async with sessions() as db:
            media = await db.get(MediaAsset, job.query_media_id)
            if media is None or media.status != "ready" or media.purpose != "search":
                raise ValueError("query_media_unavailable")
            fields, reason = feature_repo.crop_fields(job.crop_bbox, media.width, media.height)
            key, sha256 = media.storage_key, media.sha256
            existing = await db.scalar(
                select(MediaFeature).where(
                    MediaFeature.media_id == media.id,
                    MediaFeature.crop_hash == fields["crop_hash"],
                    MediaFeature.pipeline_id == pipeline.pipeline_id,
                    MediaFeature.status.in_(["indexed", "rejected"]),
                )
            )
            if existing is not None:
                return existing.id, await self._row(db, existing)
        if reason is None:
            raw = await read_canonical(self.storage, key, sha256)
            extraction = await self.runtime.extract(raw, spec.bbox_or_none(job.crop_bbox))
        else:
            extraction = None
        async with sessions() as db:
            values = {
                "id": uuid4(),
                "media_id": job.query_media_id,
                "purpose": "query",
                "input_sha256": sha256,
                "attempts": 0,
                "user_breed_text": job.breed_text,
                "user_color_text": job.coat_color_text,
                **fields,
                **feature_repo.provenance(pipeline),
            }
            row = MediaFeature(**values, status="pending")
            if extraction is None:
                row.status, row.reason_code = "rejected", reason
            else:
                feature_repo.apply_extraction(row, extraction)
            try:
                async with db.begin_nested():
                    db.add(row)
                    await db.flush()
                await db.commit()
            except IntegrityError:
                await db.rollback()  # Concurrent attempt stored the same crop first.
                row = await db.scalar(
                    select(MediaFeature).where(
                        MediaFeature.media_id == job.query_media_id,
                        MediaFeature.crop_hash == fields["crop_hash"],
                        MediaFeature.pipeline_id == pipeline.pipeline_id,
                    )
                )
            return row.id, await self._row(db, row)

    async def _row(self, db, row: MediaFeature) -> dict:
        if row.status == "rejected":
            return {"rejected": row.reason_code}
        if row.status != "indexed":
            raise RuntimeError("query features not ready")
        return {
            "rejected": None,
            "embedding": await feature_repo.load_embedding(db, row.id),
            "rgb": row.rgb_histogram,
            "breed": (row.breed_prediction or {}).get("distribution"),
            "color": row.color_prediction,
        }

    async def search(self, sessions, job) -> MatchOutcome:
        if self.runtime is None:
            raise CVUnavailable(self.load_error or "cv_runtime_not_loaded")
        pipeline = self.runtime.pipeline
        base = {
            "provider": pipeline.provider,
            "cv_mode": pipeline.mode,
            "model_version": pipeline.model_version,
            "pipeline_id": pipeline.pipeline_id,
            "embedding_space_id": pipeline.embedding_space_id,
            "fusion_version": pipeline.fusion,
        }
        feature_id, query = await self.query_features(sessions, job, pipeline)
        if query["rejected"]:
            return MatchOutcome(
                outcome="rejected",
                reason_code=query["rejected"],
                query_feature_id=feature_id,
                **base,
            )
        embedding = query["embedding"]
        if len(embedding) != pipeline.backbone.dimension:
            raise RuntimeError("query embedding dimension does not match the active space")
        fusion = pipeline.fusion
        pool = (
            job.limit if fusion == "embedding-v1" else max(job.limit, self.settings.cv_rerank_pool)
        )
        rows, corpus = await self.nearest(sessions, job, pipeline, embedding, pool)
        scored = []
        for row in rows:
            cosine = 1.0 - float(row.distance)
            candidate = {
                "rgb": row.rgb_histogram,
                "color": row.color_prediction,
                "breed": (row.breed_prediction or {}).get("distribution"),
            }
            score = fuse(fusion, self.settings, cosine, query, candidate)
            scored.append(RankedCandidate(row.listing_id, row.media_id, score))
        # Stable order for equal scores: listing id, then photo id.
        scored.sort(key=lambda c: (-c.score, str(c.listing_id), str(c.media_id)))
        return MatchOutcome(
            outcome="candidates",
            matches=scored[: job.limit],
            query_feature_id=feature_id,
            meta={
                "eligible_listings": corpus,
                "pool": len(rows),
                "aggregation": "max-image-similarity",
                "retrieval": "exact-cosine-pgvector",
                "user_text_used_for_ranking": False,
            },
            **base,
        )

    async def nearest(self, sessions, job, pipeline, embedding, pool: int):
        query = bindparam("query_embedding", embedding, type_=Vector())
        distance = MediaFeature.embedding.op("<=>", return_type=Float)(query)
        stmt = (
            select(
                Listing.id.label("listing_id"),
                MediaFeature.media_id.label("media_id"),
                distance.label("distance"),
                MediaFeature.rgb_histogram,
                MediaFeature.color_prediction,
                MediaFeature.breed_prediction,
            )
            .join(MediaAsset, MediaAsset.id == MediaFeature.media_id)
            .join(Listing, Listing.id == MediaAsset.listing_id)
            .where(
                *public_conditions(),
                MediaAsset.status == "ready",
                MediaAsset.purpose == "listing",
                MediaFeature.purpose == "listing",
                MediaFeature.status == "indexed",
                MediaFeature.superseded_at.is_(None),
                # Never compare across embedding spaces, dimensions or pipeline versions.
                MediaFeature.pipeline_id == pipeline.pipeline_id,
                MediaFeature.embedding_space_id == pipeline.embedding_space_id,
                MediaFeature.dimension == pipeline.backbone.dimension,
                MediaFeature.media_id != job.query_media_id,
            )
        )
        stmt = apply_filters(stmt, Filters.model_validate(job.scope))
        best = (
            stmt.ext(distinct_on(Listing.id))
            .order_by(Listing.id, distance, MediaFeature.media_id)
            .subquery()
        )
        ranked = (
            select(best, func.count().over().label("eligible"))
            .order_by(best.c.distance, best.c.listing_id, best.c.media_id)
            .limit(pool)
        )
        async with sessions() as db:
            rows = (await db.execute(ranked)).all()
        corpus = int(rows[0].eligible) if rows else 0
        return rows, corpus
