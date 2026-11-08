"""ML readiness and index status (aggregate counts only; no user data)."""

from fastapi import APIRouter, Request
from sqlalchemy import select

from app.api.dependencies import DB
from app.core.security import utcnow
from app.db.models import CVWorkerStatus
from app.modules.cv_index import repository as repo

router = APIRouter(prefix="/cv", tags=["cv status"])

# A crashed worker stops counting after ~4 missed heartbeats (interval 10 s, job timeout 30 s).
STALE_AFTER_SECONDS = 45


async def status_payload(db, settings) -> dict:
    workers = list(
        (
            await db.scalars(select(CVWorkerStatus).order_by(CVWorkerStatus.heartbeat_at.desc()))
        ).all()
    )
    now = utcnow()
    fresh = [w for w in workers if (now - w.heartbeat_at).total_seconds() <= STALE_AFTER_SECONDS]
    active = next((w for w in fresh if w.state == "ready" and w.cv_mode == settings.cv_mode), None)
    details = (active.details or {}) if active else {}
    index = await repo.counts(db, active.pipeline_id) if active and active.pipeline_id else None
    return {
        "configured_mode": settings.cv_mode,
        "ready": active is not None,
        "provider": active.provider if active else None,
        "model_stage": active.cv_mode if active else None,
        "is_mock": settings.cv_mode == "mock",
        "pipeline_id": active.pipeline_id if active else None,
        "embedding_space_id": active.embedding_space_id if active else None,
        "fusion": details.get("fusion"),
        "trained_heads": {
            "available": bool(active and active.bundle_id),
            "bundle_id": active.bundle_id if active else None,
            "note": None
            if active and active.bundle_id
            else "No trained breed/colour heads are active; no breed/colour predictions exist.",
        },
        "workers": [
            {
                "state": w.state,
                "cv_mode": w.cv_mode,
                "error_code": w.error_code,
                "heartbeat_age_seconds": round((now - w.heartbeat_at).total_seconds(), 1),
            }
            for w in fresh
        ],
        "index": index,
    }


@router.get("/status")
async def status(db: DB, request: Request):
    """Active embedding space, provider/model stage and indexing counts for the active pipeline."""
    return await status_payload(db, request.app.state.settings)
