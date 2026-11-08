from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.api.router import router, router_v2
from app.core.config import SCHEMA_REVISION, Settings
from app.db.session import make_database
from app.integrations.storage.local import LocalFileStorage


def create_app(settings: Settings | None = None):
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        engine, sessions = make_database(config)
        storage = LocalFileStorage(config.media_root)
        app.state.engine, app.state.sessions = engine, sessions
        app.state.settings, app.state.storage = config, storage
        try:
            await run_in_threadpool(storage.check)
            yield
        finally:
            await engine.dispose()

    app = FastAPI(
        title="PawLy API",
        version="0.1.0",
        debug=config.debug,
        lifespan=lifespan,
        description="Lost/found and shelter adoption. Dog photo search (v2) returns candidate "
        "listings ranked by pretrained visual embeddings; the order is for manual review and is "
        "not a probability of identity. CV_MODE=mock is labelled provider mock-v1.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token", "Idempotency-Key"],
        expose_headers=["Location"],
    )

    @app.middleware("http")
    async def privacy_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc):
        return JSONResponse(status_code=503, content={"detail": {"code": "database_unavailable"}})

    @app.get("/health/live", tags=["health"])
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready(request: Request):
        try:
            async with request.app.state.sessions() as db:
                revision = (
                    await db.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar_one()
            if revision != SCHEMA_REVISION:
                raise ValueError("Schema mismatch")
            await run_in_threadpool(request.app.state.storage.check)
        except (SQLAlchemyError, OSError, ValueError):
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "schema_revision": revision}

    @app.get("/health/ml", tags=["health"])
    async def ml_ready(request: Request):
        """ML readiness, separate from CRUD readiness: a fresh worker heartbeat in CV_MODE."""
        from app.modules.cv_index.router import status_payload

        try:
            async with request.app.state.sessions() as db:
                payload = await status_payload(db, request.app.state.settings)
        except SQLAlchemyError:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        body = {
            "status": "ready" if payload["ready"] else "not_ready",
            "configured_mode": payload["configured_mode"],
            "provider": payload["provider"],
            "pipeline_id": payload["pipeline_id"],
            "workers": payload["workers"],
        }
        return JSONResponse(status_code=200 if payload["ready"] else 503, content=body)

    app.include_router(router)
    app.include_router(router_v2)
    return app


app = create_app()
