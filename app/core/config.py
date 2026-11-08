from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

DEV_SECRET = "unsafe-dev-only-pawly-secret-change-before-production"
SCHEMA_REVISION = "0002_cv_pgvector"
SEARCH_MAX_LIMIT = 50  # v2 candidate budget; legacy v1 keeps 1..5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    db_host: str = "db"
    db_port: int = 5432
    db_name: str = "pawly"
    db_user: str = "pawly"
    db_password: str = "unsafe-dev-password"
    jwt_secret: str = DEV_SECRET
    jwt_issuer: str = "pawly"
    jwt_audience: str = "pawly-api"
    access_ttl_seconds: int = Field(900, gt=0)
    refresh_ttl_seconds: int = Field(604800, gt=0)
    cookie_secure: bool = False
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    media_root: Path = Path("/data/media")
    # Real pretrained pipeline by default; "mock" only for plumbing tests (always labelled mock).
    cv_mode: Literal["mock", "pretrained_features", "trained_heads"] = "pretrained_features"
    cv_backbone: Literal["efficientnet_b0", "resnet50"] = "efficientnet_b0"
    cv_weights_dir: Path = Path("/opt/pawly/weights")
    cv_bundle_dir: Path | None = None
    # Experimental default, not a calibrated value; see docs/ml.md.
    cv_dog_score_threshold: float = Field(0.5, gt=0, lt=1)
    cv_fusion: Literal["embedding-v1", "embedding-rgb-v1", "embedding-rgb-heads-v1"] = (
        "embedding-v1"
    )
    cv_fusion_rgb_weight: float = Field(0.2, ge=0, le=1)
    cv_fusion_heads_weight: float = Field(0.1, ge=0, le=1)
    cv_rerank_pool: int = Field(200, ge=1, le=2000)
    cv_index_sweep_seconds: float = Field(10, gt=0)
    cv_index_batch: int = Field(200, ge=1, le=5000)
    cv_heartbeat_seconds: float = Field(10, gt=0)
    cv_torch_threads: int = Field(2, ge=1, le=64)
    search_ttl_seconds: int = Field(86400, gt=0)
    worker_poll_seconds: float = Field(1, gt=0)
    worker_timeout_seconds: float = Field(30, gt=0)
    worker_lease_seconds: int = Field(60, gt=0)
    worker_max_attempts: int = Field(3, gt=0)
    upload_max_bytes: int = Field(10 * 1024 * 1024, gt=0, le=10 * 1024 * 1024)
    image_max_pixels: int = Field(20_000_000, gt=0, le=20_000_000)

    @field_validator("cv_bundle_dir", mode="before")
    @classmethod
    def empty_bundle_is_none(cls, value):
        # Compose forwards an unset variable as an empty string.
        return None if value in ("", None) else value

    @model_validator(mode="after")
    def validate_security(self):
        if "*" in self.cors_origins:
            raise ValueError("CORS must use an explicit origin allowlist")
        if self.worker_lease_seconds <= self.worker_timeout_seconds:
            raise ValueError("worker lease must exceed task timeout")
        if self.cv_mode == "trained_heads" and self.cv_bundle_dir is None:
            raise ValueError("CV_MODE=trained_heads requires CV_BUNDLE_DIR")
        if self.cv_fusion == "embedding-rgb-heads-v1" and self.cv_mode != "trained_heads":
            raise ValueError("embedding-rgb-heads-v1 fusion requires CV_MODE=trained_heads")
        if self.cv_fusion_rgb_weight + self.cv_fusion_heads_weight > 1:
            raise ValueError("fusion weights must sum to at most 1")
        if self.app_env == "production":
            if self.jwt_secret == DEV_SECRET or len(self.jwt_secret) < 32:
                raise ValueError("Production requires a strong, explicit JWT secret")
            if self.debug or not self.cookie_secure:
                raise ValueError("Production requires debug=false and cookie_secure=true")
            if self.db_password == "unsafe-dev-password":
                raise ValueError("Production requires a non-development DB password")
            if any(not origin.startswith("https://") for origin in self.cors_origins):
                raise ValueError("Production CORS origins must use HTTPS")
        return self

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+asyncpg",
            username=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )
