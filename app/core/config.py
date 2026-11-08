from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

DEV_SECRET = "unsafe-dev-only-pawly-secret-change-before-production"
SCHEMA_REVISION = "0001_initial"


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
    cv_mode: Literal["mock"] = "mock"
    search_ttl_seconds: int = Field(86400, gt=0)
    worker_poll_seconds: float = Field(1, gt=0)
    worker_timeout_seconds: float = Field(30, gt=0)
    worker_lease_seconds: int = Field(60, gt=0)
    worker_max_attempts: int = Field(3, gt=0)
    upload_max_bytes: int = Field(10 * 1024 * 1024, gt=0, le=10 * 1024 * 1024)
    image_max_pixels: int = Field(20_000_000, gt=0, le=20_000_000)

    @model_validator(mode="after")
    def validate_security(self):
        if "*" in self.cors_origins:
            raise ValueError("CORS must use an explicit origin allowlist")
        if self.worker_lease_seconds <= self.worker_timeout_seconds:
            raise ValueError("worker lease must exceed task timeout")
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
