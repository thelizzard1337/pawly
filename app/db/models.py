from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from app.db.base import Base, Created, Identity, Updated


class Geography(UserDefinedType):
    cache_ok = True

    def __init__(self, geometry_type="Point", srid=4326):
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **kw):
        return f"geography({self.geometry_type},{self.srid})"


class User(Identity, Updated, Base):
    __tablename__ = "users"
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    display_name: Mapped[str] = mapped_column(String(100))
    is_admin: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    is_active: Mapped[bool] = mapped_column(default=True, server_default=text("true"))


class UserContact(Base):
    __tablename__ = "user_contacts"
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), primary_key=True)
    phone: Mapped[str | None] = mapped_column(String(40))
    preferred_channel: Mapped[str | None] = mapped_column(String(100))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        onupdate=text("now()"),
    )


class RefreshSession(Identity, Created, Base):
    __tablename__ = "refresh_sessions"
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    family_id: Mapped[UUID] = mapped_column(index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    replaced_by_id: Mapped[UUID | None] = mapped_column(ForeignKey("refresh_sessions.id"))


class Shelter(Identity, Updated, Base):
    __tablename__ = "shelters"
    __table_args__ = (
        CheckConstraint(
            "verification_status IN ('pending','verified','rejected')",
            name="verification",
        ),
    )
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    city: Mapped[str] = mapped_column(String(200))
    address_public: Mapped[str] = mapped_column(String(300), default="")
    verification_status: Mapped[str] = mapped_column(String(20), default="pending")


class ShelterMember(Base):
    __tablename__ = "shelter_members"
    __table_args__ = (CheckConstraint("role = 'manager'", name="role"),)
    shelter_id: Mapped[UUID] = mapped_column(ForeignKey("shelters.id"), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), primary_key=True, index=True)
    role: Mapped[str] = mapped_column(String(20), default="manager")


class Listing(Identity, Updated, Base):
    __tablename__ = "listings"
    __table_args__ = (
        CheckConstraint("kind IN ('lost','found','adoption')", name="kind"),
        CheckConstraint("status IN ('draft','active','resolved','archived')", name="status"),
        CheckConstraint("(kind = 'adoption') = (shelter_id IS NOT NULL)", name="kind_shelter"),
        CheckConstraint(
            "status <> 'active' OR (species IS NOT NULL AND length(trim(species)) > 0 "
            "AND location IS NOT NULL AND description IS NOT NULL "
            "AND length(trim(description)) > 0 AND (kind='adoption' OR event_at IS NOT NULL))",
            name="active_complete",
        ),
        Index("ix_listings_filters", "kind", "status", "species", "created_at"),
        Index("ix_listings_cursor", "created_at", "id"),
        Index("ix_listings_location", "location", postgresql_using="gist"),
    )
    author_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    shelter_id: Mapped[UUID | None] = mapped_column(ForeignKey("shelters.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="draft")
    is_hidden: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    species: Mapped[str | None] = mapped_column(String(60))
    sex: Mapped[str | None] = mapped_column(String(30))
    color: Mapped[str | None] = mapped_column(String(100))
    breed: Mapped[str | None] = mapped_column(String(100))
    title: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Geography())
    location_label: Mapped[str | None] = mapped_column(String(300))
    event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MediaAsset(Identity, Created, Base):
    __tablename__ = "media_assets"
    __table_args__ = (
        CheckConstraint("purpose IN ('listing','search')", name="purpose"),
        CheckConstraint("status IN ('pending','ready','failed','deleted')", name="status"),
        CheckConstraint(
            "(purpose='listing' AND listing_id IS NOT NULL AND position IS NOT NULL "
            "AND position >= 0) OR "
            "(purpose='search' AND listing_id IS NULL AND position IS NULL)",
            name="attachment",
        ),
        CheckConstraint(
            "status <> 'ready' OR (sha256 IS NOT NULL AND content_type IS NOT NULL "
            "AND size_bytes IS NOT NULL AND width IS NOT NULL AND height IS NOT NULL "
            "AND size_bytes > 0 AND width > 0 AND height > 0)",
            name="ready_metadata",
        ),
        Index(
            "uq_media_position",
            "listing_id",
            "position",
            unique=True,
            postgresql_where=text("status IN ('pending','ready')"),
        ),
        Index("ix_media_owner_created", "owner_id", "created_at"),
        Index("ix_media_status_created", "status", "created_at"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    listing_id: Mapped[UUID | None] = mapped_column(ForeignKey("listings.id"), index=True)
    purpose: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    storage_key: Mapped[str] = mapped_column(String(100), unique=True)
    sha256: Mapped[str | None] = mapped_column(String(64))
    content_type: Mapped[str | None] = mapped_column(String(50))
    size_bytes: Mapped[int | None]
    width: Mapped[int | None]
    height: Mapped[int | None]
    position: Mapped[int | None]
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class AdoptionRequest(Identity, Updated, Base):
    __tablename__ = "adoption_requests"
    __table_args__ = (
        CheckConstraint("status IN ('pending','approved','rejected','withdrawn')", name="status"),
        Index(
            "uq_adoption_pending",
            "listing_id",
            "applicant_id",
            unique=True,
            postgresql_where=text("status='pending'"),
        ),
        Index(
            "uq_adoption_approved",
            "listing_id",
            unique=True,
            postgresql_where=text("status='approved'"),
        ),
        Index("ix_adoption_listing_status", "listing_id", "status"),
        Index("ix_adoption_applicant_created", "applicant_id", "created_at"),
    )
    listing_id: Mapped[UUID] = mapped_column(ForeignKey("listings.id"))
    applicant_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    reviewed_by: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ContactRequest(Identity, Created, Base):
    __tablename__ = "contact_requests"
    __table_args__ = (CheckConstraint("status='pending'", name="status"),)
    listing_id: Mapped[UUID] = mapped_column(ForeignKey("listings.id"), index=True)
    sender_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending")


class PhotoSearch(Identity, Created, Base):
    __tablename__ = "photo_searches"
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','completed','failed')", name="status"),
        CheckConstraint('"limit" BETWEEN 1 AND 5', name="limit"),
        CheckConstraint("attempts >= 0", name="attempts"),
        CheckConstraint(
            "status <> 'running' OR (lease_token IS NOT NULL AND lease_until IS NOT NULL)",
            name="running_lease",
        ),
        Index(
            "uq_search_idempotency",
            "owner_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL AND idempotency_key <> ''"),
        ),
        Index("ix_search_claim", "status", "available_at"),
        Index("ix_search_lease", "status", "lease_until"),
        Index("ix_search_owner_created", "owner_id", "created_at"),
    )
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    query_media_id: Mapped[UUID] = mapped_column(ForeignKey("media_assets.id"), index=True)
    scope: Mapped[dict] = mapped_column(JSONB)
    limit: Mapped[int]
    status: Mapped[str] = mapped_column(String(20), default="queued")
    provider: Mapped[str] = mapped_column(String(50), default="mock-v1")
    model_version: Mapped[str | None] = mapped_column(String(100))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(100))
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    request_fingerprint: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class PhotoSearchResult(Base):
    __tablename__ = "photo_search_results"
    __table_args__ = (
        UniqueConstraint("search_id", "rank"),
        CheckConstraint("rank BETWEEN 1 AND 5", name="rank"),
        CheckConstraint("similarity_score IS NULL", name="mock_score"),
    )
    search_id: Mapped[UUID] = mapped_column(
        ForeignKey("photo_searches.id", ondelete="CASCADE"),
        primary_key=True,
    )
    listing_id: Mapped[UUID] = mapped_column(ForeignKey("listings.id"), primary_key=True)
    matched_media_id: Mapped[UUID] = mapped_column(ForeignKey("media_assets.id"))
    rank: Mapped[int]
    similarity_score: Mapped[float | None]
