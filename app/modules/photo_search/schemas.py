from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import SEARCH_MAX_LIMIT
from app.cv import spec
from app.modules.listings.schemas import Filters, Kind, ListingOut

RANKING_NOTICE = (
    "Candidates are ordered for manual review. The order is not a probability that the same "
    "animal was found, and breed/colour similarity does not prove identity."
)


class ScopeFields(BaseModel):
    """Existing listing filters; applied identically to retrieval and evaluation scope."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    target_kind: Kind
    species: str | None = Field(None, max_length=60)
    sex: str | None = Field(None, max_length=30)
    color: str | None = Field(None, max_length=100)
    event_from: datetime | None = None
    event_to: datetime | None = None
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    radius_m: float | None = Field(None, gt=0, le=100000)

    def filters(self) -> Filters:
        return Filters(
            kind=self.target_kind,
            **self.model_dump(include=set(Filters.model_fields) - {"kind"}),
        )

    @model_validator(mode="after")
    def validate_scope(self):
        self.filters()
        return self


class SearchCreate(ScopeFields):
    """Legacy v1 contract (no breed/colour text, at most 5 candidates)."""

    query_media_id: UUID
    limit: int = Field(5, ge=1, le=5)


class BBoxIn(BaseModel):
    """Normalized crop on the canonical (EXIF-oriented) image; see crop rule crop-v1."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        self.to_spec()
        return self

    def to_spec(self) -> spec.BBox:
        try:
            return spec.BBox(self.x_min, self.y_min, self.x_max, self.y_max)
        except spec.InvalidCrop as exc:
            raise ValueError(str(exc)) from exc


class SearchCreateV2(ScopeFields):
    """Dog photo search: photo + user's breed text + user's coat colour text (+ optional crop).

    breed_text/coat_color_text are stored verbatim as user data (e.g. "немецкая овчарка +
    терьер"); they are not ML labels and are not hard filters.
    """

    query_media_id: UUID
    breed_text: str = Field(min_length=1, max_length=200)
    coat_color_text: str = Field(min_length=1, max_length=200)
    bbox: BBoxIn | None = None
    limit: int = Field(20, ge=1, le=SEARCH_MAX_LIMIT)

    @field_validator("breed_text", "coat_color_text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class SearchAccepted(BaseModel):
    id: UUID
    status: Literal["queued", "running", "completed", "failed"]


class SearchResultOut(BaseModel):
    listing: ListingOut
    matched_media_id: UUID
    rank: int
    similarity_score: float | None = Field(
        None, description="Always null: similarity scores are not exposed."
    )


class SearchOut(SearchAccepted):
    provider: str
    model_version: str | None
    is_mock: bool = True
    outcome: Literal["candidates", "rejected"] | None = None
    reason_code: str | None = None
    results: list[SearchResultOut]
    error: str | None
    created_at: datetime
    expires_at: datetime


class SearchResultV2(BaseModel):
    listing: ListingOut
    matched_media_id: UUID
    rank: int


class SearchQueryOut(BaseModel):
    breed_text: str | None
    coat_color_text: str | None
    bbox: list[float] | None


class SearchOutV2(SearchAccepted):
    api_version: str
    outcome: Literal["candidates", "rejected"] | None
    reason_code: str | None = Field(
        None, description="no_dog, multiple_dogs, invalid_crop, invalid_image, unsupported_image"
    )
    provider: str
    cv_mode: str | None
    is_mock: bool
    model_version: str | None
    embedding_space_id: str | None
    pipeline_id: str | None
    fusion_version: str | None
    query: SearchQueryOut
    results: list[SearchResultV2]
    error: str | None
    notice: str = RANKING_NOTICE
    created_at: datetime
    expires_at: datetime
