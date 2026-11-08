from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.modules.listings.schemas import Filters, Kind, ListingOut


class SearchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    query_media_id: UUID
    target_kind: Kind
    species: str | None = Field(None, max_length=60)
    sex: str | None = Field(None, max_length=30)
    color: str | None = Field(None, max_length=100)
    event_from: datetime | None = None
    event_to: datetime | None = None
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    radius_m: float | None = Field(None, gt=0, le=100000)
    limit: int = Field(5, ge=1, le=5)

    def filters(self) -> Filters:
        return Filters(
            kind=self.target_kind,
            **self.model_dump(
                exclude={"query_media_id", "target_kind", "limit"},
            ),
        )

    @model_validator(mode="after")
    def validate_scope(self):
        self.filters()
        return self


class SearchAccepted(BaseModel):
    id: UUID
    status: Literal["queued", "running", "completed", "failed"]


class SearchResultOut(BaseModel):
    listing: ListingOut
    matched_media_id: UUID
    rank: int
    similarity_score: float | None = None


class SearchOut(SearchAccepted):
    provider: str
    model_version: str | None
    is_mock: bool = True
    results: list[SearchResultOut]
    error: str | None
    created_at: datetime
    expires_at: datetime
