from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Kind = Literal["lost", "found", "adoption"]
Status = Literal["draft", "active", "resolved", "archived"]


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class ListingFields(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    species: str | None = Field(None, min_length=1, max_length=60)
    sex: str | None = Field(None, min_length=1, max_length=30)
    color: str | None = Field(None, min_length=1, max_length=100)
    breed: str | None = Field(None, max_length=100)
    title: str | None = Field(None, max_length=200)
    description: str | None = Field(None, max_length=10000)
    location: Location | None = None
    location_label: str | None = Field(None, max_length=300)
    event_at: datetime | None = None

    @field_validator("event_at")
    @classmethod
    def aware_event(cls, value):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("event_at must include timezone")
        return value


class ListingCreate(ListingFields):
    kind: Kind
    shelter_id: UUID | None = None

    @model_validator(mode="after")
    def kind_shelter(self):
        if (self.kind == "adoption") != (self.shelter_id is not None):
            raise ValueError("adoption requires shelter_id; lost/found exclude shelter_id")
        return self


class ListingPatch(ListingFields):
    pass


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Kind | None = None
    species: str | None = Field(None, max_length=60)
    sex: str | None = Field(None, max_length=30)
    color: str | None = Field(None, max_length=100)
    event_from: datetime | None = None
    event_to: datetime | None = None
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    radius_m: float | None = Field(None, gt=0, le=100000)

    @model_validator(mode="after")
    def validate_scope(self):
        geo = (self.latitude, self.longitude, self.radius_m)
        if any(x is not None for x in geo) and not all(x is not None for x in geo):
            raise ValueError("latitude, longitude and radius_m must be supplied together")
        for value in (self.event_from, self.event_to):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("dates must include timezone")
        if self.event_from and self.event_to and self.event_from > self.event_to:
            raise ValueError("event_from must not exceed event_to")
        return self


class ListingQuery(Filters):
    limit: int = Field(20, ge=1, le=100)
    cursor: str | None = Field(None, max_length=300)


class PhotoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    status: str
    content_type: str | None
    width: int | None
    height: int | None
    position: int | None


class ListingOut(ListingFields):
    id: UUID
    author_id: UUID
    shelter_id: UUID | None
    kind: Kind
    status: Status
    is_hidden: bool
    created_at: datetime
    updated_at: datetime
    photos: list[PhotoOut] = []


class MapPoint(BaseModel):
    id: UUID
    kind: Kind
    species: str | None
    title: str | None
    location: Location


class MapOut(BaseModel):
    items: list[MapPoint]
    truncated: bool
    limit: int
