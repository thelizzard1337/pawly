from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ShelterCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field("", max_length=10000)
    city: str = Field(min_length=1, max_length=200)
    address_public: str = Field("", max_length=300)


class ShelterPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=10000)
    city: str | None = Field(None, min_length=1, max_length=200)
    address_public: str | None = Field(None, max_length=300)


class ShelterOut(ShelterCreate):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    verification_status: Literal["pending", "verified", "rejected"]
    created_at: datetime
    updated_at: datetime


class VerifyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verification_status: Literal["verified", "rejected"] = "verified"
