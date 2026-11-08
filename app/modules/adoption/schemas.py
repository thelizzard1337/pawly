from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=3000)


class AdoptionOut(MessageIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    listing_id: UUID
    applicant_id: UUID
    status: Literal["pending", "approved", "rejected", "withdrawn"]
    reviewed_by: UUID | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ContactOut(MessageIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    listing_id: UUID
    sender_id: UUID
    status: str
    created_at: datetime
