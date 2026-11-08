from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: str
    display_name: str
    is_admin: bool
    is_active: bool
    created_at: datetime


class ProfileOut(UserOut):
    phone: str | None = None
    preferred_channel: str | None = None


class ProfilePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str | None = Field(None, min_length=1, max_length=100)
    phone: str | None = Field(None, max_length=40)
    preferred_channel: str | None = Field(None, max_length=100)
