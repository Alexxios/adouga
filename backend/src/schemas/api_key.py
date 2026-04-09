import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ApiKeyCreate(BaseModel):
    service_name: str = Field(min_length=1, max_length=100)


class ApiKeyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prefix: str
    service_name: str
    is_active: bool
    created_at: datetime


class ApiKeyCreated(ApiKeyRead):
    raw_key: str
