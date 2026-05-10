import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionCreate(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def _to_utc_naive(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            v = v.astimezone(timezone.utc).replace(tzinfo=None)
        return v


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    timestamp: datetime
    created_at: datetime


class PaginatedPredictions(BaseModel):
    items: list[PredictionRead]
    total: int
    page: int
    page_size: int
    pages: int
