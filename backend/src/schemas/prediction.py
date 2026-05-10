import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionCreate(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    timestamp: datetime


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
