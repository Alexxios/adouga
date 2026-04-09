import math
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.deps import get_db, verify_api_key
from src.models.api_key import ApiKey
from src.models.prediction import Prediction
from src.schemas.prediction import PaginatedPredictions, PredictionRead

router = APIRouter(prefix="/external", tags=["external"])


@router.get("/predictions", response_model=PaginatedPredictions)
async def list_predictions(
    api_key: ApiKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
    user_id: uuid.UUID | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    predicted_class: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    from src.main import EXTERNAL_REQUESTS

    EXTERNAL_REQUESTS.labels(service_name=api_key.service_name).inc()

    query = select(Prediction)
    count_query = select(func.count()).select_from(Prediction)

    if user_id:
        query = query.where(Prediction.user_id == user_id)
        count_query = count_query.where(Prediction.user_id == user_id)
    if date_from:
        query = query.where(Prediction.timestamp >= date_from)
        count_query = count_query.where(Prediction.timestamp >= date_from)
    if date_to:
        query = query.where(Prediction.timestamp <= date_to)
        count_query = count_query.where(Prediction.timestamp <= date_to)
    if predicted_class:
        query = query.where(Prediction.predicted_class == predicted_class)
        count_query = count_query.where(Prediction.predicted_class == predicted_class)

    total = (await db.execute(count_query)).scalar() or 0
    query = query.order_by(Prediction.timestamp.desc()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)

    return PaginatedPredictions(
        items=list(result.scalars().all()),
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/predictions/{prediction_id}", response_model=PredictionRead)
async def get_prediction(
    prediction_id: uuid.UUID,
    api_key: ApiKey = Depends(verify_api_key),
    db: AsyncSession = Depends(get_db),
):
    from src.main import EXTERNAL_REQUESTS

    EXTERNAL_REQUESTS.labels(service_name=api_key.service_name).inc()

    result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return prediction
