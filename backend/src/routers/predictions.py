import math
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.deps import get_current_user, get_db
from src.models.prediction import Prediction
from src.models.user import User
from src.schemas.prediction import PaginatedPredictions, PredictionCreate, PredictionRead

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("", response_model=PredictionRead, status_code=status.HTTP_201_CREATED)
async def create_prediction(
    data: PredictionCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from src.main import PREDICTION_UPLOADS

    prediction = Prediction(user_id=user.id, **data.model_dump())
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)
    PREDICTION_UPLOADS.labels(predicted_class=data.predicted_class).inc()
    return prediction


@router.get("", response_model=PaginatedPredictions)
async def list_predictions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    predicted_class: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    query = select(Prediction).where(Prediction.user_id == user.id)
    count_query = select(func.count()).select_from(Prediction).where(Prediction.user_id == user.id)

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


@router.get("/{prediction_id}", response_model=PredictionRead)
async def get_prediction(
    prediction_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user.id)
    )
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return prediction


@router.delete("/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prediction(
    prediction_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user.id)
    )
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    await db.delete(prediction)
    await db.commit()
