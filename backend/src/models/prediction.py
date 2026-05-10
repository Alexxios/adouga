import uuid
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_user_timestamp", "user_id", "timestamp"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    predicted_class: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float]
    probabilities: Mapped[dict] = mapped_column(JSON)
    timestamp: Mapped[datetime] = mapped_column(index=True)

    user = relationship("User", back_populates="predictions")
