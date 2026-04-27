import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_version_id: Mapped[str] = mapped_column(
        String, ForeignKey("model_versions.id"), nullable=False
    )
    engine_id: Mapped[int] = mapped_column(Integer, nullable=False)
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    lower_bound: Mapped[float] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[float] = mapped_column(Float, nullable=True)
    true_label: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    model_version: Mapped["ModelVersion"] = relationship(
        back_populates="predictions"
    )