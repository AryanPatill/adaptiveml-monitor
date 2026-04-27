import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base
import enum


class ModelType(str, enum.Enum):
    baseline = "baseline"
    continual = "continual"


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiments.id"), nullable=False
    )
    model_type: Mapped[ModelType] = mapped_column(
        Enum(ModelType), nullable=False
    )
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=True)
    precision: Mapped[float] = mapped_column(Float, nullable=True)
    recall: Mapped[float] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float] = mapped_column(Float, nullable=True)
    coverage: Mapped[float] = mapped_column(Float, nullable=True)
    avg_interval_width: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    experiment: Mapped["Experiment"] = relationship(
        back_populates="model_versions"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )