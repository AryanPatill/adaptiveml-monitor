import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Float, Integer, ForeignKey, Enum, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base
import enum


class LabelStatus(str, enum.Enum):
    pending = "pending"
    reviewed = "reviewed"


class LabelQueue(Base):
    __tablename__ = "label_queue"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id: Mapped[str] = mapped_column(
        String, ForeignKey("experiments.id"), nullable=False
    )
    engine_id: Mapped[int] = mapped_column(Integer, nullable=False)
    cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    features_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    pseudo_label: Mapped[int] = mapped_column(Integer, nullable=True)
    uncertainty_score: Mapped[float] = mapped_column(Float, nullable=True)
    human_label: Mapped[int] = mapped_column(Integer, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    status: Mapped[LabelStatus] = mapped_column(
        Enum(LabelStatus), default=LabelStatus.pending
    )

    experiment: Mapped["Experiment"] = relationship(
        back_populates="label_queue"
    )