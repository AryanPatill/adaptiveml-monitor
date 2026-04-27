import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Integer, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database import Base
import enum


class ExperimentStatus(str, enum.Enum):
    running = "running"
    complete = "complete"
    failed = "failed"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    label_delay_days: Mapped[int] = mapped_column(Integer, default=30)
    dataset_subset: Mapped[str] = mapped_column(String(10), default="FD001")
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus), default=ExperimentStatus.running
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    label_queue: Mapped[list["LabelQueue"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )