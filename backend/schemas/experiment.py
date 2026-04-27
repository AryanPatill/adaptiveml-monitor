from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from backend.models.experiment import ExperimentStatus


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    label_delay_days: int = Field(default=30, ge=1, le=365)
    dataset_subset: str = Field(default="FD001")


class ExperimentResponse(BaseModel):
    id: str
    name: str
    label_delay_days: int
    dataset_subset: str
    status: ExperimentStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentResponse]
    total: int