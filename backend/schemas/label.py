from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from backend.models.label_queue import LabelStatus


class LabelQueueItemResponse(BaseModel):
    id: str
    experiment_id: str
    engine_id: int
    cycle: int
    features_json: dict
    pseudo_label: Optional[int]
    uncertainty_score: Optional[float]
    human_label: Optional[int]
    reviewed_at: Optional[datetime]
    status: LabelStatus

    model_config = {"from_attributes": True}


class HumanLabelSubmit(BaseModel):
    label_id: str
    human_label: int = Field(..., ge=0, le=1)


class LabelQueueResponse(BaseModel):
    items: list[LabelQueueItemResponse]
    total_pending: int
    total_reviewed: int