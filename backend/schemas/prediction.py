from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class PredictionResponse(BaseModel):
    id: str
    model_version_id: str
    engine_id: int
    cycle: int
    predicted_label: int
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    true_label: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelMetricsResponse(BaseModel):
    model_version_id: str
    model_type: str
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    coverage: Optional[float]
    avg_interval_width: Optional[float]
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelComparisonResponse(BaseModel):
    baseline: Optional[ModelMetricsResponse]
    continual: Optional[ModelMetricsResponse]
    improvement_f1: Optional[float]
    improvement_coverage: Optional[float]