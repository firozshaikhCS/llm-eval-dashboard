from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class EvalRunCreate(BaseModel):
    model_version: str = Field(..., example="gpt-4o-mini-v1.2")
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    sample_count: int = Field(..., gt=0)
    notes: Optional[str] = None


class EvalRunResponse(BaseModel):
    run_id: str
    model_version: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    sample_count: int
    deployment_gate_passed: bool
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class DriftAlert(BaseModel):
    detected: bool
    previous_score: float
    current_score: float
    drop: float
    message: str


class EvalSummary(BaseModel):
    latest_run_id: str
    model_version: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    deployment_gate_passed: bool
    drift_alert: Optional[DriftAlert]
    evaluated_at: datetime
