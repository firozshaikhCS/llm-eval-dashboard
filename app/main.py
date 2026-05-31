"""
LLM Evaluation Dashboard — FastAPI backend
Tracks faithfulness, answer relevance, and context precision scores across eval runs.
Exposes endpoints for the Grafana dashboard and for triggering new eval runs.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import get_db, engine
from app.models import Base, EvalRun
from app.schemas import EvalRunCreate, EvalRunResponse, EvalSummary, DriftAlert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="LLM Eval Dashboard",
    description="Tracks LLM quality metrics across eval runs. Detects drift before it hits production.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "dev-key")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.10"))
DEPLOYMENT_GATE = float(os.getenv("DEPLOYMENT_GATE", "0.75"))


def verify_api_key(authorization: str = Header(...)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return authorization


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "llm-eval-dashboard"}


# ---------------------------------------------------------------------------
# Eval runs — write (triggered by CI or manually)
# ---------------------------------------------------------------------------

@app.post("/evals/", response_model=EvalRunResponse, status_code=201,
          dependencies=[Depends(verify_api_key)])
def create_eval_run(payload: EvalRunCreate, db: Session = Depends(get_db)):
    """
    Record the result of one eval run.
    Called by the RAGAS runner after scoring a golden dataset.
    Automatically flags drift and blocks deployment if scores fall below gate.
    """
    run = EvalRun(
        run_id=str(uuid.uuid4()),
        model_version=payload.model_version,
        faithfulness=payload.faithfulness,
        answer_relevance=payload.answer_relevance,
        context_precision=payload.context_precision,
        sample_count=payload.sample_count,
        deployment_gate_passed=payload.faithfulness >= DEPLOYMENT_GATE,
        notes=payload.notes,
        created_at=datetime.utcnow()
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    logger.info(
        f"Eval run {run.run_id} — faithfulness={run.faithfulness:.2f} "
        f"gate={'PASS' if run.deployment_gate_passed else 'BLOCK'}"
    )
    return run


# ---------------------------------------------------------------------------
# Eval runs — read (Grafana + dashboard)
# ---------------------------------------------------------------------------

@app.get("/evals/", response_model=List[EvalRunResponse],
         dependencies=[Depends(verify_api_key)])
def list_eval_runs(
    limit: int = Query(default=50, le=200),
    model_version: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Return historical eval runs, newest first. Grafana polls this endpoint."""
    q = db.query(EvalRun).order_by(EvalRun.created_at.desc())
    if model_version:
        q = q.filter(EvalRun.model_version == model_version)
    return q.limit(limit).all()


@app.get("/evals/{run_id}", response_model=EvalRunResponse,
         dependencies=[Depends(verify_api_key)])
def get_eval_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(EvalRun).filter(EvalRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found")
    return run


# ---------------------------------------------------------------------------
# Summary — latest scores at a glance
# ---------------------------------------------------------------------------

@app.get("/evals/summary/latest", response_model=EvalSummary,
         dependencies=[Depends(verify_api_key)])
def get_latest_summary(db: Session = Depends(get_db)):
    """
    Returns the most recent eval scores plus a drift alert if
    faithfulness has dropped more than DRIFT_THRESHOLD vs previous run.
    This is the endpoint the Grafana stat panels poll every 30s.
    """
    runs = db.query(EvalRun).order_by(EvalRun.created_at.desc()).limit(2).all()
    if not runs:
        raise HTTPException(status_code=404, detail="No eval runs recorded yet")

    latest = runs[0]
    drift_alert = None

    if len(runs) == 2:
        prev = runs[1]
        drop = prev.faithfulness - latest.faithfulness
        if drop >= DRIFT_THRESHOLD:
            drift_alert = DriftAlert(
                detected=True,
                previous_score=prev.faithfulness,
                current_score=latest.faithfulness,
                drop=round(drop, 3),
                message=(
                    f"Faithfulness dropped {drop:.2f} points "
                    f"({prev.faithfulness:.2f} → {latest.faithfulness:.2f}). "
                    f"Deployment blocked."
                )
            )

    return EvalSummary(
        latest_run_id=latest.run_id,
        model_version=latest.model_version,
        faithfulness=latest.faithfulness,
        answer_relevance=latest.answer_relevance,
        context_precision=latest.context_precision,
        deployment_gate_passed=latest.deployment_gate_passed,
        drift_alert=drift_alert,
        evaluated_at=latest.created_at
    )


# ---------------------------------------------------------------------------
# Drift history — the 0.91 → 0.73 story
# ---------------------------------------------------------------------------

@app.get("/evals/history/drift", dependencies=[Depends(verify_api_key)])
def get_drift_history(db: Session = Depends(get_db)):
    """
    Returns faithfulness scores over time for the Grafana time-series panel.
    The drop from 0.91 to 0.73 across runs 4-5 is visible here — the exact
    signal that triggered the judge model version pin fix documented in DECISIONS.md.
    """
    runs = db.query(EvalRun).order_by(EvalRun.created_at.asc()).all()
    return [
        {
            "run_id": r.run_id,
            "model_version": r.model_version,
            "faithfulness": r.faithfulness,
            "deployment_gate_passed": r.deployment_gate_passed,
            "timestamp": r.created_at.isoformat()
        }
        for r in runs
    ]
