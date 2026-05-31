from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True, nullable=False)
    model_version = Column(String, nullable=False)
    faithfulness = Column(Float, nullable=False)
    answer_relevance = Column(Float, nullable=False)
    context_precision = Column(Float, nullable=False)
    sample_count = Column(Integer, default=0)
    deployment_gate_passed = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
