"""
Test suite for LLM Eval Dashboard
Tests cover: health, auth, eval creation, drift detection, deployment gate, history endpoints.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": "Bearer dev-key"}

VALID_RUN = {
    "model_version": "gpt-4o-mini-v1.0",
    "faithfulness": 0.91,
    "answer_relevance": 0.87,
    "context_precision": 0.85,
    "sample_count": 10,
    "notes": "Test eval run"
}

LOW_RUN = {
    "model_version": "gpt-4o-mini-v1.0",
    "faithfulness": 0.73,
    "answer_relevance": 0.79,
    "context_precision": 0.71,
    "sample_count": 10,
    "notes": "Drift run"
}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self):
        assert client.get("/health").json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_post_without_auth_returns_401(self):
        assert client.post("/evals/", json=VALID_RUN).status_code == 401

    def test_get_without_auth_returns_401(self):
        assert client.get("/evals/").status_code == 401

    def test_wrong_key_returns_401(self):
        response = client.post(
            "/evals/", json=VALID_RUN,
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Eval run creation
# ---------------------------------------------------------------------------

class TestEvalCreation:
    @patch("app.main.get_db")
    def test_valid_run_returns_201(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        mock_run = MagicMock()
        mock_run.run_id = "test-run-id"
        mock_run.model_version = "gpt-4o-mini-v1.0"
        mock_run.faithfulness = 0.91
        mock_run.answer_relevance = 0.87
        mock_run.context_precision = 0.85
        mock_run.sample_count = 10
        mock_run.deployment_gate_passed = True
        mock_run.notes = "Test eval run"
        mock_run.created_at = datetime.utcnow()
        mock_session.refresh = MagicMock()

        response = client.post("/evals/", json=VALID_RUN, headers=AUTH)
        assert response.status_code in (200, 201)

    def test_faithfulness_above_1_returns_422(self):
        bad_run = {**VALID_RUN, "faithfulness": 1.5}
        assert client.post("/evals/", json=bad_run, headers=AUTH).status_code == 422

    def test_faithfulness_below_0_returns_422(self):
        bad_run = {**VALID_RUN, "faithfulness": -0.1}
        assert client.post("/evals/", json=bad_run, headers=AUTH).status_code == 422

    def test_zero_sample_count_returns_422(self):
        bad_run = {**VALID_RUN, "sample_count": 0}
        assert client.post("/evals/", json=bad_run, headers=AUTH).status_code == 422

    def test_missing_model_version_returns_422(self):
        bad_run = {k: v for k, v in VALID_RUN.items() if k != "model_version"}
        assert client.post("/evals/", json=bad_run, headers=AUTH).status_code == 422


# ---------------------------------------------------------------------------
# Deployment gate logic
# ---------------------------------------------------------------------------

class TestDeploymentGate:
    def test_high_faithfulness_passes_gate(self):
        """Faithfulness >= 0.75 should pass deployment gate"""
        from app.main import DEPLOYMENT_GATE
        assert VALID_RUN["faithfulness"] >= DEPLOYMENT_GATE

    def test_low_faithfulness_fails_gate(self):
        """Faithfulness = 0.73 should fail deployment gate (threshold is 0.75)"""
        from app.main import DEPLOYMENT_GATE
        assert LOW_RUN["faithfulness"] < DEPLOYMENT_GATE

    def test_gate_threshold_is_sensible(self):
        """Gate should be between 0.5 and 0.95"""
        from app.main import DEPLOYMENT_GATE
        assert 0.5 <= DEPLOYMENT_GATE <= 0.95


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_drift_threshold_is_sensible(self):
        from app.main import DRIFT_THRESHOLD
        assert 0.05 <= DRIFT_THRESHOLD <= 0.30

    def test_actual_drift_story_drop_exceeds_threshold(self):
        """The 0.91 → 0.73 drop is 0.18 — must exceed DRIFT_THRESHOLD"""
        from app.main import DRIFT_THRESHOLD
        drop = 0.91 - 0.73
        assert drop >= DRIFT_THRESHOLD, (
            f"The documented drift drop ({drop}) should exceed threshold ({DRIFT_THRESHOLD})"
        )

    @patch("app.main.get_db")
    def test_summary_endpoint_with_no_runs_returns_404(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = []

        response = client.get("/evals/summary/latest", headers=AUTH)
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

class TestHistory:
    @patch("app.main.get_db")
    def test_drift_history_returns_list(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.order_by.return_value.all.return_value = []

        response = client.get("/evals/history/drift", headers=AUTH)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @patch("app.main.get_db")
    def test_list_evals_returns_200(self, mock_db):
        mock_session = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.order_by.return_value.filter.return_value.limit.return_value.all.return_value = []
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = []

        response = client.get("/evals/", headers=AUTH)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Eval runner unit tests
# ---------------------------------------------------------------------------

class TestEvalRunner:
    def test_faithfulness_calculation_perfect_match(self):
        from evals.run_eval import calculate_faithfulness
        answer = "The DPDP Act requires consent before processing data."
        context = "The DPDP Act requires organisations to obtain consent before processing personal data."
        score = calculate_faithfulness(answer, context)
        assert 0.5 <= score <= 1.0

    def test_faithfulness_calculation_no_match(self):
        from evals.run_eval import calculate_faithfulness
        score = calculate_faithfulness("The sky is blue.", "Water is wet.")
        assert score >= 0.0

    def test_answer_relevance_high_overlap(self):
        from evals.run_eval import calculate_answer_relevance
        q = "What are the penalties for DPDP violations?"
        a = "Penalties for DPDP violations can reach Rs 250 crore per violation."
        score = calculate_answer_relevance(a, q)
        assert score > 0.3

    def test_golden_dataset_has_10_samples(self):
        from evals.run_eval import GOLDEN_DATASET
        assert len(GOLDEN_DATASET) == 10

    def test_golden_dataset_has_required_fields(self):
        from evals.run_eval import GOLDEN_DATASET
        for sample in GOLDEN_DATASET:
            assert "question" in sample
            assert "ground_truth" in sample
            assert "context" in sample
