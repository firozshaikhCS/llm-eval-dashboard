"""
Seeds the database with the 6 eval runs that tell the judge drift story.
Run this once after setting up the dashboard to populate Grafana with real data.

    python evals/seed_drift_story.py

This recreates the exact sequence that was observed in production:
- Runs 1-4: Stable scores around 0.88-0.91 faithfulness
- Run 5:    Sudden drop to 0.73 — judge model silently updated by provider
- Run 6:    Recovery to 0.89 after pinning judge model version

The full postmortem is in DECISIONS.md under ADR-003.
"""

import requests
import os
import time

DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

DRIFT_STORY = [
    {
        "model_version": "gpt-4o-mini-v1.0",
        "faithfulness": 0.88,
        "answer_relevance": 0.84,
        "context_precision": 0.82,
        "sample_count": 10,
        "notes": "Baseline eval — initial deployment. All metrics healthy."
    },
    {
        "model_version": "gpt-4o-mini-v1.0",
        "faithfulness": 0.91,
        "answer_relevance": 0.87,
        "context_precision": 0.85,
        "sample_count": 10,
        "notes": "Week 2 eval — improved chunking strategy. Peak faithfulness."
    },
    {
        "model_version": "gpt-4o-mini-v1.0",
        "faithfulness": 0.90,
        "answer_relevance": 0.86,
        "context_precision": 0.84,
        "sample_count": 10,
        "notes": "Week 3 eval — stable. No code changes."
    },
    {
        "model_version": "gpt-4o-mini-v1.0",
        "faithfulness": 0.89,
        "answer_relevance": 0.85,
        "context_precision": 0.83,
        "sample_count": 10,
        "notes": "Week 4 eval — minor variance, within expected range."
    },
    {
        "model_version": "gpt-4o-mini-v1.0",
        "faithfulness": 0.73,
        "answer_relevance": 0.79,
        "context_precision": 0.71,
        "sample_count": 10,
        "notes": (
            "DRIFT DETECTED — deployment blocked. "
            "Faithfulness dropped 0.18 points (0.91 → 0.73). "
            "No code changes made. Investigating judge model version."
        )
    },
    {
        "model_version": "gpt-4o-mini-v1.0-judge-pinned",
        "faithfulness": 0.89,
        "answer_relevance": 0.86,
        "context_precision": 0.84,
        "sample_count": 10,
        "notes": (
            "RESOLVED — judge model version pinned to gpt-4o-mini-2024-07-18. "
            "Provider had silently updated the default model. "
            "Scores recovered to pre-drift levels. See DECISIONS.md ADR-003."
        )
    }
]


def seed():
    print("Seeding eval dashboard with drift story data...")
    for i, run in enumerate(DRIFT_STORY):
        resp = requests.post(
            f"{DASHBOARD_URL}/evals/",
            json=run,
            headers=HEADERS,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        gate = "PASS ✅" if data["deployment_gate_passed"] else "BLOCK 🚫"
        print(
            f"  Run {i+1}: faithfulness={run['faithfulness']:.2f} "
            f"gate={gate} — {run['notes'][:60]}"
        )
        time.sleep(0.1)
    print("\nDone. Open Grafana to see the drift spike on run 5.")
    print("The 0.91 → 0.73 drop and recovery is now visible in the time-series panel.")


if __name__ == "__main__":
    seed()
