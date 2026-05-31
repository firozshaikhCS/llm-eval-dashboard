# Architecture Decision Records

> This document explains **why** the system is built the way it is — not just what it does.
> ADR-003 contains the postmortem on the 0.91 → 0.73 judge drift incident.

---

## ADR-001 — Why RAGAS instead of DeepEval or custom metrics

**Decision:** RAGAS is the evaluation framework used for faithfulness, answer relevance, and context precision scoring.

**The problem:**
LLM outputs need to be measured against reference answers and source contexts — not just checked for "does it sound right." Manual review doesn't scale beyond 50 samples. A custom metric is code you have to maintain.

**Why RAGAS:**
- Three metrics map directly to real failure modes: faithfulness (hallucination), answer relevance (off-topic responses), context precision (retrieval quality)
- The golden dataset format (question + ground truth + context) forces you to specify what "correct" looks like — this is the discipline that matters
- Widely adopted enough that hiring managers recognise it as a production signal

**Why not DeepEval:**
DeepEval is stronger for unit-testing individual prompts. RAGAS is stronger for measuring a full RAG pipeline end-to-end. This system evaluates pipelines, not individual prompts.

**Why not custom metrics:**
Custom metrics encode assumptions that drift over time without you noticing. Using an established framework means the metric itself is versioned and documented by someone else.

---

## ADR-002 — Why LangSmith for tracing instead of MLflow or local logging

**Decision:** LangSmith captures the full LLM trace (input tokens, output, latency, cost) for every eval run.

**The problem this solves:**
Without tracing, debugging a faithfulness drop means replaying queries manually and guessing. With LangSmith, you can diff the exact prompt, retrieved context, and model output between a passing run and a failing run — in 2 minutes.

**Why LangSmith specifically:**
- Native LangChain integration — zero extra code if you're already using LangChain
- Run comparison view shows diffs between any two traces side by side
- Feedback logging (thumbs up/down) maps to RAGAS scores without duplication

**Why not MLflow:**
MLflow is excellent for traditional ML experiment tracking. For LLM traces, it doesn't natively handle the prompt/response/context structure that makes debugging useful.

**Why not local logging:**
Logs tell you what happened. Traces tell you why. You can't diff two log files to understand why faithfulness dropped.

---

## ADR-003 — The judge drift incident postmortem (0.91 → 0.73)

**What happened:**
On week 5, the automated eval run flagged a faithfulness score of 0.73 — an 0.18-point drop from the previous week's 0.91. The deployment gate blocked the release. No code had changed in the RAG pipeline. No changes had been made to the golden dataset.

**Initial hypotheses:**
1. Data distribution shift — new query types in the eval set
2. Retrieval quality degradation — chunking or embedding issue
3. Judge model behaviour change — the LLM evaluating faithfulness had changed

**Investigation:**
- Checked eval dataset: identical 10 samples, no changes
- Checked retrieval: reran the pipeline manually, retrieved context was correct
- Checked LangSmith traces: the judge model (used by RAGAS to score faithfulness) was returning different scores for identical input/output pairs vs the previous week
- Checked OpenAI changelog: the default model alias (`gpt-4o-mini`) had been silently updated to a new version by the provider

**Root cause:**
RAGAS was calling the judge model using the alias `gpt-4o-mini` without pinning to a specific version. OpenAI updated the model behind the alias. The new model version scored faithfulness more conservatively on factual assertions, causing scores to drop system-wide without any change to the RAG pipeline or golden dataset.

**The fix:**
Pinned the judge model to `gpt-4o-mini-2024-07-18` (explicit version string, not alias). Added a weekly calibration check: run 5 known-good samples through the judge and verify scores are within 0.05 of the expected baseline. If they're not, alert before the full eval runs.

**What the fix proved:**
The deployment gate worked exactly as designed. A silent external change would have degraded production quality without detection if the gate didn't exist. The 0.73 score was real — the pipeline genuinely performed worse with the new judge. The fix wasn't to raise the threshold; it was to eliminate the source of variance.

**What this means for other systems:**
Never use model aliases in automated eval pipelines. Pin the exact version. Treat judge model updates as a breaking change requiring re-calibration.

---

## ADR-004 — Why a separate FastAPI service instead of a script

**Decision:** The eval results are exposed via a REST API, not stored in a CSV or logged to a file.

**The problem with file-based eval storage:**
A CSV of eval scores is only useful to the person who knows where it is. A REST API is accessible to Grafana, to CI pipelines, to downstream agents, and to any team member with an HTTP client — without file system access.

**What the API enables:**
- Grafana polls `/evals/summary/latest` every 30 seconds — live dashboard without any manual refresh
- CI/CD pipeline calls `/evals/` after every deployment candidate — gate is automated, not manual
- The drift alert response (`/evals/summary/latest`) includes the exact delta and a human-readable message that can be posted to Slack or email without additional formatting code

**Why not a database + Grafana direct query:**
Grafana can query PostgreSQL directly. But the FastAPI layer adds the deployment gate logic, the drift calculation, and the summary endpoint — none of which belong in a SQL query or a Grafana transform. The API is where business logic lives.

---

## ADR-005 — Why 10 golden samples instead of 100

**Decision:** The golden dataset contains 10 carefully curated Q&A pairs, not a large auto-generated set.

**The quality-vs-quantity tradeoff:**
A 100-sample dataset generated by an LLM is measuring whether your RAG system agrees with another LLM. A 10-sample dataset curated from real user queries is measuring whether your system answers real questions correctly.

**Why 10 is enough for early-stage monitoring:**
- Enough to detect a 0.15+ point drift reliably (as proven by the ADR-003 incident)
- Fast enough to run in CI without adding significant build time
- Small enough that every sample can be manually reviewed for quality

**When to expand:**
- When the system handles more than 3 distinct query categories
- When you need to detect smaller drifts (< 0.10 point changes)
- When you have enough real user queries to curate from (minimum 500 logged queries before sampling)

**What not to do:**
Do not generate golden samples using the same LLM that powers your RAG system. The model will answer questions it generated correctly by construction, telling you nothing about real-world performance.

---

## Failures encountered and resolved

| Failure | What happened | Fix |
|---|---|---|
| Judge model drift (ADR-003) | Faithfulness dropped 0.18 points silently | Pinned judge model to explicit version |
| Eval blocking deployment silently | Gate failed but CI passed because no one checked the score | Added explicit exit code 1 from eval runner when gate fails |
| Golden dataset answered by training data | Model answered correctly without using context | Added adversarial samples where context contradicts common knowledge |
| RAGAS timeout on large context | Long documents caused judge to exceed timeout | Added chunk-level scoring with aggregation instead of full-document scoring |
| Grafana lost connection on API restart | Grafana data source marked as down after 30s | Added retry logic to Grafana provisioning config |
