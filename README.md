---
title: Operational Risk Triage Environment
emoji: 🚦
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Operational Risk Triage Environment

This repository hosts a deterministic OpenEnv environment for operational decision-making under uncertainty and distribution shift. The agent processes a queue of cases and must choose exactly one action per case:

- `accept`
- `reject`
- `review`

The workflow is grounded in a fraud-like approval pipeline, with moderation-style ambiguity and anomaly-style OOD behavior introduced through the `easy`, `medium`, and `hard` tasks.

## Current Stage

The core environment, deterministic task bank, and deterministic grader are now in place:

- typed decision and observation models
- deterministic multi-case queue handling
- explicit task selection via `reset(task="easy" | "medium" | "hard")`
- fixed review budgets per task
- task-bank calibration against naive shortcut policies
- deterministic raw business-value grading with normalized episode scores
- baseline `inference.py` entrypoint using the OpenAI client

## Quick Start

### Run the server locally

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Interact with the environment

```python
from my_env import RiskTriageEnv, TriageAction

client = RiskTriageEnv(base_url="http://localhost:8000").sync()

with client:
    result = client.reset(task="easy")
    print(result.observation.current_case.case_id)
    print(result.observation.current_case.evidence_text)

    result = client.step(
        TriageAction(
            decision="review",
            rationale="Signals conflict and review budget is still available.",
            confidence=0.62,
        )
    )
    print(result.reward, result.done)
```

### Run the baseline entrypoint

```bash
API_BASE_URL=https://your-model-endpoint/v1 \
MODEL_NAME=gpt-4.1-mini \
HF_TOKEN=your_token \
OPENAI_API_KEY=your_api_key \
.venv/bin/python inference.py --env-url http://localhost:8000 --task easy
```

## Environment Contract

### Action

`TriageAction`

- `decision`: one of `accept`, `reject`, `review`
- `rationale`: short explanation of the action
- `confidence`: optional score in `[0.0, 1.0]`

### Observation

`TriageObservation` exposes:

- the current visible case record
- queue progress and remaining review budget
- cumulative raw reward, normalized score, and action counts
- deterministic feedback for the previous action

The visible case includes:

- contextual identifiers: `case_id`, `task_name`, `domain_hint`, `event_type`
- structured signals: `impact_score`, `risk_score`, `anomaly_score`, `history_risk_score`
- model outputs: `model_recommendation`, `model_confidence`
- uncertainty and novelty: `uncertainty_score`, `novelty_score`
- evidence quality: `feature_completeness`, `policy_flags`, `missing_fields`
- analyst-style summary: `evidence_text`

### Tasks

- `easy`: 20 cases, review budget 4
- `medium`: 25 cases, review budget 4
- `hard`: 30 cases, review budget 3

All tasks share the same API and differ only in ambiguity, missing evidence, and OOD behavior.

## Scoring

The environment uses two related score views:

- `reward` and `cumulative_reward` are raw business-value signals used during the episode
- `normalized_score` is the public evaluation score clipped into `[0.0, 1.0]`

Normalization is deterministic:

```text
normalized_score = clip(raw_score / raw_score_optimal, 0.0, 1.0)
```

Where:

- `raw_score` is the total business value earned by the agent
- `raw_score_optimal` is the deterministic hidden-policy optimum for that task

This means:

- positive but suboptimal play earns partial credit
- net-negative business value earns `0.0`
- the best possible task episode earns `1.0`

Aggregate evaluation across `easy`, `medium`, and `hard` is the arithmetic mean of the three normalized task scores.

## Project Structure

```text
my_env/
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── PROGRESS.md
├── pyproject.toml
├── README.md
├── tests/
│   └── test_grader.py
└── server/
    ├── app.py
    ├── grader.py
    ├── my_env_environment.py
    ├── task_bank.py
    └── Dockerfile
```

## Docker

Build the environment image with:

```bash
docker build -t operational-risk-triage:latest -f server/Dockerfile .
```

## Deployment

This repo follows the standard OpenEnv layout and can be deployed to Hugging Face Spaces with:

```bash
openenv push
```
