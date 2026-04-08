---
title: Operational Risk Triage Environment
emoji: 🚦
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - fastapi
  - risk-triage
---

# Operational Risk Triage Environment
*A production-style decision environment for fraud, trust, and operational risk triage under uncertainty.*

This project models a real queue management problem used in fraud operations, payments risk, trust and safety, and anomaly investigation systems.

At each step, an agent sees a structured case and must choose exactly one action:

- `accept`
- `reject`
- `review`

The environment is deterministic, OpenEnv-compatible, and designed around the actual business tradeoffs that make operational triage difficult:

- false accepts are expensive
- false rejects still carry meaningful cost
- review capacity is limited
- model recommendations are informative, but not always safe to follow blindly

This is not a toy classification benchmark. It is a workflow benchmark for high-stakes decision systems.

## 🚨 Why This Problem Matters

Real risk systems do not stop at a score.

In production, a platform still needs to decide:

- should this event be approved immediately?
- should it be blocked immediately?
- should it consume scarce human analyst capacity?

That is the problem this environment simulates.

Examples of where this workflow shows up:

- payments fraud review
- merchant risk operations
- trust and safety escalation
- account integrity investigation
- anomaly triage for suspicious platform activity

What makes this hard in practice is not just prediction. It is operational decision-making under uncertainty, with asymmetric costs and constrained review bandwidth.

## 🧠 Environment Overview

**Environment name:** `operational_risk_triage`

**Tasks:**

- `easy`: 20 cases, review budget 4
- `medium`: 25 cases, review budget 4
- `hard`: 30 cases, review budget 3

All three tasks share the same schema and action space. Difficulty increases through:

- noisier or more conflicting signals
- less complete evidence
- weaker alignment between upstream recommendation and hidden optimal action
- tighter review-budget pressure

## 📦 Observation Space

Each step exposes the current case plus queue-level control state.

Core case signals include:

- `risk_score`
- `anomaly_score`
- `history_risk_score`
- `uncertainty_score`
- `novelty_score`
- `feature_completeness`
- `model_recommendation`
- `model_confidence`
- `policy_flags`
- `missing_fields`
- `evidence_text`

Queue and episode context includes:

- `queue_position`
- `remaining_cases`
- `remaining_review_budget`
- `cumulative_reward`
- `normalized_score`
- action counts
- deterministic feedback from the previous decision

These features are intentionally chosen to resemble a real decision console: model outputs are present, but the final action still depends on multiple signals, evidence quality, and operating constraints.

## 🎯 Action Space

The action space is fixed and operationally meaningful:

- `accept`: allow the case through immediately
- `reject`: block the case immediately
- `review`: escalate to bounded human review

`review` is useful, but costly. The budget is task-level and limited by design.

## 💰 Reward Design

The reward function is step-level and business-aligned.

The intuition is simple:

- correct accepts and correct rejects are rewarded
- false accepts are penalized the most
- false rejects are penalized materially
- unnecessary review is penalized because analyst capacity is expensive

Final public evaluation uses a deterministic normalized score derived from the accumulated raw business reward.

This means the environment measures decision quality, not just label agreement.

## ⚙️ Approach

The submitted agent is a **deterministic heuristic policy** implemented in `inference.py`.

This policy is:

- signal-driven
- interpretable
- deterministic
- cheap to run
- independent of any required LLM dependency for the reported results

Most importantly, the policy is **not keyed on case IDs** and does **not memorize a fixed answer list**. It makes decisions from the observable risk signals and policy flags available in the environment.

That distinction matters:

- a hardcoded policy would overfit exact case identities
- this policy applies shared rules over signal combinations
- the same logic runs across `easy`, `medium`, and `hard`

The baseline behaves like a production decision layer sitting on top of upstream scoring systems.

## 🧱 Key Design Principles

- **Conservative acceptance**  
  Accept is reserved for genuinely clean cases: low history risk, low anomaly, good evidence quality, and no severe escalation flags.

- **Aggressive rejection on strong signals**  
  Severe policy flags, high anomaly, and compounding risk signals push the policy toward decisive rejection.

- **Efficient review usage**  
  Review is treated as scarce analyst capacity, not as a generic fallback. The policy spends review budget only where ambiguity is real and escalation is worth the cost.

- **Uncertainty handling**  
  Uncertainty matters in context. High uncertainty with weak evidence quality can justify review; uncertainty alone does not automatically override stronger signals.

- **Signal-over-recommendation discipline**  
  `model_recommendation` is used as an input, not as the final authority. This is especially important on harder cases where the recommendation is intentionally imperfect.

## 📈 Results

Current deterministic heuristic scores:

| Task | Normalized Score |
|---|---:|
| Easy | 1.00 |
| Medium | 1.00 |
| Hard | 0.94 |

**Mean normalized score:** `0.98`

Why performance is strong:

- the policy approves clean traffic quickly
- it rejects high-risk clusters decisively
- it protects against the most expensive operational error: bad accepts
- it uses review budget efficiently instead of wasting it on low-value escalation
- it improves `hard` substantially without degrading `easy` or `medium`

This is near-optimal performance from a deterministic, interpretable ruleset, which is exactly the kind of baseline an operations team could inspect and trust.

## 🏗️ Implementation

This environment is built with:

- **FastAPI**
- **OpenEnv**
- **typed Pydantic schemas**
- **deterministic task generation**
- **deterministic grading**
- **Docker-based Hugging Face deployment**

Key files:

- `server/app.py`: FastAPI + OpenEnv server
- `server/my_env_environment.py`: environment dynamics
- `server/task_bank.py`: deterministic task definitions
- `server/grader.py`: dense business-aligned grading
- `models.py`: typed action / observation / state models
- `inference.py`: deterministic baseline inference policy

## 🛠️ Setup

### Local server

Run the environment locally:

```bash
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Baseline inference

Run the deterministic baseline against the local server:

```bash
.venv/bin/python inference.py --env-url http://localhost:8000 --task easy
```

Change `--task` to `easy`, `medium`, or `hard`.

### Docker

Build and run locally with Docker:

```bash
docker build -t operational-risk-triage:latest .
docker run --rm -p 7860:7860 operational-risk-triage:latest
```

Then call the environment at:

```text
http://localhost:7860
```

## 🌐 Deployment

This project is deployed as a **Docker-based Hugging Face Space**.

Deployment details:

- runtime: Docker
- app port: `7860`
- FastAPI app: `server.app:app`

Live deployment:

```text
https://sehtaj-openenv-triage.hf.space
```

Important endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /metadata`
- `GET /schema`

The validator-critical endpoint is:

```text
POST /reset
```

## ✅ OpenEnv Compliance

This submission implements the expected OpenEnv interaction model:

- `reset()`
- `step(action)`
- `state()`

It also preserves the required structured inference logs:

```text
[START] task=<task_name> env=<env_name> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

The environment is deterministic end to end:

- deterministic tasks
- deterministic grader
- deterministic policy baseline
- reproducible scores across runs

## 🔍 Why This Submission Is Strong

This environment is strong because it captures the part of decision systems that often gets ignored:

- not just prediction
- not just classification
- but queue-level operational action selection under uncertainty and cost

It rewards disciplined triage behavior:

- approve low-risk cases fast
- reject dangerous cases early
- escalate only when ambiguity justifies analyst cost

That makes it a better proxy for real risk operations than a simple score-threshold benchmark.

## 🔭 Future Work

There is clear room to extend this system beyond a deterministic rules baseline:

- RL-based policy optimization over the same environment
- adaptive thresholds learned from offline data
- stronger hidden evaluation sets for harder anti-overfitting pressure
- calibrated review-allocation strategies
- optional hybrid policy with LLM-assisted reasoning on selected ambiguous cases

The current system is intentionally deterministic and interpretable. Future iterations can push toward more adaptive policies without changing the environment contract.

## 📌 Summary

Operational Risk Triage is a deterministic OpenEnv environment for high-stakes accept / reject / review decision-making.

It is built around a real operational workflow, not a toy task. The environment captures the tradeoff between fraud prevention, false-positive cost, and scarce analyst review capacity.

The current deterministic heuristic baseline achieves:

- `easy = 1.00`
- `medium = 1.00`
- `hard = 0.94`

with a signal-driven, interpretable policy and a fully deployable FastAPI + OpenEnv system on Hugging Face Spaces.
