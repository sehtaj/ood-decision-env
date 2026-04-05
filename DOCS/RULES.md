> This file has higher priority than all other documentation.
> Any agent violating this file is considered incorrect.

# 1. CORE OBJECTIVE

- This repository MUST implement a real-world OpenEnv environment.
- The environment MUST NOT be a toy, demo, game, puzzle, or synthetic mini-benchmark.
- The environment MUST simulate a real human decision workflow.
- The environment MUST model high-stakes operational decision-making under uncertainty.

# 2. OPENENV SPEC COMPLIANCE

- The environment MUST use typed Pydantic models for `Action`, `Observation`, and `State`.
- The environment MUST implement `step(action)` and it MUST return `observation`, `reward`, `done`, and `info`.
- The environment MUST implement `reset()` and it MUST return the initial observation.
- The environment MUST implement `state()` and it MUST return the current state.
- The repository MUST include a valid `openenv.yaml` file.
- The repository MUST pass `openenv validate`.

# 3. REAL-WORLD TASK REQUIREMENT

- The environment MUST NOT be a game or toy.
- The environment MUST represent a real operational workflow.
- Allowed workflow types include fraud detection, moderation, triage, and anomaly detection.

# 4. TASK REQUIREMENTS

- The environment MUST provide a minimum of 3 tasks.
- The tasks MUST be named `easy`, `medium`, and `hard`.
- All tasks MUST use the same action schema.
- All tasks MUST use the same observation schema.
- All tasks MUST be deterministic.
- Randomness is STRICTLY FORBIDDEN.
- Task difficulty MUST increase from `easy` to `medium` to `hard`.

# 5. GRADER REQUIREMENTS

- Grading MUST be deterministic ONLY.
- Every score MUST be in the range `[0.0, 1.0]`.
- Scores MUST be reproducible across runs.
- The grader MUST NOT return constant scores.
- The grader MUST reflect real task performance.

# 6. REWARD FUNCTION RULES

- Rewards MUST be dense.
- Rewards MUST be provided at the step level.
- Rewards MUST reflect real business cost.
- Rewards MUST penalize false accepts.
- Rewards MUST penalize false rejects.
- Rewards MUST penalize unnecessary review.
- Rewards MUST reward correct decisions.

# 7. INFERENCE SCRIPT REQUIREMENTS (CRITICAL)

## 7.1 File Rules

- The inference script MUST be named `inference.py`.
- `inference.py` MUST be located in the repository root.

## 7.2 Environment Variables

- `inference.py` MUST use `API_BASE_URL`.
- `inference.py` MUST use `MODEL_NAME`.
- `inference.py` MUST use `HF_TOKEN`.

## 7.3 OpenAI Client Requirement

- `inference.py` MUST use `from openai import OpenAI`.

## 7.4 STDOUT LOG FORMAT (STRICT — NON-NEGOTIABLE)

- The script MUST output EXACTLY these 3 line types:

```text
[START] task=<task_name> env=<env_name> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

## 7.5 Logging Rules

- NO extra logs are allowed.
- Debug prints are STRICTLY FORBIDDEN.
- Multiline output is STRICTLY FORBIDDEN.
- Exact formatting is NON-NEGOTIABLE.
- `reward` MUST be formatted to 2 decimal places.
- `done` and `success` MUST use lowercase `true` or `false`.
- `error` MUST be `null` or a string.

# 8. DOCKER REQUIREMENTS

- A `Dockerfile` MUST exist.
- The repository MUST pass `docker build`.
- The container MUST run the environment server automatically.

# 9. HUGGING FACE DEPLOYMENT

- The environment MUST deploy to Hugging Face Spaces.
- The deployment MUST respond to `POST /reset`.
- `POST /reset` MUST return HTTP `200`.

# 10. VALIDATION SCRIPT REQUIREMENTS

- The repository MUST pass `validate-submission.sh`.
- The validator requirements are NON-NEGOTIABLE:
  1. HF Space responds to `/reset` with HTTP `200`.
  2. Docker build succeeds.
  3. `openenv validate` passes.
- ALL THREE are REQUIRED.

# 11. RUNTIME CONSTRAINTS

- The full system MUST run under `2 vCPU`.
- The full system MUST run under `8GB RAM`.
- Inference MUST complete in under `20 minutes`.

# 12. DISQUALIFICATION CONDITIONS

- Immediate failure if the HF Space is not responding.
- Immediate failure if Docker build fails.
- Immediate failure if `openenv validate` fails.
- Immediate failure if `inference.py` is missing.
- Immediate failure if logs do not match the required format.
- Immediate failure if there are fewer than 3 tasks.
- Immediate failure if grading is non-deterministic.
- Immediate failure if any score is outside `[0,1]`.

# BASELINE REPRODUCIBILITY REQUIREMENT

- The inference script MUST produce reproducible scores across runs.
- The same task MUST produce the same score given identical inputs.
- Any randomness in inference MUST be controlled or eliminated.

# DOCUMENTATION REQUIREMENTS

- The repository MUST include a README.md.
- README MUST describe:
  - environment purpose
  - action space
  - observation space
  - task definitions (easy, medium, hard)
  - setup and run instructions

# EPISODE DESIGN REQUIREMENTS

- reset() MUST create a clean initial state.
- Each episode MUST have a well-defined start and end.
- done MUST only be true when the episode is complete.
- State transitions MUST be consistent and deterministic.

# ANTI-SHORTCUT REQUIREMENTS

- The environment MUST NOT allow trivial policies to succeed.
- Always-accept MUST perform poorly on at least one task.
- Always-reject MUST perform poorly on at least one task.
- Always-review MUST be penalized due to cost or budget.

# CREATIVITY REQUIREMENT

- The environment SHOULD introduce non-trivial decision complexity.
- The design SHOULD include realistic ambiguity or uncertainty.
- The task SHOULD not be a trivial threshold-based problem.

# EXECUTION VALIDATION REQUIREMENT

- inference.py MUST successfully complete an episode end-to-end.
- All tasks (easy, medium, hard) MUST be runnable programmatically.

# 13. AGENT EXECUTION CONSTRAINTS (VERY IMPORTANT)

- Any agent modifying this repository MUST preserve OpenEnv compliance.
- Any agent modifying this repository MUST preserve task structure.
- Any agent modifying this repository MUST preserve deterministic behavior.
- Any agent modifying this repository MUST ensure the validator passes.
- Any agent modifying this repository MUST NOT change the action space `accept`, `reject`, `review`.
- Any agent modifying this repository MUST NOT break determinism.
- Any agent modifying this repository MUST NOT modify the required logging format.
- Any agent modifying this repository MUST NOT introduce randomness.
- Any agent modifying this repository MUST NOT bypass grader logic.
- Any agent modifying this repository MUST NOT add external dependencies that require runtime downloads.
