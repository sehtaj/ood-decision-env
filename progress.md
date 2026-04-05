# OOD Decision-Making Environment Progress Tracker

## Stage 0
### Stage 0 Locked Decisions
- Environment direction: a production-style operational risk triage environment for high-stakes decisions under uncertainty and distribution shift.
- Base workflow: closest to fraud detection, with moderation-style ambiguity and anomaly-style distribution shift expressed through task variants.
- Fixed action space: `accept`, `reject`, `review`.
- Episode structure: multi-case decision queue, one case per step, deterministic task selection via `reset(...)`.
- Task ladder: `easy`, `medium`, `hard`, all sharing the same action and observation interface.
- Review semantics: `review` resolves the current case immediately, consumes limited review capacity, and incurs an explicit cost so it is useful on borderline cases but suboptimal on obvious ones.
- Episode calibration target: `easy` uses 20 cases with review budget 4, `medium` uses 25 cases with review budget 4, and `hard` uses 30 cases with review budget 3.
- Cost asymmetry: false accepts are materially more costly than false rejects because they allow harmful events to pass, while false rejects primarily degrade user experience or operational efficiency.

### 0.1 Product Scope Lock
- [x] Lock the environment name and one-sentence pitch around out-of-distribution decision-making for real-world AI operations.
- [x] Lock the action space to exactly `accept`, `reject`, and `review`.
- [x] Lock the primary decision workflow to match production-style triage rather than a toy classification loop.
- [x] Lock the business objective as maximizing correct decisions while minimizing costly mistakes and unnecessary reviews.
- [x] Lock the supported domains to fraud detection, anomaly detection, and moderation under one shared decision interface.
- [x] Lock the target user of the environment as an LLM-driven decision agent with optional human-review escalation.

### 0.2 Real-World Scenario Definition

- Locked case representation: each step is one operational triage record from a fixed queue, grounded in a fraud-like approval pipeline with `payment`, `content`, or `system` domain context. Each record represents a real event with actor history, event context, automated model outputs, rule flags, evidence quality signals, and a short analyst-style evidence summary.
- Locked visible observation fields:
  - Identity and context: `case_id`, `task_name`, `domain_hint`, `event_type`
  - Structured risk signals: `impact_score`, `risk_score`, `anomaly_score`, `history_risk_score`
  - Model outputs: `model_recommendation`, `model_confidence`
  - Uncertainty and OOD hints: `uncertainty_score`, `novelty_score`
  - Evidence quality: `feature_completeness`, `policy_flags`, `missing_fields`
  - Free-text evidence: `evidence_text`
  - Queue state: `queue_position`, `remaining_cases`, `remaining_review_budget`
- Locked hidden grading fields:
  - Ground truth: `true_case_class`, `optimal_decision`
  - Shift metadata: `is_ood`, `ood_type`
  - Business costs: `cost_tier`, `false_accept_cost`, `false_reject_cost`, `review_cost`
  - Deterministic action values: `action_value_accept`, `action_value_reject`, `action_value_review`
- Locked uncertainty and ambiguity rules:
  - Low confidence appears through lower `model_confidence` and higher `uncertainty_score`
  - Misleading high confidence is allowed on OOD cases, especially in `hard`
  - Missing or noisy evidence appears through `feature_completeness` and `missing_fields`
  - Explicit OOD hints appear through `novelty_score` and certain policy flags
  - Implicit OOD appears as disagreement between scores, recommendation, flags, and `evidence_text`
- Locked task-specific uncertainty profile:
  - `easy`: mostly in-distribution payment-style cases, aligned signals, high completeness, low novelty
  - `medium`: mixed payment/content cases, moderate missingness, moderate shift, more signal conflict
  - `hard`: payment/content/system cases with stronger OOD behavior, misleading confidence, more missing critical fields, and adversarial signal conflict
- Locked business-value matrix:
  - Legitimate case: correct `accept` is positive, `reject` incurs false-reject cost, `review` incurs review cost
  - Harmful case: correct `reject` is positive, `accept` incurs false-accept cost, `review` incurs review cost
  - Ambiguous case: `review` is the optimal action, while direct actions incur deterministic penalties
- Locked cost tiers:
  - `standard`: false accept `10`, false reject `4`, review `1`
  - `sensitive`: false accept `14`, false reject `5`, review `1`
  - `critical`: false accept `18`, false reject `6`, review `2`
- Locked positive action values:
  - Correct `accept`: `+3`
  - Correct `reject`: `+4`
  - Correct `review`: `+2`
- [x] Define how each case represents a realistic operational record instead of a synthetic single-field prompt.
- [x] Define which fields are visible to the agent for every case.
- [x] Define which fields remain hidden and are used only for grading.
- [x] Define how uncertainty, ambiguity, and missing signals appear in observations.
- [x] Define how review capacity is limited or penalized so `review` remains meaningful.
- [x] Define why false accepts and false rejects have asymmetric business cost.

### 0.3 Success Criteria Lock
- [x] Freeze the requirement for at least three tasks labeled easy, medium, and hard.
- [x] Freeze the requirement that every task must be solvable through the same typed action interface.
- [x] Freeze the requirement that grading is deterministic and normalized to the range `0.0` to `1.0`.
- [x] Freeze the requirement that rewards are meaningful at each step and not purely terminal.
- [x] Freeze the requirement that the final implementation must pass `openenv` validation.

## Stage 1
### 1.1 Repo Conversion Plan
- [x] Replace the starter echo-environment assumptions in `README.md` with OOD decision-making language.
- [x] Replace the starter action and observation schema in `models.py`.
- [x] Replace the starter transition logic in `server/my_env_environment.py`.
- [x] Confirm whether `client.py` remains the package client or is only kept for OpenEnv compatibility.
- [x] Add a root-level `inference.py` file and reserve that exact filename for the baseline submission script.
- [x] Decide whether new helper modules are needed for task generation, grading, prompts, and fixtures.

### 1.2 File Ownership Map
- [x] Mark `models.py` as the source of truth for typed action and observation models.
- [x] Mark `server/my_env_environment.py` as the source of truth for environment state, reset, and step behavior.
- [x] Mark `server/app.py` as the source of truth for OpenEnv server wiring.
- [x] Mark `openenv.yaml` as the source of truth for runtime metadata and entrypoint configuration.
- [x] Mark `server/Dockerfile` as the source of truth for container build and launch.
- [x] Mark `inference.py` as the source of truth for the baseline agent runner.

## Stage 2
### 2.1 Task Design
- [x] Define the easy task with strong in-distribution signals and low ambiguity.
- [x] Define the medium task with moderate distribution shift, missing fields, and mixed evidence.
- [x] Define the hard task with strong OOD behavior, adversarial or conflicting features, and higher review pressure.
- [x] Ensure all three tasks use the same action schema and grading API.
- [x] Ensure all three tasks feel like operational decision queues rather than unrelated mini-games.
- [x] Ensure each task has a clear narrative grounded in fraud, anomaly, or moderation workflows.
- [x] Verify OpenEnv can enumerate and run all tasks programmatically

### 2.2 Case Schema Design
- [x] Define the full case record structure for account, transaction, content, or event-level evidence.
- [x] Define which numeric features are visible to the agent.
- [x] Define which categorical features are visible to the agent.
- [x] Define which free-text rationale or notes are visible to the agent.
- [x] Define how task difficulty changes which fields are noisy, shifted, missing, or misleading.
- [x] Define hidden ground-truth labels, hidden OOD tags, and hidden business cost fields for grading.

### 2.3 Episode Structure
- [x] Define the episode length for each task.
- [x] Define whether all tasks use the same number of cases or task-specific counts.
- [x] Define how `reset()` selects the task and initial episode content.
- [x] Define whether the environment supports explicit task selection through reset metadata or config.
- [x] Define the terminal condition for the episode.
- [x] Define how step count, processed cases, and remaining budget are tracked in state.

## Stage 3
### 3.1 Typed Model Redesign
- [x] Replace `MyAction` with a typed action model that only permits valid decision actions.
- [x] Add typed action fields for decision rationale, optional confidence, and optional review note if needed.
- [x] Replace `MyObservation` with a typed observation model reflecting the current case, queue status, and recent feedback.
- [x] Add typed observation fields for task name, difficulty, remaining cases, review budget, and cumulative metrics.
- [x] Ensure all model fields have explicit types and clear field descriptions.
- [x] Ensure no required field depends on ambiguous free-form runtime inference.

### 3.2 State Design
- [x] Define the environment state object content beyond `episode_id` and `step_count`.
- [x] Decide where the current task definition is stored.
- [x] Decide where pending cases are stored.
- [x] Decide where cumulative reward and score components are stored.
- [x] Decide where review usage and error counts are stored.
- [x] Decide which state fields are exposed through the OpenEnv `state` property versus kept internal.

### 3.3 Schema Stability
- [x] Ensure the action schema remains stable across easy, medium, and hard tasks.
- [x] Ensure the observation schema remains stable across all tasks and steps.
- [ ] Ensure all metadata keys are deterministic and documented.
- [ ] Ensure the schema supports OpenEnv serialization without custom hacks.
- [ ] Ensure the schema remains lightweight enough for sub-20-minute evaluation runs.

## Stage 4
### 4.1 Deterministic Task Bank
- [x] Create a deterministic task bank for easy, medium, and hard episodes.
- [x] Assign a fixed seed strategy so the same task definition always produces the same cases.
- [x] Create realistic feature distributions for each domain scenario.
- [x] Create explicit OOD shift patterns for medium and hard tasks.
- [x] Create adversarial or borderline examples that make `review` strategically necessary.
- [x] Create enough case diversity to prevent trivial one-rule policies from winning.

### 4.2 Data Quality Controls
- [x] Verify every case has a single authoritative hidden label.
- [x] Verify every case has a business cost profile for false accept, false reject, and review actions.
- [x] Verify every case has enough visible evidence for an agent to justify its choice.
- [x] Verify no case contains contradictory grading targets.
- [x] Verify no task accidentally leaks hidden labels through visible fields.
- [ ] Verify task definitions are deterministic across local runs and container runs.

### 4.3 Difficulty Calibration
- [x] Calibrate the easy task so a simple baseline can achieve a non-trivial but imperfect score.
- [x] Calibrate the medium task so robust reasoning and review usage clearly outperform naive thresholds.
- [x] Calibrate the hard task so OOD awareness matters and overconfident accept/reject behavior is punished.
- [x] Verify task difficulty increases in a visible and defensible way from easy to hard.
- [ ] Verify the hard task is still tractable within hackathon runtime limits.

## Stage 5
### 5.1 Environment Reset Logic
- [x] Redesign `reset()` in `server/my_env_environment.py` around task initialization instead of echo behavior.
- [x] Ensure `reset()` creates a fresh episode id.
- [x] Ensure `reset()` resets step count, cumulative reward, score counters, and review budget.
- [x] Ensure `reset()` returns the first actionable observation immediately.
- [x] Ensure `reset()` works deterministically for a given task selection and seed.
- [ ] Ensure the deployed Hugging Face Space responds successfully to `reset()`.

### 5.2 Environment Step Logic
- [x] Redesign `step()` to consume one case decision per call.
- [x] Validate the action before mutating state.
- [x] Apply business-cost logic for `accept`.
- [x] Apply business-cost logic for `reject`.
- [x] Apply business-cost logic for `review`.
- [x] Advance to the next case after scoring the current action.
- [x] Mark `done` only when the episode is complete.
- [x] Return updated observation, reward, done flag, and deterministic metadata on every step.
- [ ] Ensure invalid or malformed actions are handled deterministically with a fallback or penalty

### 5.3 Review Workflow Realism
- [ ] Define exactly what `review` means operationally for each domain scenario.
- [x] Decide whether `review` resolves the case immediately or defers with a bounded penalty.
- [x] Apply a measurable cost to overusing review.
- [ ] Prevent `review` from becoming a dominant always-safe policy.
- [x] Ensure some borderline cases are best handled by `review`.
- [x] Ensure some clearly labeled cases are worse if sent to `review`.

## Stage 6
### 6.1 Reward Function Design
- [x] Define per-step reward contributions for correct `accept`, correct `reject`, and justified `review`.
- [x] Define per-step penalties for costly false accepts.
- [x] Define per-step penalties for costly false rejects.
- [x] Define per-step penalties for wasteful review usage.
- [x] Ensure the reward function is dense enough to guide agent behavior during the full episode.
- [ ] Ensure the reward function aligns with the final grader rather than rewarding contradictory behavior.

### 6.2 Score Normalization
- [ ] Define the deterministic mapping from raw business value to normalized score.
- [ ] Ensure every episode score is clipped or normalized into `0.0` to `1.0`.
- [ ] Ensure impossible edge cases cannot produce scores below `0.0` or above `1.0`.
- [ ] Define whether the public reward shown during steps matches the final normalized score or a shaped intermediate value.
- [ ] Define how aggregate score is computed across easy, medium, and hard tasks.

### 6.3 Anti-Shortcut Checks
- [x] Verify an always-accept policy scores poorly on at least one task.
- [x] Verify an always-reject policy scores poorly on at least one task.
- [x] Verify an always-review policy scores poorly due to explicit review cost or budget pressure.
- [ ] Verify the optimal policy requires reading task evidence and reacting to OOD signals.

## Stage 7
### 7.1 Deterministic Grader Implementation
- [ ] Implement a deterministic grading path that uses only hidden task data and recorded actions.
- [ ] Ensure the grader has no dependence on model outputs other than the chosen action and allowed auxiliary fields.
- [ ] Ensure grading produces the same result across repeated runs.
- [ ] Ensure grading works for each step and for the final episode summary.
- [ ] Ensure grading handles invalid actions with a deterministic penalty path.

### 7.2 Grader Test Coverage
- [ ] Add tests for correct scoring of `accept` on positive and negative cases.
- [ ] Add tests for correct scoring of `reject` on positive and negative cases.
- [ ] Add tests for correct scoring of `review` on borderline and obvious cases.
- [ ] Add tests for review budget exhaustion behavior if a budget exists.
- [ ] Add tests for normalization boundaries at `0.0` and `1.0`.
- [ ] Add golden tests for one fully worked easy episode, one medium episode, and one hard episode.

### 7.3 Grader Transparency
- [ ] Define deterministic metadata fields that explain score components without leaking future labels.
- [ ] Ensure observation metadata is useful for debugging but does not reveal hidden truth before the action.
- [ ] Ensure final evaluation outputs are traceable to business-cost components.

## Stage 8
### 8.1 OpenEnv Client Compatibility
- [x] Update `client.py` to parse the new observation schema correctly.
- [x] Ensure the client continues to support `reset()`, `step()`, and `state()` operations.
- [x] Ensure state parsing reflects the new episode counters and identifiers.
- [x] Ensure package exports remain valid after model and environment renames.
- [x] Ensure no starter echo-environment assumptions remain in client-side parsing.

### 8.2 Baseline Agent Contract
- [x] Create `inference.py` at the repository root.
- [x] Ensure `inference.py` uses the OpenAI client for inference.
- [x] Ensure `inference.py` reads `API_BASE_URL` from the environment.
- [x] Ensure `inference.py` reads `MODEL_NAME` from the environment.
- [x] Ensure `inference.py` reads `HF_TOKEN` from the environment if deployment or gated assets require it.
- [ ] Ensure `inference.py` can connect to the running environment and complete an episode end-to-end.

### 8.3 Baseline Agent Behavior
- [x] Define the system prompt for operational decision-making behavior.
- [x] Define the user prompt format from the current observation.
- [x] Define strict output parsing so the model only returns valid actions.
- [x] Add retry or fallback behavior for malformed model responses.
- [x] Log per-step decisions, rewards, and final score for debugging.
- [ ] Ensure the baseline finishes within the runtime budget.

## Stage 9
### 9.1 OpenEnv Server Wiring
- [x] Update `server/app.py` so the app wires the final environment class and final models.
- [x] Ensure the app still exposes the expected OpenEnv endpoints.
- [ ] Ensure the app works with local development and Docker execution.
- [ ] Ensure session behavior is correct for the intended concurrency model.
- [x] Ensure `env_name` and runtime labels match the final project identity.

### 9.2 Manifest and Packaging
- [x] Update `openenv.yaml` with the final valid spec fields.
- [x] Verify the `app` target in `openenv.yaml` matches the deployed server entrypoint.
- [x] Verify the port in `openenv.yaml` matches the runtime server port.
- [x] Verify `pyproject.toml` metadata matches the final environment purpose.
- [ ] Verify package imports work both from source and inside the container.

### 9.3 Documentation
- [x] Rewrite `README.md` around the OOD decision-making environment.
- [x] Document the easy, medium, and hard tasks in `README.md`.
- [x] Document the action semantics for `accept`, `reject`, and `review`.
- [ ] Document the reward and grading philosophy at a high level.
- [x] Document required environment variables including `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
- [ ] Document local run, Docker run, and validation commands.

## Stage 10
### 10.1 Docker Build Path
- [ ] Update `server/Dockerfile` if new files or dependencies require different copy or install behavior.
- [ ] Ensure the Docker build completes from the repository root.
- [ ] Ensure the container starts the OpenEnv FastAPI app without manual patching.
- [ ] Ensure the container exposes the expected port and health endpoint.
- [ ] Ensure the container image contains `inference.py` if evaluation expects it in the project root.

### 10.2 Runtime Budget Validation
- [ ] Measure one full baseline run locally.
- [ ] Measure one full baseline run inside Docker.
- [ ] Verify the full evaluation path remains under 20 minutes on a 2 vCPU and 8 GB RAM budget.
- [ ] Reduce task size, prompt size, or model round-trips if runtime exceeds the budget.
- [ ] Re-run timing after every major change to task count or prompt format.

### 10.3 Resource Stability
- [ ] Verify memory usage remains stable across repeated resets.
- [ ] Verify no task generation path requires internet access at runtime.
- [ ] Verify deterministic fixtures are bundled with the repo if external downloads are not allowed.
- [ ] Verify there are no hidden heavyweight dependencies that slow cold start.
- [ ] Ensure no runtime dependency on external APIs or downloads (except LLM calls)

## Stage 11
### 11.1 Local Validation
- [ ] Run unit tests for models, environment logic, and grader behavior.
- [ ] Run integration tests for reset-step-state sequences.
- [ ] Run a baseline episode through `inference.py`.
- [ ] Run Docker build and container smoke tests.
- [ ] Run `openenv` validation locally and fix all reported issues.
- [ ] Re-run validation after every manifest, schema, or Docker change.
- [ ] Verify baseline produces reproducible scores across multiple runs

### 11.2 Hugging Face Space Validation
- [ ] Prepare the repo for Hugging Face Space deployment without local-only assumptions.
- [ ] Verify the Space starts successfully with the Docker configuration.
- [ ] Verify the Space responds to `reset()`.
- [ ] Verify the Space supports a full episode through the OpenEnv API.
- [ ] Verify any required tokens or secrets are mapped to the expected environment variables.
- [ ] Verify README front matter and app metadata do not conflict with the deployment target.

### 11.3 Final Quality Pass
- [ ] Remove all echo-environment references from code, docs, and metadata.
- [ ] Remove dead starter logic and unused dependencies.
- [ ] Ensure logs are useful for debugging but not excessively verbose.
- [ ] Ensure deterministic seeds are centralized and documented.
- [ ] Ensure no TODO markers remain in submission-critical files.

## Special Section: OpenEnv Spec Compliance
- [x] Confirm the environment exposes typed action and observation models.
- [x] Confirm `reset()` returns a valid observation for a fresh episode.
- [x] Confirm `step()` accepts the typed action model and returns a valid step result.
- [x] Confirm `state` returns a valid OpenEnv state object.
- [ ] Confirm schema serialization works for action, observation, and state models.
- [x] Confirm `openenv.yaml` is valid and points to the correct app entrypoint.
- [ ] Confirm the project passes OpenEnv validation without manual overrides.

## Special Section: inference.py Compliance
- [x] Confirm the file is named exactly `inference.py`.
- [x] Confirm `inference.py` uses the OpenAI client.
- [x] Confirm `inference.py` reads `API_BASE_URL`.
- [x] Confirm `inference.py` reads `MODEL_NAME`.
- [x] Confirm `inference.py` reads `HF_TOKEN`.
- [ ] Confirm `inference.py` can run against the local server.
- [ ] Confirm `inference.py` can run against the deployed Hugging Face Space.
- [x] Confirm `inference.py` produces valid `accept`, `reject`, or `review` actions only.

## Special Section: Grading Correctness
- [x] Confirm graders are deterministic.
- [ ] Confirm graders return values in the range `0.0` to `1.0`.
- [x] Confirm graders treat easy, medium, and hard tasks consistently.
- [x] Confirm graders penalize false accepts, false rejects, and unnecessary reviews appropriately.
- [ ] Confirm graders reward correct high-confidence decisions and justified reviews appropriately.
- [ ] Confirm shaped rewards shown during interaction do not contradict final grading.
- [ ] Confirm golden tests lock expected scores for representative episodes.

## Special Section: Deployment Validation
- [ ] Confirm the Dockerfile builds successfully from a clean environment.
- [ ] Confirm the Docker container starts and serves the OpenEnv app.
- [ ] Confirm the deployed Hugging Face Space exposes working API endpoints.
- [ ] Confirm the deployed Hugging Face Space responds to `reset()`.
- [ ] Confirm the full evaluation path stays under the runtime limit.
- [ ] Confirm all required environment variables are documented and supported.
- [ ] Confirm no secret is hardcoded in the repository.

## Final Pre-Submission Checklist
- [ ] At least three tasks exist and are labeled easy, medium, and hard.
- [ ] The environment represents a real-world OOD decision system rather than a toy example.
- [ ] The only decision actions are `accept`, `reject`, and `review`.
- [ ] Reward is meaningful and non-sparse across the episode.
- [ ] Grading is deterministic and normalized to `0.0` to `1.0`.
- [ ] Typed models are complete and OpenEnv-compliant.
- [ ] `reset()`, `step()`, and `state` all behave correctly.
- [ ] `inference.py` exists and meets all naming and environment-variable requirements.
- [ ] `openenv.yaml` is valid.
- [ ] Docker build and run are successful.
- [ ] Hugging Face Space deployment responds to `reset()`.
- [ ] Runtime stays below 20 minutes on the target hardware budget.
- [ ] OpenEnv validation passes.
- [ ] README, metadata, and manifests reflect the final environment accurately.
- [ ] No starter echo-environment behavior remains anywhere in the submission.
