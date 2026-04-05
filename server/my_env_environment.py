# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operational risk triage environment implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import Decision, TaskName, TriageAction, TriageObservation, TriageState
    from .grader import REVIEW_EXHAUSTION_PENALTY, ScoreBounds, StepGrade, grade_step, score_bounds_for_task
    from .task_bank import TaskCase, TaskDefinition, build_task_bank
except ImportError:
    from models import Decision, TaskName, TriageAction, TriageObservation, TriageState  # type: ignore
    from server.grader import (  # type: ignore
        REVIEW_EXHAUSTION_PENALTY,
        ScoreBounds,
        StepGrade,
        grade_step,
        score_bounds_for_task,
    )
    from server.task_bank import TaskCase, TaskDefinition, build_task_bank  # type: ignore


@dataclass
class EpisodeState:
    """Internal mutable episode state not exposed directly through OpenEnv."""

    task_name: TaskName
    definition: TaskDefinition
    score_bounds: ScoreBounds
    current_index: int = 0
    remaining_review_budget: int = 0
    cumulative_reward: float = 0.0
    normalized_score: float | None = None
    accepted_count: int = 0
    rejected_count: int = 0
    reviewed_count: int = 0
    last_decision: Decision | None = None
    last_feedback: str | None = None
    last_outcome_category: str | None = None
    last_grade: StepGrade | None = None
    done: bool = False

    @property
    def processed_cases(self) -> int:
        return self.current_index

    @property
    def remaining_cases(self) -> int:
        return len(self.definition.cases) - self.current_index

    @property
    def current_case(self) -> TaskCase | None:
        if self.done or self.current_index >= len(self.definition.cases):
            return None
        return self.definition.cases[self.current_index]


class MyEnvironment(Environment[TriageAction, TriageObservation, TriageState]):
    """Deterministic multi-case decision queue for operational risk triage."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task_bank = build_task_bank()
        self._state = TriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name="easy",
            episode_status="ready",
        )
        self._episode: EpisodeState | None = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: TaskName = "easy",
        **_: object,
    ) -> TriageObservation:
        """Reset the environment to a deterministic task queue."""

        if task not in self._task_bank:
            raise ValueError(f"Unknown task '{task}'. Expected one of: easy, medium, hard.")

        definition = self._task_bank[task]
        self._episode = EpisodeState(
            task_name=task,
            definition=definition,
            score_bounds=score_bounds_for_task(definition),
            current_index=0,
            remaining_review_budget=definition.review_budget,
        )
        self._state = TriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task,
            episode_status="ready",
            cases_total=len(definition.cases),
            processed_cases=0,
            remaining_cases=len(definition.cases),
            review_budget_total=definition.review_budget,
            remaining_review_budget=definition.review_budget,
            cumulative_reward=0.0,
            normalized_score=None,
            raw_score_min=self._episode.score_bounds.min_raw_score,
            raw_score_max=self._episode.score_bounds.max_raw_score,
            raw_score_optimal=self._episode.score_bounds.optimal_raw_score,
            accepted_count=0,
            rejected_count=0,
            reviewed_count=0,
            last_decision=None,
            last_outcome_category=None,
        )
        return self._build_observation(
            reward=None,
            feedback=(
                f"Loaded the deterministic '{task}' queue with {len(definition.cases)} cases "
                f"and review budget {definition.review_budget}."
            ),
            seed=seed,
        )

    def step(
        self,
        action: TriageAction,
        timeout_s: Optional[float] = None,
        **_: object,
    ) -> TriageObservation:
        """Process exactly one case decision and advance the queue."""

        del timeout_s

        if self._episode is None:
            raise RuntimeError("Environment must be reset before step() is called.")
        if self._episode.done or self._episode.current_case is None:
            return self._build_observation(
                reward=0.0,
                feedback="Episode already completed. Reset the environment to start a new queue.",
            )

        case = self._episode.current_case
        requested_decision = action.decision
        step_grade = grade_step(
            definition=self._episode.definition,
            case_index=self._episode.current_index,
            requested_decision=requested_decision,
            remaining_review_budget=self._episode.remaining_review_budget,
            cumulative_raw_score_before=self._episode.cumulative_reward,
        )
        applied_decision = step_grade.applied_decision
        reward = step_grade.raw_reward
        review_honored = step_grade.review_honored

        self._episode.remaining_review_budget = step_grade.remaining_review_budget
        self._episode.normalized_score = step_grade.normalized_score
        self._episode.last_grade = step_grade

        if applied_decision == "accept":
            self._episode.accepted_count += 1
        elif applied_decision == "reject":
            self._episode.rejected_count += 1
        elif applied_decision == "review" and review_honored:
            self._episode.reviewed_count += 1

        self._episode.current_index += 1
        self._episode.cumulative_reward = step_grade.cumulative_raw_score
        self._episode.last_decision = applied_decision
        self._episode.last_outcome_category = step_grade.outcome_category
        self._episode.done = self._episode.current_index >= len(self._episode.definition.cases)
        self._episode.last_feedback = self._decision_feedback(
            case=case,
            requested_decision=requested_decision,
            reward=reward,
            review_honored=review_honored,
        )

        self._state.step_count += 1
        self._sync_public_state()

        return self._build_observation(reward=reward, feedback=self._episode.last_feedback)

    @property
    def state(self) -> TriageState:
        """Return the serializable OpenEnv state."""

        return self._state

    def _sync_public_state(self) -> None:
        if self._episode is None:
            return

        episode_status = "completed" if self._episode.done else "in_progress"
        self._state.task_name = self._episode.task_name
        self._state.episode_status = episode_status
        self._state.cases_total = len(self._episode.definition.cases)
        self._state.processed_cases = self._episode.processed_cases
        self._state.remaining_cases = self._episode.remaining_cases
        self._state.review_budget_total = self._episode.definition.review_budget
        self._state.remaining_review_budget = self._episode.remaining_review_budget
        self._state.cumulative_reward = self._episode.cumulative_reward
        self._state.normalized_score = self._episode.normalized_score
        self._state.raw_score_min = self._episode.score_bounds.min_raw_score
        self._state.raw_score_max = self._episode.score_bounds.max_raw_score
        self._state.raw_score_optimal = self._episode.score_bounds.optimal_raw_score
        self._state.accepted_count = self._episode.accepted_count
        self._state.rejected_count = self._episode.rejected_count
        self._state.reviewed_count = self._episode.reviewed_count
        self._state.last_decision = self._episode.last_decision
        self._state.last_outcome_category = self._episode.last_outcome_category

    def _build_observation(
        self,
        reward: float | None,
        feedback: str,
        seed: Optional[int] = None,
    ) -> TriageObservation:
        if self._episode is None:
            raise RuntimeError("Episode state is unavailable. Call reset() first.")

        current_case = self._episode.current_case
        observation_done = self._episode.done
        episode_status = "completed" if observation_done else ("ready" if self._state.step_count == 0 else "in_progress")
        case_view = None
        if current_case is not None:
            case_view = current_case.to_view(
                task_name=self._episode.task_name,
                queue_position=self._episode.current_index + 1,
                remaining_cases=max(len(self._episode.definition.cases) - self._episode.current_index - 1, 0),
                remaining_review_budget=self._episode.remaining_review_budget,
            )

        metadata = {
            "available_tasks": ["easy", "medium", "hard"],
            "review_budget_total": self._episode.definition.review_budget,
            "score_raw_min": self._episode.score_bounds.min_raw_score,
            "score_raw_max": self._episode.score_bounds.max_raw_score,
            "score_raw_optimal": self._episode.score_bounds.optimal_raw_score,
            "score_aggregate_method": "mean_normalized_score_across_tasks",
            "public_reward_mode": "raw_business_value",
            "normalized_score_definition": "clip(raw_score / raw_score_optimal, 0.0, 1.0)",
            "seed": seed,
        }
        if current_case is not None:
            metadata["case_domain"] = current_case.domain_hint
            metadata["case_cost_tier"] = current_case.cost_tier
        if self._episode.last_grade is not None:
            metadata["last_outcome_category"] = self._episode.last_grade.outcome_category
            metadata["last_review_honored"] = self._episode.last_grade.review_honored
            metadata["last_normalized_score"] = self._episode.last_grade.normalized_score
        if observation_done:
            metadata["final_normalized_score"] = self._episode.normalized_score

        return TriageObservation(
            task_name=self._episode.task_name,
            episode_status=episode_status,
            current_case=case_view,
            cases_total=len(self._episode.definition.cases),
            processed_cases=self._episode.processed_cases,
            remaining_cases=self._episode.remaining_cases,
            remaining_review_budget=self._episode.remaining_review_budget,
            cumulative_reward=self._episode.cumulative_reward,
            normalized_score=self._episode.normalized_score,
            raw_score_min=self._episode.score_bounds.min_raw_score,
            raw_score_max=self._episode.score_bounds.max_raw_score,
            raw_score_optimal=self._episode.score_bounds.optimal_raw_score,
            accepted_count=self._episode.accepted_count,
            rejected_count=self._episode.rejected_count,
            reviewed_count=self._episode.reviewed_count,
            last_decision=self._episode.last_decision,
            last_feedback=feedback,
            last_outcome_category=self._episode.last_outcome_category,
            done=observation_done,
            reward=reward,
            metadata=metadata,
        )

    def _decision_feedback(
        self,
        case: TaskCase,
        requested_decision: Decision,
        reward: float,
        review_honored: bool,
    ) -> str:
        if requested_decision == "review" and not review_honored:
            return "Review capacity was exhausted, so the escalation request was penalized."
        if requested_decision == case.optimal_decision:
            if requested_decision == "accept":
                return "Direct approval aligned with hidden adjudication for this case."
            if requested_decision == "reject":
                return "Direct rejection aligned with hidden adjudication for this case."
            return "Escalation was the preferred action for this ambiguous case."
        if requested_decision == "accept":
            return "Approval exposed the system to higher hidden business risk than the preferred action."
        if requested_decision == "reject":
            return "Rejection created more downstream cost than the preferred action."
        if reward < 0:
            return "Escalation consumed analyst capacity on a case that was better handled directly."
        return "Decision completed with a deterministic intermediate outcome."


# Backwards-compatible alias while the module path remains unchanged.
TriageEnvironment = MyEnvironment
