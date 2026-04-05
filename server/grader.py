"""Deterministic raw-score grading and normalization for triage episodes."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

try:
    from ..models import Decision, TaskName
    from .task_bank import TaskCase, TaskDefinition
except ImportError:
    from models import Decision, TaskName  # type: ignore
    from server.task_bank import TaskCase, TaskDefinition  # type: ignore


REVIEW_EXHAUSTION_PENALTY = -3.0
INVALID_ACTION_PENALTY = 2.0
VALID_DECISIONS: tuple[Decision, ...] = ("accept", "reject", "review")


@dataclass(frozen=True)
class ScoreBounds:
    """Deterministic raw-score bounds for a task or processed prefix."""

    min_raw_score: float
    max_raw_score: float
    optimal_raw_score: float


@dataclass(frozen=True)
class StepGrade:
    """Deterministic result for one action on one case."""

    requested_decision: str
    applied_decision: Decision | None
    outcome_category: str
    raw_reward: float
    review_honored: bool
    remaining_review_budget: int
    cumulative_raw_score: float
    normalized_score: float | None
    score_bounds: ScoreBounds


@dataclass(frozen=True)
class EpisodeGrade:
    """Deterministic summary for a full task episode."""

    task_name: TaskName
    raw_score: float
    normalized_score: float
    score_bounds: ScoreBounds
    steps: tuple[StepGrade, ...]
    processed_cases: int
    review_budget_total: int
    remaining_review_budget: int


def _case_min_raw_score(case: TaskCase) -> float:
    return min(case.action_value_accept, case.action_value_reject, case.action_value_review)


def _case_max_raw_score(case: TaskCase) -> float:
    return max(case.action_value_accept, case.action_value_reject, case.action_value_review)


def score_bounds_for_cases(cases: Iterable[TaskCase]) -> ScoreBounds:
    """Return deterministic raw-score bounds for an ordered case sequence."""

    case_list = tuple(cases)
    min_raw_score = round(sum(_case_min_raw_score(case) for case in case_list), 3)
    max_raw_score = round(sum(_case_max_raw_score(case) for case in case_list), 3)
    optimal_raw_score = round(sum(case.action_value(case.optimal_decision) for case in case_list), 3)
    return ScoreBounds(
        min_raw_score=min_raw_score,
        max_raw_score=max_raw_score,
        optimal_raw_score=optimal_raw_score,
    )


def score_bounds_for_task(definition: TaskDefinition) -> ScoreBounds:
    """Return deterministic full-episode raw-score bounds for a task."""

    return score_bounds_for_cases(definition.cases)


def prefix_score_bounds(definition: TaskDefinition, processed_cases: int) -> ScoreBounds:
    """Return bounds for the processed prefix of an episode."""

    if processed_cases <= 0:
        return ScoreBounds(min_raw_score=0.0, max_raw_score=0.0, optimal_raw_score=0.0)
    return score_bounds_for_cases(definition.cases[:processed_cases])


def normalize_raw_score(raw_score: float, bounds: ScoreBounds) -> float:
    """Map a raw business-value score into the deterministic range [0.0, 1.0]."""

    if bounds.optimal_raw_score <= 0:
        return 1.0 if raw_score >= bounds.optimal_raw_score else 0.0
    normalized = raw_score / bounds.optimal_raw_score
    return round(max(0.0, min(1.0, normalized)), 6)


def _outcome_category(case: TaskCase, decision: Decision) -> str:
    if decision == case.optimal_decision:
        if decision == "accept":
            return "correct_accept"
        if decision == "reject":
            return "correct_reject"
        return "correct_review"
    if decision == "accept":
        return "false_accept"
    if decision == "reject":
        return "false_reject"
    return "wasteful_review"


def grade_step(
    definition: TaskDefinition,
    case_index: int,
    requested_decision: str,
    remaining_review_budget: int,
    cumulative_raw_score_before: float,
) -> StepGrade:
    """Grade one action deterministically against hidden task data."""

    case = definition.cases[case_index]
    applied_decision: Decision | None = None
    review_honored = False

    if requested_decision not in VALID_DECISIONS:
        raw_reward = round(_case_min_raw_score(case) - INVALID_ACTION_PENALTY, 3)
        remaining_review_budget_after = remaining_review_budget
        outcome_category = "invalid_action"
    else:
        applied_decision = requested_decision
        if requested_decision == "review":
            if remaining_review_budget > 0:
                raw_reward = case.action_value("review")
                remaining_review_budget_after = remaining_review_budget - 1
                review_honored = True
                outcome_category = _outcome_category(case, "review")
            else:
                raw_reward = round(REVIEW_EXHAUSTION_PENALTY - case.review_cost, 3)
                remaining_review_budget_after = 0
                outcome_category = "review_budget_exhausted"
        else:
            raw_reward = case.action_value(applied_decision)
            remaining_review_budget_after = remaining_review_budget
            outcome_category = _outcome_category(case, applied_decision)

    cumulative_raw_score = round(cumulative_raw_score_before + raw_reward, 3)
    bounds = prefix_score_bounds(definition, case_index + 1)
    normalized_score = None if case_index < 0 else normalize_raw_score(cumulative_raw_score, bounds)
    return StepGrade(
        requested_decision=requested_decision,
        applied_decision=applied_decision,
        outcome_category=outcome_category,
        raw_reward=raw_reward,
        review_honored=review_honored,
        remaining_review_budget=remaining_review_budget_after,
        cumulative_raw_score=cumulative_raw_score,
        normalized_score=normalized_score,
        score_bounds=bounds,
    )


def grade_episode(definition: TaskDefinition, actions: Iterable[str]) -> EpisodeGrade:
    """Grade a full deterministic episode from an ordered action sequence."""

    remaining_review_budget = definition.review_budget
    cumulative_raw_score = 0.0
    steps: list[StepGrade] = []
    action_list = list(actions)

    for case_index in range(len(definition.cases)):
        requested_decision = action_list[case_index] if case_index < len(action_list) else "__missing__"
        step_grade = grade_step(
            definition=definition,
            case_index=case_index,
            requested_decision=requested_decision,
            remaining_review_budget=remaining_review_budget,
            cumulative_raw_score_before=cumulative_raw_score,
        )
        steps.append(step_grade)
        remaining_review_budget = step_grade.remaining_review_budget
        cumulative_raw_score = step_grade.cumulative_raw_score

    full_bounds = score_bounds_for_task(definition)
    return EpisodeGrade(
        task_name=definition.name,
        raw_score=round(cumulative_raw_score, 3),
        normalized_score=normalize_raw_score(cumulative_raw_score, full_bounds),
        score_bounds=full_bounds,
        steps=tuple(steps),
        processed_cases=len(steps),
        review_budget_total=definition.review_budget,
        remaining_review_budget=remaining_review_budget,
    )


def aggregate_task_scores(grades: Iterable[EpisodeGrade]) -> float:
    """Aggregate normalized task scores using an equal-weight mean."""

    grade_list = tuple(grades)
    if not grade_list:
        raise ValueError("At least one episode grade is required to aggregate task scores.")
    return round(mean(grade.normalized_score for grade in grade_list), 6)
