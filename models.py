# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the operational risk triage environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["accept", "reject", "review"]
TaskName = Literal["easy", "medium", "hard"]
DomainHint = Literal["payment", "content", "system"]
CaseClass = Literal["legitimate", "harmful", "ambiguous"]
CostTier = Literal["standard", "sensitive", "critical"]
OODType = Literal["none", "feature_shift", "context_shift", "adversarial_conflict"]
EpisodeStatus = Literal["ready", "in_progress", "completed"]


class TriageAction(Action):
    """Typed agent decision for the operational risk triage workflow."""

    decision: Decision = Field(
        ...,
        description="Final action to take for the current case.",
    )
    rationale: str = Field(
        default="",
        max_length=512,
        description="Short explanation for the chosen decision.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional self-reported confidence in the chosen decision.",
    )


class TriageCaseView(BaseModel):
    """Visible case record presented to the agent at a single step."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    case_id: str = Field(..., description="Stable identifier for the current case.")
    task_name: TaskName = Field(..., description="Difficulty of the active episode.")
    domain_hint: DomainHint = Field(..., description="Operational domain for the case.")
    event_type: str = Field(..., description="Specific event category within the domain.")
    impact_score: int = Field(
        ...,
        ge=1,
        le=100,
        description="Estimated business impact if the case is mishandled.",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Primary risk model score for the case.",
    )
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Novelty or abnormality score from behavioral systems.",
    )
    history_risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Risk estimate derived from prior actor or account history.",
    )
    model_recommendation: Decision = Field(
        ...,
        description="Recommendation from the automated triage model.",
    )
    model_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence attached to the model recommendation.",
    )
    uncertainty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated uncertainty of the automated stack on this case.",
    )
    novelty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Explicit signal that the case differs from familiar patterns.",
    )
    feature_completeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of expected evidence fields that are present.",
    )
    policy_flags: list[str] = Field(
        default_factory=list,
        description="Deterministic rule-based alerts attached to the case.",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Important unavailable fields for the current case.",
    )
    evidence_text: str = Field(
        ...,
        description="Short analyst-style summary of the case evidence.",
    )
    queue_position: int = Field(
        ...,
        ge=1,
        description="One-based index of the case inside the episode queue.",
    )
    remaining_cases: int = Field(
        ...,
        ge=0,
        description="Number of cases remaining after the current case.",
    )
    remaining_review_budget: int = Field(
        ...,
        ge=0,
        description="Review actions still available after the current case.",
    )


class TriageObservation(Observation):
    """Full observation returned by reset() and step()."""

    task_name: TaskName = Field(..., description="Active task for the episode.")
    episode_status: EpisodeStatus = Field(
        default="ready",
        description="Lifecycle state of the current episode.",
    )
    current_case: TriageCaseView | None = Field(
        default=None,
        description="Visible case to act on. None when the episode is complete.",
    )
    cases_total: int = Field(
        default=0,
        ge=0,
        description="Total number of cases in the episode.",
    )
    processed_cases: int = Field(
        default=0,
        ge=0,
        description="Number of cases already processed in the episode.",
    )
    remaining_cases: int = Field(
        default=0,
        ge=0,
        description="Number of cases still pending in the episode.",
    )
    remaining_review_budget: int = Field(
        default=0,
        ge=0,
        description="Remaining review actions for the current episode.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Running sum of step rewards for the active episode.",
    )
    normalized_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Running normalized score for processed cases, if available.",
    )
    raw_score_min: float = Field(
        default=0.0,
        description="Deterministic lower bound for the full-episode raw score.",
    )
    raw_score_max: float = Field(
        default=0.0,
        description="Deterministic upper bound for the full-episode raw score.",
    )
    raw_score_optimal: float = Field(
        default=0.0,
        description="Deterministic raw score achieved by the hidden optimal policy.",
    )
    accepted_count: int = Field(
        default=0,
        ge=0,
        description="Number of accepted cases so far.",
    )
    rejected_count: int = Field(
        default=0,
        ge=0,
        description="Number of rejected cases so far.",
    )
    reviewed_count: int = Field(
        default=0,
        ge=0,
        description="Number of successful review escalations so far.",
    )
    last_decision: Decision | None = Field(
        default=None,
        description="Most recent decision applied by the agent.",
    )
    last_feedback: str | None = Field(
        default=None,
        description="Deterministic feedback about the most recent decision.",
    )
    last_outcome_category: str | None = Field(
        default=None,
        description="Deterministic score-category label for the most recent action.",
    )


class TriageState(State):
    """Serializable environment state exposed through the OpenEnv state endpoint."""

    task_name: TaskName = Field(
        default="easy",
        description="Task currently loaded in the environment.",
    )
    episode_status: EpisodeStatus = Field(
        default="ready",
        description="Lifecycle state of the active episode.",
    )
    cases_total: int = Field(
        default=0,
        ge=0,
        description="Total number of cases in the current episode.",
    )
    processed_cases: int = Field(
        default=0,
        ge=0,
        description="Number of cases already processed.",
    )
    remaining_cases: int = Field(
        default=0,
        ge=0,
        description="Number of cases still pending.",
    )
    review_budget_total: int = Field(
        default=0,
        ge=0,
        description="Review budget configured for the task.",
    )
    remaining_review_budget: int = Field(
        default=0,
        ge=0,
        description="Review actions still available.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Running sum of rewards across the episode.",
    )
    normalized_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Running normalized score for processed cases, if available.",
    )
    raw_score_min: float = Field(
        default=0.0,
        description="Deterministic lower bound for the full-episode raw score.",
    )
    raw_score_max: float = Field(
        default=0.0,
        description="Deterministic upper bound for the full-episode raw score.",
    )
    raw_score_optimal: float = Field(
        default=0.0,
        description="Deterministic raw score achieved by the hidden optimal policy.",
    )
    accepted_count: int = Field(
        default=0,
        ge=0,
        description="Number of accepted decisions applied so far.",
    )
    rejected_count: int = Field(
        default=0,
        ge=0,
        description="Number of rejected decisions applied so far.",
    )
    reviewed_count: int = Field(
        default=0,
        ge=0,
        description="Number of successful review escalations so far.",
    )
    last_decision: Decision | None = Field(
        default=None,
        description="Most recent decision applied by the environment.",
    )
    last_outcome_category: str | None = Field(
        default=None,
        description="Deterministic score-category label for the most recent action.",
    )


# Backwards-compatible aliases while the package name remains `my_env`.
MyAction = TriageAction
MyObservation = TriageObservation
MyState = TriageState
