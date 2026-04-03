# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic task definitions for the operational risk triage environment."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ..models import (
        CaseClass,
        CostTier,
        Decision,
        DomainHint,
        OODType,
        TaskName,
        TriageCaseView,
    )
except ImportError:
    from models import (  # type: ignore
        CaseClass,
        CostTier,
        Decision,
        DomainHint,
        OODType,
        TaskName,
        TriageCaseView,
    )


FALSE_ACCEPT_COST = {
    "standard": 10.0,
    "sensitive": 14.0,
    "critical": 18.0,
}
FALSE_REJECT_COST = {
    "standard": 4.0,
    "sensitive": 5.0,
    "critical": 6.0,
}
REVIEW_COST = {
    "standard": 1.0,
    "sensitive": 1.0,
    "critical": 2.0,
}
POSITIVE_ACTION_VALUE = {
    "accept": 3.0,
    "reject": 4.0,
    "review": 2.0,
}
VARIANT_NOTES = {
    "easy": (
        "Prior queue activity for the actor is stable.",
        "Recent metadata shows a routine account change but no confirmed incident.",
        "External corroboration is present and internally consistent.",
        "The case arrives during a normal traffic window.",
    ),
    "medium": (
        "Upstream features disagree slightly across subsystems.",
        "A recent context change reduces confidence in the default model policy.",
        "Analyst handoff notes are incomplete for this record.",
        "Rule-based detections and learned signals do not fully agree.",
        "The queue labels this item as a moderate-shift pattern.",
    ),
    "hard": (
        "Cross-system telemetry is partially stale for this record.",
        "The strongest automated signal comes from a model with known calibration drift.",
        "An operational exception may explain part of the anomaly.",
        "The evidence bundle mixes normal and adversarial-looking traits.",
        "Prior similar cases split evenly between benign and harmful outcomes.",
    ),
}
TASK_EXTRA_MISSING = {
    "easy": (),
    "medium": ("device_graph", "caller_identity", "prior_history"),
    "hard": ("device_graph", "deployment_context", "vendor_attestation"),
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, round(value, 3)))


def _optimal_decision(true_case_class: CaseClass) -> Decision:
    if true_case_class == "legitimate":
        return "accept"
    if true_case_class == "harmful":
        return "reject"
    return "review"


def _action_values(true_case_class: CaseClass, cost_tier: CostTier) -> dict[Decision, float]:
    false_accept_cost = FALSE_ACCEPT_COST[cost_tier]
    false_reject_cost = FALSE_REJECT_COST[cost_tier]
    review_cost = REVIEW_COST[cost_tier]

    if true_case_class == "legitimate":
        return {
            "accept": POSITIVE_ACTION_VALUE["accept"],
            "reject": -false_reject_cost,
            "review": -review_cost,
        }
    if true_case_class == "harmful":
        return {
            "accept": -false_accept_cost,
            "reject": POSITIVE_ACTION_VALUE["reject"],
            "review": -review_cost,
        }
    return {
        "accept": -(false_accept_cost - 4.0),
        "reject": -false_reject_cost,
        "review": POSITIVE_ACTION_VALUE["review"],
    }


@dataclass(frozen=True)
class CaseBlueprint:
    """Base case profile expanded deterministically into queue items."""

    domain_hint: DomainHint
    event_type: str
    impact_score: int
    risk_score: float
    anomaly_score: float
    history_risk_score: float
    model_recommendation: Decision
    model_confidence: float
    uncertainty_score: float
    novelty_score: float
    feature_completeness: float
    policy_flags: tuple[str, ...]
    missing_fields: tuple[str, ...]
    evidence_text: str
    true_case_class: CaseClass
    cost_tier: CostTier
    ood_type: OODType = "none"


@dataclass(frozen=True)
class TaskCase:
    """Deterministic case record including hidden grading information."""

    case_id: str
    domain_hint: DomainHint
    event_type: str
    impact_score: int
    risk_score: float
    anomaly_score: float
    history_risk_score: float
    model_recommendation: Decision
    model_confidence: float
    uncertainty_score: float
    novelty_score: float
    feature_completeness: float
    policy_flags: tuple[str, ...]
    missing_fields: tuple[str, ...]
    evidence_text: str
    true_case_class: CaseClass
    optimal_decision: Decision
    is_ood: bool
    ood_type: OODType
    cost_tier: CostTier
    false_accept_cost: float
    false_reject_cost: float
    review_cost: float
    action_value_accept: float
    action_value_reject: float
    action_value_review: float

    def action_value(self, decision: Decision) -> float:
        if decision == "accept":
            return self.action_value_accept
        if decision == "reject":
            return self.action_value_reject
        return self.action_value_review

    def to_view(
        self,
        task_name: TaskName,
        queue_position: int,
        remaining_cases: int,
        remaining_review_budget: int,
    ) -> TriageCaseView:
        return TriageCaseView(
            case_id=self.case_id,
            task_name=task_name,
            domain_hint=self.domain_hint,
            event_type=self.event_type,
            impact_score=self.impact_score,
            risk_score=self.risk_score,
            anomaly_score=self.anomaly_score,
            history_risk_score=self.history_risk_score,
            model_recommendation=self.model_recommendation,
            model_confidence=self.model_confidence,
            uncertainty_score=self.uncertainty_score,
            novelty_score=self.novelty_score,
            feature_completeness=self.feature_completeness,
            policy_flags=list(self.policy_flags),
            missing_fields=list(self.missing_fields),
            evidence_text=self.evidence_text,
            queue_position=queue_position,
            remaining_cases=remaining_cases,
            remaining_review_budget=remaining_review_budget,
        )


@dataclass(frozen=True)
class TaskDefinition:
    """Fixed episode definition for one task difficulty."""

    name: TaskName
    review_budget: int
    cases: tuple[TaskCase, ...]


def _build_case(
    task_name: TaskName,
    case_index: int,
    variant_index: int,
    blueprint: CaseBlueprint,
) -> TaskCase:
    variant_strengths = {
        "easy": (0.0, 0.018, -0.015, 0.022),
        "medium": (0.0, 0.028, -0.02, 0.034, -0.014),
        "hard": (0.0, 0.038, -0.026, 0.03, -0.018),
    }
    strength = variant_strengths[task_name][variant_index % len(variant_strengths[task_name])]
    ood_bonus = 0.06 if blueprint.ood_type != "none" else 0.0
    confidence_shift = -0.4 * strength
    if task_name == "hard" and blueprint.ood_type != "none":
        confidence_shift = abs(strength) * 0.25

    extra_missing = ()
    if variant_index > 0 and TASK_EXTRA_MISSING[task_name]:
        extra_missing = (
            TASK_EXTRA_MISSING[task_name][variant_index % len(TASK_EXTRA_MISSING[task_name])],
        )

    action_values = _action_values(blueprint.true_case_class, blueprint.cost_tier)
    policy_flags = tuple(dict.fromkeys(blueprint.policy_flags))
    missing_fields = tuple(dict.fromkeys(blueprint.missing_fields + extra_missing))
    evidence_text = (
        f"{blueprint.evidence_text} "
        f"{VARIANT_NOTES[task_name][variant_index % len(VARIANT_NOTES[task_name])]}"
    )

    return TaskCase(
        case_id=f"{task_name}-{case_index:03d}",
        domain_hint=blueprint.domain_hint,
        event_type=blueprint.event_type,
        impact_score=blueprint.impact_score,
        risk_score=_clamp(blueprint.risk_score + strength),
        anomaly_score=_clamp(blueprint.anomaly_score + abs(strength) * 0.8 + ood_bonus),
        history_risk_score=_clamp(blueprint.history_risk_score + strength * 0.6),
        model_recommendation=blueprint.model_recommendation,
        model_confidence=_clamp(blueprint.model_confidence + confidence_shift),
        uncertainty_score=_clamp(blueprint.uncertainty_score + abs(strength) * 0.9 + ood_bonus),
        novelty_score=_clamp(blueprint.novelty_score + abs(strength) * 0.7 + ood_bonus),
        feature_completeness=_clamp(
            blueprint.feature_completeness - abs(strength) * 0.8 - (0.03 if extra_missing else 0.0)
        ),
        policy_flags=policy_flags,
        missing_fields=missing_fields,
        evidence_text=evidence_text,
        true_case_class=blueprint.true_case_class,
        optimal_decision=_optimal_decision(blueprint.true_case_class),
        is_ood=blueprint.ood_type != "none",
        ood_type=blueprint.ood_type,
        cost_tier=blueprint.cost_tier,
        false_accept_cost=FALSE_ACCEPT_COST[blueprint.cost_tier],
        false_reject_cost=FALSE_REJECT_COST[blueprint.cost_tier],
        review_cost=REVIEW_COST[blueprint.cost_tier],
        action_value_accept=action_values["accept"],
        action_value_reject=action_values["reject"],
        action_value_review=action_values["review"],
    )


def _expand_task(
    task_name: TaskName,
    review_budget: int,
    blueprint_counts: tuple[tuple[int, CaseBlueprint], ...],
) -> TaskDefinition:
    cases: list[TaskCase] = []
    case_index = 1
    for repeat_count, blueprint in blueprint_counts:
        for variant_index in range(repeat_count):
            cases.append(_build_case(task_name, case_index, variant_index, blueprint))
            case_index += 1
    return TaskDefinition(name=task_name, review_budget=review_budget, cases=tuple(cases))


def build_task_bank() -> dict[TaskName, TaskDefinition]:
    """Create the deterministic task bank used by Stage 1 scaffolding."""

    easy = _expand_task(
        "easy",
        review_budget=4,
        blueprint_counts=(
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="card_payment",
                    impact_score=40,
                    risk_score=0.14,
                    anomaly_score=0.12,
                    history_risk_score=0.18,
                    model_recommendation="accept",
                    model_confidence=0.92,
                    uncertainty_score=0.08,
                    novelty_score=0.12,
                    feature_completeness=0.98,
                    policy_flags=(),
                    missing_fields=(),
                    evidence_text=(
                        "Returning customer purchases a routine basket from a familiar device "
                        "and billing region."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="account_takeover",
                    impact_score=92,
                    risk_score=0.91,
                    anomaly_score=0.88,
                    history_risk_score=0.84,
                    model_recommendation="reject",
                    model_confidence=0.95,
                    uncertainty_score=0.12,
                    novelty_score=0.16,
                    feature_completeness=0.97,
                    policy_flags=("velocity_spike", "device_mismatch", "credential_reset"),
                    missing_fields=(),
                    evidence_text=(
                        "High-value wallet cash-out follows a password reset and login from "
                        "an unseen device."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="subscription_renewal",
                    impact_score=35,
                    risk_score=0.22,
                    anomaly_score=0.18,
                    history_risk_score=0.12,
                    model_recommendation="accept",
                    model_confidence=0.86,
                    uncertainty_score=0.1,
                    novelty_score=0.09,
                    feature_completeness=0.95,
                    policy_flags=("known_merchant",),
                    missing_fields=(),
                    evidence_text=(
                        "Recurring subscription renews on schedule with unchanged merchant, "
                        "amount, and account history."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="mule_transfer",
                    impact_score=78,
                    risk_score=0.83,
                    anomaly_score=0.79,
                    history_risk_score=0.74,
                    model_recommendation="reject",
                    model_confidence=0.89,
                    uncertainty_score=0.18,
                    novelty_score=0.24,
                    feature_completeness=0.94,
                    policy_flags=("velocity_spike", "linked_chargeback_cluster"),
                    missing_fields=(),
                    evidence_text=(
                        "Freshly linked beneficiary receives rapid back-to-back transfers after "
                        "small probing transactions."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="card_payment",
                    impact_score=64,
                    risk_score=0.56,
                    anomaly_score=0.61,
                    history_risk_score=0.49,
                    model_recommendation="review",
                    model_confidence=0.58,
                    uncertainty_score=0.72,
                    novelty_score=0.33,
                    feature_completeness=0.87,
                    policy_flags=("geo_mismatch", "device_mismatch"),
                    missing_fields=("prior_history",),
                    evidence_text=(
                        "Cross-border card-not-present purchase comes from a recently added "
                        "device, but shipping and billing details remain internally consistent."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                ),
            ),
        ),
    )

    medium = _expand_task(
        "medium",
        review_budget=4,
        blueprint_counts=(
            (
                4,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="post_submission",
                    impact_score=48,
                    risk_score=0.68,
                    anomaly_score=0.34,
                    history_risk_score=0.21,
                    model_recommendation="review",
                    model_confidence=0.63,
                    uncertainty_score=0.49,
                    novelty_score=0.31,
                    feature_completeness=0.88,
                    policy_flags=("policy_keyword",),
                    missing_fields=("content_context",),
                    evidence_text=(
                        "Quoted language from a news repost triggers a keyword rule, but prior "
                        "appeals for the account were upheld."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="card_payment",
                    impact_score=90,
                    risk_score=0.87,
                    anomaly_score=0.76,
                    history_risk_score=0.69,
                    model_recommendation="reject",
                    model_confidence=0.82,
                    uncertainty_score=0.22,
                    novelty_score=0.29,
                    feature_completeness=0.91,
                    policy_flags=("velocity_spike", "billing_mismatch"),
                    missing_fields=(),
                    evidence_text=(
                        "Multiple high-value checkout attempts fan out across merchants after a "
                        "fresh device enrollment."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="direct_message",
                    impact_score=72,
                    risk_score=0.73,
                    anomaly_score=0.57,
                    history_risk_score=0.67,
                    model_recommendation="accept",
                    model_confidence=0.79,
                    uncertainty_score=0.36,
                    novelty_score=0.46,
                    feature_completeness=0.84,
                    policy_flags=("watchlist_hit", "policy_keyword"),
                    missing_fields=("conversation_history",),
                    evidence_text=(
                        "Account with prior enforcement sends short coded abuse phrases that the "
                        "language model treats as low severity."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                    ood_type="adversarial_conflict",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="travel_booking",
                    impact_score=59,
                    risk_score=0.49,
                    anomaly_score=0.58,
                    history_risk_score=0.27,
                    model_recommendation="review",
                    model_confidence=0.55,
                    uncertainty_score=0.62,
                    novelty_score=0.41,
                    feature_completeness=0.83,
                    policy_flags=("geo_mismatch", "travel_pattern_change"),
                    missing_fields=("device_id",),
                    evidence_text=(
                        "Established customer books travel from a new country shortly after "
                        "confirming an itinerary on a known device."
                    ),
                    true_case_class="legitimate",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="post_submission",
                    impact_score=55,
                    risk_score=0.61,
                    anomaly_score=0.52,
                    history_risk_score=0.39,
                    model_recommendation="reject",
                    model_confidence=0.66,
                    uncertainty_score=0.69,
                    novelty_score=0.43,
                    feature_completeness=0.79,
                    policy_flags=("policy_keyword", "rapid_editing"),
                    missing_fields=("content_context", "prior_history"),
                    evidence_text=(
                        "Sarcastic post quotes prohibited language and removes surrounding "
                        "context through rapid edits."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
            (
                5,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="api_spike",
                    impact_score=84,
                    risk_score=0.69,
                    anomaly_score=0.91,
                    history_risk_score=0.44,
                    model_recommendation="review",
                    model_confidence=0.57,
                    uncertainty_score=0.74,
                    novelty_score=0.58,
                    feature_completeness=0.82,
                    policy_flags=("schema_shift", "api_burst"),
                    missing_fields=("caller_identity",),
                    evidence_text=(
                        "Traffic surge hits rarely used endpoints with a new payload shape during "
                        "off-hours."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
        ),
    )

    hard = _expand_task(
        "hard",
        review_budget=3,
        blueprint_counts=(
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="enterprise_payout",
                    impact_score=76,
                    risk_score=0.71,
                    anomaly_score=0.83,
                    history_risk_score=0.18,
                    model_recommendation="reject",
                    model_confidence=0.88,
                    uncertainty_score=0.31,
                    novelty_score=0.69,
                    feature_completeness=0.86,
                    policy_flags=("volume_shift", "new_beneficiary"),
                    missing_fields=("vendor_attestation",),
                    evidence_text=(
                        "Quarter-end enterprise payout batch originates from a new treasury "
                        "subnet but matches an approved internal schedule."
                    ),
                    true_case_class="legitimate",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="api_spike",
                    impact_score=95,
                    risk_score=0.44,
                    anomaly_score=0.93,
                    history_risk_score=0.37,
                    model_recommendation="accept",
                    model_confidence=0.91,
                    uncertainty_score=0.28,
                    novelty_score=0.84,
                    feature_completeness=0.88,
                    policy_flags=("schema_shift", "api_burst", "credential_reset"),
                    missing_fields=("caller_identity",),
                    evidence_text=(
                        "Credential-stuffing traffic reuses approved partner headers while probing "
                        "admin endpoints at low-and-slow cadence."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                    ood_type="adversarial_conflict",
                ),
            ),
            (
                4,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="synthetic_identity",
                    impact_score=89,
                    risk_score=0.58,
                    anomaly_score=0.81,
                    history_risk_score=0.42,
                    model_recommendation="accept",
                    model_confidence=0.86,
                    uncertainty_score=0.41,
                    novelty_score=0.77,
                    feature_completeness=0.83,
                    policy_flags=("identity_fragmentation", "velocity_spike"),
                    missing_fields=("prior_history",),
                    evidence_text=(
                        "Fresh account clears onboarding, then chains small successful payments "
                        "before a large instant transfer."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                    ood_type="feature_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="model_config_change",
                    impact_score=68,
                    risk_score=0.63,
                    anomaly_score=0.66,
                    history_risk_score=0.29,
                    model_recommendation="review",
                    model_confidence=0.62,
                    uncertainty_score=0.81,
                    novelty_score=0.71,
                    feature_completeness=0.73,
                    policy_flags=("schema_shift", "rare_pattern"),
                    missing_fields=("deployment_context", "caller_identity"),
                    evidence_text=(
                        "A config rollout coincides with unusual API behavior, but the anomaly "
                        "overlaps a documented maintenance window."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
            (
                5,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="post_submission",
                    impact_score=47,
                    risk_score=0.74,
                    anomaly_score=0.49,
                    history_risk_score=0.19,
                    model_recommendation="reject",
                    model_confidence=0.84,
                    uncertainty_score=0.38,
                    novelty_score=0.62,
                    feature_completeness=0.81,
                    policy_flags=("policy_keyword", "watchlist_hit"),
                    missing_fields=("conversation_history",),
                    evidence_text=(
                        "User quotes reclaimed slurs inside educational commentary, and the "
                        "semantic model overweights the term frequency."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                    ood_type="adversarial_conflict",
                ),
            ),
            (
                5,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="live_stream",
                    impact_score=88,
                    risk_score=0.66,
                    anomaly_score=0.71,
                    history_risk_score=0.61,
                    model_recommendation="review",
                    model_confidence=0.72,
                    uncertainty_score=0.67,
                    novelty_score=0.75,
                    feature_completeness=0.77,
                    policy_flags=("watchlist_hit", "coordination_signal"),
                    missing_fields=("prior_history", "content_context"),
                    evidence_text=(
                        "Seemingly mild livestream captions align with a coordinated harassment "
                        "campaign detected across linked accounts."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                    ood_type="context_shift",
                ),
            ),
            (
                5,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="login_burst",
                    impact_score=82,
                    risk_score=0.52,
                    anomaly_score=0.89,
                    history_risk_score=0.57,
                    model_recommendation="review",
                    model_confidence=0.65,
                    uncertainty_score=0.78,
                    novelty_score=0.82,
                    feature_completeness=0.76,
                    policy_flags=("credential_stuffing", "rare_pattern", "geo_mismatch"),
                    missing_fields=("device_graph",),
                    evidence_text=(
                        "Distributed login spray rotates across residential IPs and common user "
                        "agents to avoid normal threshold triggers."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                    ood_type="adversarial_conflict",
                ),
            ),
        ),
    )

    return {
        "easy": easy,
        "medium": medium,
        "hard": hard,
    }
