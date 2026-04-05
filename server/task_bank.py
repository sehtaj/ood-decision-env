# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic task definitions and validation for operational risk triage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from random import Random

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
TASK_CASE_COUNTS = {
    "easy": 20,
    "medium": 25,
    "hard": 30,
}
TASK_REVIEW_BUDGETS = {
    "easy": 4,
    "medium": 4,
    "hard": 3,
}
TASK_SEEDS = {
    "easy": 104729,
    "medium": 130363,
    "hard": 155921,
}
TASK_VARIANT_NOTES = {
    "easy": (
        "The queue marks the pattern as familiar and well-covered by standard controls.",
        "Linked telemetry stays consistent across device, billing, and prior account history.",
        "An analyst would expect the automated signals to be broadly reliable on this pattern.",
    ),
    "medium": (
        "Context arriving from upstream systems is slightly stale and only partially reconciled.",
        "Recent product changes mean some learned signals may lag behind the operational reality.",
        "Rule-based alerts and model outputs disagree in a way that normally triggers analyst caution.",
        "The record contains enough evidence to decide, but not enough to trust any single score blindly.",
    ),
    "hard": (
        "This pattern sits near a known distribution boundary where model calibration has degraded before.",
        "Cross-system evidence is internally inconsistent, forcing a tradeoff between false blocks and costly misses.",
        "Operational notes suggest part of the anomaly may be benign, but the hidden cost of a miss is still high.",
        "The strongest automated signal here is precisely the one most likely to break under shift.",
    ),
}
TASK_EXTRA_MISSING = {
    "easy": (),
    "medium": ("device_graph", "caller_identity", "prior_history", "content_context"),
    "hard": (
        "device_graph",
        "deployment_context",
        "vendor_attestation",
        "conversation_history",
        "regional_baseline",
    ),
}
TASK_PROFILES = {
    "easy": {
        "jitter_choices": (-0.012, -0.006, 0.0, 0.008, 0.012),
        "ood_bonus": 0.0,
        "extra_missing_probability": 0.0,
        "confidence_drift": 0.18,
    },
    "medium": {
        "jitter_choices": (-0.022, -0.014, -0.006, 0.0, 0.014, 0.022),
        "ood_bonus": 0.04,
        "extra_missing_probability": 0.45,
        "confidence_drift": 0.26,
    },
    "hard": {
        "jitter_choices": (-0.032, -0.024, -0.012, 0.0, 0.016, 0.028, 0.036),
        "ood_bonus": 0.07,
        "extra_missing_probability": 0.72,
        "confidence_drift": 0.36,
    },
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
    seed: int
    review_budget: int
    cases: tuple[TaskCase, ...]


@dataclass(frozen=True)
class TaskBankValidationReport:
    """Compact summary proving the deterministic bank passes Stage 4 checks."""

    fingerprint: str
    policy_scores: dict[TaskName, dict[str, float]]
    easy_simple_baseline_score: float
    average_novelty: dict[TaskName, float]
    average_completeness: dict[TaskName, float]
    average_ood_ratio: dict[TaskName, float]


def _stable_rng(task_name: TaskName, task_seed: int, case_index: int, variant_index: int) -> Random:
    return Random(f"{task_name}:{task_seed}:{case_index}:{variant_index}")


def _build_case(
    task_name: TaskName,
    task_seed: int,
    case_index: int,
    variant_index: int,
    blueprint: CaseBlueprint,
) -> TaskCase:
    rng = _stable_rng(task_name, task_seed, case_index, variant_index)
    profile = TASK_PROFILES[task_name]
    strength = rng.choice(profile["jitter_choices"])
    ood_bonus = profile["ood_bonus"] if blueprint.ood_type != "none" else 0.0
    review_bias = 0.04 if blueprint.true_case_class == "ambiguous" else 0.0
    note = rng.choice(TASK_VARIANT_NOTES[task_name])

    extra_missing = ()
    if TASK_EXTRA_MISSING[task_name] and rng.random() < profile["extra_missing_probability"]:
        extra_missing = (rng.choice(TASK_EXTRA_MISSING[task_name]),)

    confidence_shift = -(abs(strength) * profile["confidence_drift"])
    model_is_wrong = blueprint.model_recommendation != _optimal_decision(blueprint.true_case_class)
    if blueprint.ood_type != "none" and task_name == "hard" and model_is_wrong:
        confidence_shift = abs(strength) * 0.45 + 0.03
    elif blueprint.ood_type != "none" and model_is_wrong:
        confidence_shift += 0.04

    action_values = _action_values(blueprint.true_case_class, blueprint.cost_tier)
    policy_flags = tuple(dict.fromkeys(blueprint.policy_flags))
    missing_fields = tuple(dict.fromkeys(blueprint.missing_fields + extra_missing))
    evidence_text = f"{blueprint.evidence_text} {note}"

    return TaskCase(
        case_id=f"{task_name}-{case_index:03d}",
        domain_hint=blueprint.domain_hint,
        event_type=blueprint.event_type,
        impact_score=blueprint.impact_score,
        risk_score=_clamp(blueprint.risk_score + strength),
        anomaly_score=_clamp(blueprint.anomaly_score + abs(strength) * 0.7 + ood_bonus + review_bias),
        history_risk_score=_clamp(blueprint.history_risk_score + strength * 0.55),
        model_recommendation=blueprint.model_recommendation,
        model_confidence=_clamp(blueprint.model_confidence + confidence_shift),
        uncertainty_score=_clamp(
            blueprint.uncertainty_score + abs(strength) * 0.9 + ood_bonus + review_bias
        ),
        novelty_score=_clamp(blueprint.novelty_score + abs(strength) * 0.75 + ood_bonus),
        feature_completeness=_clamp(
            blueprint.feature_completeness
            - abs(strength) * 0.8
            - (0.04 if extra_missing else 0.0)
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
    blueprint_counts: tuple[tuple[int, CaseBlueprint], ...],
) -> TaskDefinition:
    cases: list[TaskCase] = []
    case_index = 1
    task_seed = TASK_SEEDS[task_name]
    for repeat_count, blueprint in blueprint_counts:
        for variant_index in range(repeat_count):
            cases.append(_build_case(task_name, task_seed, case_index, variant_index, blueprint))
            case_index += 1

    if len(cases) != TASK_CASE_COUNTS[task_name]:
        raise ValueError(
            f"Task '{task_name}' generated {len(cases)} cases, expected {TASK_CASE_COUNTS[task_name]}."
        )

    return TaskDefinition(
        name=task_name,
        seed=task_seed,
        review_budget=TASK_REVIEW_BUDGETS[task_name],
        cases=tuple(cases),
    )


def _build_task_bank_definitions() -> dict[TaskName, TaskDefinition]:
    easy = _expand_task(
        "easy",
        blueprint_counts=(
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="card_payment",
                    impact_score=34,
                    risk_score=0.12,
                    anomaly_score=0.11,
                    history_risk_score=0.16,
                    model_recommendation="accept",
                    model_confidence=0.94,
                    uncertainty_score=0.06,
                    novelty_score=0.08,
                    feature_completeness=0.98,
                    policy_flags=("known_customer",),
                    missing_fields=(),
                    evidence_text=(
                        "Returning customer places a routine domestic order from a familiar device, "
                        "shipping pattern, and billing region."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="subscription_renewal",
                    impact_score=26,
                    risk_score=0.19,
                    anomaly_score=0.14,
                    history_risk_score=0.11,
                    model_recommendation="accept",
                    model_confidence=0.88,
                    uncertainty_score=0.08,
                    novelty_score=0.09,
                    feature_completeness=0.97,
                    policy_flags=("known_merchant",),
                    missing_fields=(),
                    evidence_text=(
                        "Recurring subscription renews on the expected cadence with matching merchant, "
                        "device, and payment instrument history."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="subscription_renewal",
                    impact_score=31,
                    risk_score=0.33,
                    anomaly_score=0.29,
                    history_risk_score=0.18,
                    model_recommendation="review",
                    model_confidence=0.67,
                    uncertainty_score=0.43,
                    novelty_score=0.18,
                    feature_completeness=0.95,
                    policy_flags=("known_merchant", "recent_plan_change"),
                    missing_fields=(),
                    evidence_text=(
                        "Subscription resumes after a legitimate plan upgrade from a corporate travel "
                        "network, which briefly makes the account look less stable than it really is."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="payroll_payout",
                    impact_score=43,
                    risk_score=0.24,
                    anomaly_score=0.18,
                    history_risk_score=0.13,
                    model_recommendation="accept",
                    model_confidence=0.86,
                    uncertainty_score=0.12,
                    novelty_score=0.12,
                    feature_completeness=0.95,
                    policy_flags=("approved_beneficiary",),
                    missing_fields=(),
                    evidence_text=(
                        "Scheduled payroll transfer goes to a long-standing beneficiary with a clean "
                        "history and no linked dispute activity."
                    ),
                    true_case_class="legitimate",
                    cost_tier="sensitive",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="account_takeover",
                    impact_score=93,
                    risk_score=0.92,
                    anomaly_score=0.88,
                    history_risk_score=0.83,
                    model_recommendation="reject",
                    model_confidence=0.95,
                    uncertainty_score=0.1,
                    novelty_score=0.16,
                    feature_completeness=0.97,
                    policy_flags=("velocity_spike", "device_mismatch", "credential_reset"),
                    missing_fields=(),
                    evidence_text=(
                        "High-value wallet cash-out follows a password reset and login from a new "
                        "device with no prior trusted-session history."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="mule_transfer",
                    impact_score=81,
                    risk_score=0.84,
                    anomaly_score=0.8,
                    history_risk_score=0.76,
                    model_recommendation="reject",
                    model_confidence=0.9,
                    uncertainty_score=0.16,
                    novelty_score=0.22,
                    feature_completeness=0.95,
                    policy_flags=("velocity_spike", "linked_chargeback_cluster"),
                    missing_fields=(),
                    evidence_text=(
                        "Fresh beneficiary receives rapid transfers immediately after low-value probes "
                        "used to test the linked funding instrument."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="gift_card_burst",
                    impact_score=74,
                    risk_score=0.79,
                    anomaly_score=0.73,
                    history_risk_score=0.71,
                    model_recommendation="reject",
                    model_confidence=0.87,
                    uncertainty_score=0.19,
                    novelty_score=0.2,
                    feature_completeness=0.94,
                    policy_flags=("gift_card_cluster", "checkout_fanout"),
                    missing_fields=(),
                    evidence_text=(
                        "Multiple digital gift card purchases spread across merchants within minutes "
                        "after an unsuccessful address update attempt."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="invoice_redirect",
                    impact_score=76,
                    risk_score=0.62,
                    anomaly_score=0.58,
                    history_risk_score=0.39,
                    model_recommendation="accept",
                    model_confidence=0.71,
                    uncertainty_score=0.34,
                    novelty_score=0.19,
                    feature_completeness=0.92,
                    policy_flags=("beneficiary_change", "invoice_metadata_change"),
                    missing_fields=(),
                    evidence_text=(
                        "Vendor payment request updates beneficiary details shortly before cutoff, and "
                        "the accompanying invoice metadata no longer matches the historical template."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                ),
            ),
            (
                2,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="travel_booking",
                    impact_score=61,
                    risk_score=0.55,
                    anomaly_score=0.58,
                    history_risk_score=0.46,
                    model_recommendation="review",
                    model_confidence=0.56,
                    uncertainty_score=0.69,
                    novelty_score=0.31,
                    feature_completeness=0.86,
                    policy_flags=("geo_mismatch", "device_mismatch"),
                    missing_fields=("prior_history",),
                    evidence_text=(
                        "Cross-border booking originates from a newly seen device, but itinerary, "
                        "billing details, and loyalty account behavior remain internally consistent."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="merchant_reactivation",
                    impact_score=57,
                    risk_score=0.59,
                    anomaly_score=0.54,
                    history_risk_score=0.42,
                    model_recommendation="review",
                    model_confidence=0.61,
                    uncertainty_score=0.64,
                    novelty_score=0.28,
                    feature_completeness=0.84,
                    policy_flags=("merchant_reactivation", "amount_shift"),
                    missing_fields=("merchant_history",),
                    evidence_text=(
                        "Dormant merchant relationship reappears with a larger amount, but the account "
                        "also shows a legitimate service-renewal trail."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="merchant_reactivation",
                    impact_score=57,
                    risk_score=0.63,
                    anomaly_score=0.57,
                    history_risk_score=0.43,
                    model_recommendation="accept",
                    model_confidence=0.69,
                    uncertainty_score=0.53,
                    novelty_score=0.21,
                    feature_completeness=0.85,
                    policy_flags=("merchant_reactivation", "amount_shift"),
                    missing_fields=("merchant_history",),
                    evidence_text=(
                        "Dormant merchant relationship reappears with a larger amount and incomplete "
                        "supporting history, leaving the payment plausible but not safe to auto-approve."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                ),
            ),
        ),
    )

    medium = _expand_task(
        "medium",
        blueprint_counts=(
            (
                3,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="post_submission",
                    impact_score=44,
                    risk_score=0.66,
                    anomaly_score=0.37,
                    history_risk_score=0.19,
                    model_recommendation="review",
                    model_confidence=0.64,
                    uncertainty_score=0.48,
                    novelty_score=0.34,
                    feature_completeness=0.88,
                    policy_flags=("policy_keyword",),
                    missing_fields=("content_context",),
                    evidence_text=(
                        "News repost quotes prohibited language in an educational framing, and prior "
                        "appeals for the account were resolved in the user's favor."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
            (
                2,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="travel_booking",
                    impact_score=58,
                    risk_score=0.47,
                    anomaly_score=0.57,
                    history_risk_score=0.26,
                    model_recommendation="review",
                    model_confidence=0.57,
                    uncertainty_score=0.63,
                    novelty_score=0.4,
                    feature_completeness=0.84,
                    policy_flags=("geo_mismatch", "travel_pattern_change"),
                    missing_fields=("device_id",),
                    evidence_text=(
                        "Established customer books an international trip from a new region shortly "
                        "after confirming the itinerary on a previously trusted device."
                    ),
                    true_case_class="legitimate",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="batch_replay",
                    impact_score=63,
                    risk_score=0.51,
                    anomaly_score=0.73,
                    history_risk_score=0.24,
                    model_recommendation="reject",
                    model_confidence=0.71,
                    uncertainty_score=0.52,
                    novelty_score=0.46,
                    feature_completeness=0.82,
                    policy_flags=("volume_shift", "rare_pattern"),
                    missing_fields=("deployment_context",),
                    evidence_text=(
                        "Scheduled vendor replay backfills delayed settlement events in a way that "
                        "resembles abuse, but the maintenance ticket aligns with the spike."
                    ),
                    true_case_class="legitimate",
                    cost_tier="sensitive",
                    ood_type="context_shift",
                ),
            ),
            (
                2,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="seller_message",
                    impact_score=41,
                    risk_score=0.58,
                    anomaly_score=0.43,
                    history_risk_score=0.21,
                    model_recommendation="accept",
                    model_confidence=0.69,
                    uncertainty_score=0.44,
                    novelty_score=0.36,
                    feature_completeness=0.86,
                    policy_flags=("policy_keyword",),
                    missing_fields=("conversation_history",),
                    evidence_text=(
                        "Marketplace seller references a restricted term while discussing safety "
                        "remediation steps with a buyer after a damaged shipment."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="checkout_fanout",
                    impact_score=91,
                    risk_score=0.88,
                    anomaly_score=0.78,
                    history_risk_score=0.69,
                    model_recommendation="reject",
                    model_confidence=0.83,
                    uncertainty_score=0.24,
                    novelty_score=0.29,
                    feature_completeness=0.92,
                    policy_flags=("velocity_spike", "billing_mismatch"),
                    missing_fields=(),
                    evidence_text=(
                        "High-value checkouts fan out across merchants immediately after a new device "
                        "enrollment and failed billing verification attempt."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="direct_message",
                    impact_score=73,
                    risk_score=0.74,
                    anomaly_score=0.57,
                    history_risk_score=0.66,
                    model_recommendation="accept",
                    model_confidence=0.78,
                    uncertainty_score=0.38,
                    novelty_score=0.47,
                    feature_completeness=0.84,
                    policy_flags=("watchlist_hit", "policy_keyword"),
                    missing_fields=("conversation_history",),
                    evidence_text=(
                        "Previously sanctioned account sends short coded abuse phrases that the "
                        "language model still rates as low severity."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                    ood_type="adversarial_conflict",
                ),
            ),
            (
                2,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="promo_abuse",
                    impact_score=67,
                    risk_score=0.72,
                    anomaly_score=0.66,
                    history_risk_score=0.59,
                    model_recommendation="review",
                    model_confidence=0.61,
                    uncertainty_score=0.57,
                    novelty_score=0.41,
                    feature_completeness=0.8,
                    policy_flags=("account_linkage", "promo_velocity"),
                    missing_fields=("prior_history",),
                    evidence_text=(
                        "Cluster of newly created accounts redeems one-time credits from the same "
                        "household graph before cashing out a resaleable item."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="api_spike",
                    impact_score=86,
                    risk_score=0.68,
                    anomaly_score=0.9,
                    history_risk_score=0.43,
                    model_recommendation="review",
                    model_confidence=0.56,
                    uncertainty_score=0.76,
                    novelty_score=0.59,
                    feature_completeness=0.81,
                    policy_flags=("schema_shift", "api_burst"),
                    missing_fields=("caller_identity",),
                    evidence_text=(
                        "Traffic surge hits rarely used admin endpoints with a new payload shape and "
                        "partner headers that only partially match the allowlist."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                    ood_type="feature_shift",
                ),
            ),
            (
                2,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="post_submission",
                    impact_score=54,
                    risk_score=0.62,
                    anomaly_score=0.53,
                    history_risk_score=0.41,
                    model_recommendation="reject",
                    model_confidence=0.65,
                    uncertainty_score=0.71,
                    novelty_score=0.44,
                    feature_completeness=0.79,
                    policy_flags=("policy_keyword", "rapid_editing"),
                    missing_fields=("content_context", "prior_history"),
                    evidence_text=(
                        "Sarcastic post quotes prohibited language, removes surrounding context through "
                        "rapid edits, and reaches an audience already primed for escalation."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
            (
                2,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="model_config_change",
                    impact_score=69,
                    risk_score=0.59,
                    anomaly_score=0.68,
                    history_risk_score=0.31,
                    model_recommendation="review",
                    model_confidence=0.6,
                    uncertainty_score=0.75,
                    novelty_score=0.49,
                    feature_completeness=0.77,
                    policy_flags=("schema_shift", "rare_pattern"),
                    missing_fields=("deployment_context", "regional_baseline"),
                    evidence_text=(
                        "A model rollout and a suspicious traffic shift land in the same maintenance "
                        "window, leaving the system unable to cleanly attribute the anomaly."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
        ),
    )

    hard = _expand_task(
        "hard",
        blueprint_counts=(
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="enterprise_payout",
                    impact_score=77,
                    risk_score=0.7,
                    anomaly_score=0.84,
                    history_risk_score=0.18,
                    model_recommendation="reject",
                    model_confidence=0.87,
                    uncertainty_score=0.34,
                    novelty_score=0.71,
                    feature_completeness=0.85,
                    policy_flags=("volume_shift", "new_beneficiary"),
                    missing_fields=("vendor_attestation",),
                    evidence_text=(
                        "Quarter-end enterprise payout batch originates from a new treasury subnet but "
                        "matches an approved internal schedule and beneficiary ledger."
                    ),
                    true_case_class="legitimate",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="post_submission",
                    impact_score=49,
                    risk_score=0.75,
                    anomaly_score=0.5,
                    history_risk_score=0.22,
                    model_recommendation="reject",
                    model_confidence=0.83,
                    uncertainty_score=0.39,
                    novelty_score=0.63,
                    feature_completeness=0.8,
                    policy_flags=("policy_keyword", "watchlist_hit"),
                    missing_fields=("conversation_history",),
                    evidence_text=(
                        "User quotes reclaimed slurs in a clearly educational thread, but the semantic "
                        "model overweights term frequency and prior enforcement history."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                    ood_type="adversarial_conflict",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="partner_api_flood",
                    impact_score=71,
                    risk_score=0.69,
                    anomaly_score=0.88,
                    history_risk_score=0.27,
                    model_recommendation="reject",
                    model_confidence=0.79,
                    uncertainty_score=0.48,
                    novelty_score=0.74,
                    feature_completeness=0.81,
                    policy_flags=("schema_shift", "api_burst", "allowlist_match"),
                    missing_fields=("partner_change_ticket",),
                    evidence_text=(
                        "Approved partner backfills delayed traffic after a routing failover, producing "
                        "an abnormal burst that mimics credential stuffing telemetry."
                    ),
                    true_case_class="legitimate",
                    cost_tier="sensitive",
                    ood_type="context_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="incident_response_login_burst",
                    impact_score=66,
                    risk_score=0.61,
                    anomaly_score=0.86,
                    history_risk_score=0.29,
                    model_recommendation="review",
                    model_confidence=0.68,
                    uncertainty_score=0.7,
                    novelty_score=0.69,
                    feature_completeness=0.78,
                    policy_flags=("geo_mismatch", "burst_login", "rare_pattern"),
                    missing_fields=("regional_baseline",),
                    evidence_text=(
                        "Distributed logins originate from responders during a live incident bridge, "
                        "creating a suspicious pattern that nonetheless aligns with the approved runbook."
                    ),
                    true_case_class="legitimate",
                    cost_tier="standard",
                    ood_type="feature_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="api_spike",
                    impact_score=95,
                    risk_score=0.46,
                    anomaly_score=0.93,
                    history_risk_score=0.36,
                    model_recommendation="accept",
                    model_confidence=0.9,
                    uncertainty_score=0.32,
                    novelty_score=0.85,
                    feature_completeness=0.87,
                    policy_flags=("schema_shift", "api_burst", "credential_reset"),
                    missing_fields=("caller_identity",),
                    evidence_text=(
                        "Credential-stuffing traffic reuses approved partner headers while probing admin "
                        "endpoints slowly enough to avoid standard threshold alarms."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                    ood_type="adversarial_conflict",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="synthetic_identity",
                    impact_score=89,
                    risk_score=0.57,
                    anomaly_score=0.82,
                    history_risk_score=0.41,
                    model_recommendation="accept",
                    model_confidence=0.85,
                    uncertainty_score=0.43,
                    novelty_score=0.79,
                    feature_completeness=0.82,
                    policy_flags=("identity_fragmentation", "velocity_spike"),
                    missing_fields=("prior_history",),
                    evidence_text=(
                        "Fresh account clears onboarding, chains low-risk purchases, then pivots into a "
                        "large instant transfer that outstrips the verified profile."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                    ood_type="feature_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="live_stream",
                    impact_score=88,
                    risk_score=0.67,
                    anomaly_score=0.72,
                    history_risk_score=0.62,
                    model_recommendation="review",
                    model_confidence=0.73,
                    uncertainty_score=0.68,
                    novelty_score=0.76,
                    feature_completeness=0.77,
                    policy_flags=("watchlist_hit", "coordination_signal"),
                    missing_fields=("prior_history", "content_context"),
                    evidence_text=(
                        "Seemingly mild livestream captions align with a coordinated harassment campaign "
                        "detected across linked accounts and off-platform rallying channels."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                    ood_type="context_shift",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="login_burst",
                    impact_score=83,
                    risk_score=0.53,
                    anomaly_score=0.9,
                    history_risk_score=0.56,
                    model_recommendation="review",
                    model_confidence=0.66,
                    uncertainty_score=0.79,
                    novelty_score=0.83,
                    feature_completeness=0.75,
                    policy_flags=("credential_stuffing", "rare_pattern", "geo_mismatch"),
                    missing_fields=("device_graph",),
                    evidence_text=(
                        "Distributed login spray rotates across residential IPs and common user agents "
                        "to evade the normal credential-stuffing signatures."
                    ),
                    true_case_class="harmful",
                    cost_tier="critical",
                    ood_type="adversarial_conflict",
                ),
            ),
            (
                3,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="refund_loop",
                    impact_score=79,
                    risk_score=0.65,
                    anomaly_score=0.75,
                    history_risk_score=0.58,
                    model_recommendation="review",
                    model_confidence=0.7,
                    uncertainty_score=0.66,
                    novelty_score=0.72,
                    feature_completeness=0.79,
                    policy_flags=("refund_velocity", "merchant_cluster", "linked_accounts"),
                    missing_fields=("device_graph",),
                    evidence_text=(
                        "Shared household graph repeatedly purchases and refunds resaleable items through "
                        "new merchants, then rotates payment instruments before settlement completes."
                    ),
                    true_case_class="harmful",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="system",
                    event_type="model_config_change",
                    impact_score=68,
                    risk_score=0.62,
                    anomaly_score=0.66,
                    history_risk_score=0.29,
                    model_recommendation="review",
                    model_confidence=0.63,
                    uncertainty_score=0.82,
                    novelty_score=0.72,
                    feature_completeness=0.73,
                    policy_flags=("schema_shift", "rare_pattern"),
                    missing_fields=("deployment_context", "caller_identity"),
                    evidence_text=(
                        "A safety model rollout coincides with unusual API behavior, and the available "
                        "telemetry cannot cleanly separate attack traffic from deployment side effects."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
                    ood_type="context_shift",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="payment",
                    event_type="charity_drive_spike",
                    impact_score=72,
                    risk_score=0.58,
                    anomaly_score=0.71,
                    history_risk_score=0.25,
                    model_recommendation="reject",
                    model_confidence=0.76,
                    uncertainty_score=0.74,
                    novelty_score=0.78,
                    feature_completeness=0.74,
                    policy_flags=("geo_mismatch", "velocity_spike", "new_beneficiary"),
                    missing_fields=("regional_baseline", "vendor_attestation"),
                    evidence_text=(
                        "Large donation burst flows through a newly onboarded nonprofit aggregator during "
                        "a crisis response, but fraud and legitimate urgency are both plausible explanations."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="sensitive",
                    ood_type="feature_shift",
                ),
            ),
            (
                1,
                CaseBlueprint(
                    domain_hint="content",
                    event_type="breaking_news_live_stream",
                    impact_score=64,
                    risk_score=0.6,
                    anomaly_score=0.64,
                    history_risk_score=0.33,
                    model_recommendation="accept",
                    model_confidence=0.74,
                    uncertainty_score=0.77,
                    novelty_score=0.75,
                    feature_completeness=0.72,
                    policy_flags=("policy_keyword", "watchlist_hit", "coordination_signal"),
                    missing_fields=("content_context", "conversation_history"),
                    evidence_text=(
                        "Live breaking-news commentary echoes violent slogans quoted from the event itself, "
                        "making the stream look simultaneously risky, newsworthy, and highly time-sensitive."
                    ),
                    true_case_class="ambiguous",
                    cost_tier="standard",
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


def _policy_score(definition: TaskDefinition, decision: Decision) -> float:
    return round(sum(case.action_value(decision) for case in definition.cases), 3)


def _model_policy_score(definition: TaskDefinition) -> float:
    return round(sum(case.action_value(case.model_recommendation) for case in definition.cases), 3)


def _simple_baseline_decision(case: TaskCase) -> Decision:
    high_risk_flags = {
        "velocity_spike",
        "credential_reset",
        "gift_card_cluster",
        "checkout_fanout",
        "identity_fragmentation",
        "linked_chargeback_cluster",
    }
    if (
        case.risk_score >= 0.79
        or case.anomaly_score >= 0.82
        or any(flag in high_risk_flags for flag in case.policy_flags)
    ):
        return "reject"
    if case.model_recommendation == "review":
        return "review"
    if case.uncertainty_score >= 0.66 and case.novelty_score >= 0.24:
        return "review"
    if (
        case.risk_score <= 0.3
        and case.anomaly_score <= 0.35
        and case.history_risk_score <= 0.3
    ):
        return "accept"
    return case.model_recommendation


def _simple_baseline_score(definition: TaskDefinition) -> float:
    return round(sum(case.action_value(_simple_baseline_decision(case)) for case in definition.cases), 3)


def _task_bank_fingerprint(task_bank: dict[TaskName, TaskDefinition]) -> str:
    serializable = {
        task_name: {
            "seed": definition.seed,
            "review_budget": definition.review_budget,
            "cases": [asdict(case) for case in definition.cases],
        }
        for task_name, definition in task_bank.items()
    }
    encoded = json.dumps(serializable, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def validate_task_bank(task_bank: dict[TaskName, TaskDefinition]) -> TaskBankValidationReport:
    """Validate Stage 4 realism, determinism, and anti-shortcut properties."""

    policy_scores: dict[TaskName, dict[str, float]] = {}
    average_novelty: dict[TaskName, float] = {}
    average_completeness: dict[TaskName, float] = {}
    average_ood_ratio: dict[TaskName, float] = {}

    for task_name, definition in task_bank.items():
        if definition.seed != TASK_SEEDS[task_name]:
            raise ValueError(f"Task '{task_name}' seed drifted from the fixed Stage 4 seed.")
        if definition.review_budget != TASK_REVIEW_BUDGETS[task_name]:
            raise ValueError(f"Task '{task_name}' review budget drifted from the locked Stage 0 value.")
        if len(definition.cases) != TASK_CASE_COUNTS[task_name]:
            raise ValueError(f"Task '{task_name}' generated the wrong number of cases.")

        seen_case_ids: set[str] = set()
        optimal_review_count = 0
        total_novelty = 0.0
        total_completeness = 0.0
        total_ood = 0

        for case in definition.cases:
            if case.case_id in seen_case_ids:
                raise ValueError(f"Duplicate case id detected: {case.case_id}")
            seen_case_ids.add(case.case_id)

            if case.optimal_decision != _optimal_decision(case.true_case_class):
                raise ValueError(f"Case {case.case_id} has an inconsistent optimal decision.")
            if _action_values(case.true_case_class, case.cost_tier) != {
                "accept": case.action_value_accept,
                "reject": case.action_value_reject,
                "review": case.action_value_review,
            }:
                raise ValueError(f"Case {case.case_id} has an inconsistent action-value table.")
            if len(case.evidence_text) < 80:
                raise ValueError(f"Case {case.case_id} evidence text is too thin for review realism.")
            if "optimal decision" in case.evidence_text.lower() or "ground truth" in case.evidence_text.lower():
                raise ValueError(f"Case {case.case_id} leaks grading language into visible evidence.")
            if case.false_accept_cost <= case.false_reject_cost:
                raise ValueError(f"Case {case.case_id} violates cost asymmetry.")
            if case.review_cost <= 0:
                raise ValueError(f"Case {case.case_id} review cost must remain positive.")

            if case.optimal_decision == "review":
                optimal_review_count += 1
            total_novelty += case.novelty_score
            total_completeness += case.feature_completeness
            total_ood += int(case.is_ood)

        if optimal_review_count < definition.review_budget:
            raise ValueError(
                f"Task '{task_name}' has review budget {definition.review_budget} but only "
                f"{optimal_review_count} review-optimal cases."
            )

        policy_scores[task_name] = {
            "always_accept": _policy_score(definition, "accept"),
            "always_reject": _policy_score(definition, "reject"),
            "always_review": _policy_score(definition, "review"),
            "follow_model": _model_policy_score(definition),
            "simple_baseline": _simple_baseline_score(definition),
        }
        if policy_scores[task_name]["always_accept"] >= 0:
            raise ValueError(f"Task '{task_name}' still rewards always-accept.")
        if policy_scores[task_name]["always_reject"] >= 0:
            raise ValueError(f"Task '{task_name}' still rewards always-reject.")
        if policy_scores[task_name]["always_review"] >= 0:
            raise ValueError(f"Task '{task_name}' still rewards always-review.")

        average_novelty[task_name] = round(total_novelty / len(definition.cases), 3)
        average_completeness[task_name] = round(total_completeness / len(definition.cases), 3)
        average_ood_ratio[task_name] = round(total_ood / len(definition.cases), 3)

    if not (
        average_novelty["easy"] < average_novelty["medium"] < average_novelty["hard"]
    ):
        raise ValueError("Novelty should increase from easy to medium to hard.")
    if not (
        average_completeness["easy"] > average_completeness["medium"] > average_completeness["hard"]
    ):
        raise ValueError("Feature completeness should decrease from easy to medium to hard.")
    if not (
        average_ood_ratio["easy"] < average_ood_ratio["medium"] < average_ood_ratio["hard"]
    ):
        raise ValueError("OOD ratio should increase from easy to medium to hard.")

    easy_optimal = round(
        sum(case.action_value(case.optimal_decision) for case in task_bank["easy"].cases),
        3,
    )
    easy_simple_baseline_score = policy_scores["easy"]["simple_baseline"]
    if easy_simple_baseline_score <= easy_optimal * 0.45:
        raise ValueError("Easy-task simple baseline should achieve a non-trivial score.")
    if easy_simple_baseline_score >= easy_optimal:
        raise ValueError("Easy-task simple baseline should remain imperfect, not optimal.")

    fingerprint = _task_bank_fingerprint(task_bank)
    repeated_fingerprint = _task_bank_fingerprint(_build_task_bank_definitions())
    if fingerprint != repeated_fingerprint:
        raise ValueError("Task bank fingerprint changed across repeated deterministic builds.")

    return TaskBankValidationReport(
        fingerprint=fingerprint,
        policy_scores=policy_scores,
        easy_simple_baseline_score=easy_simple_baseline_score,
        average_novelty=average_novelty,
        average_completeness=average_completeness,
        average_ood_ratio=average_ood_ratio,
    )


def build_task_bank(*, validate: bool = True) -> dict[TaskName, TaskDefinition]:
    """Create the deterministic task bank used by the environment runtime."""

    task_bank = _build_task_bank_definitions()
    if validate:
        validate_task_bank(task_bank)
    return task_bank
