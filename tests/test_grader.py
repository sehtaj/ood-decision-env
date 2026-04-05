"""Deterministic grading tests for the operational risk triage environment."""

from __future__ import annotations

from server.grader import (
    REVIEW_EXHAUSTION_PENALTY,
    aggregate_task_scores,
    grade_episode,
    grade_step,
    normalize_raw_score,
    score_bounds_for_task,
)
from server.task_bank import build_task_bank


TASK_BANK = build_task_bank()


def _first_case(task_name: str, true_case_class: str):
    return next(case for case in TASK_BANK[task_name].cases if case.true_case_class == true_case_class)


def test_accept_scoring_on_legitimate_and_harmful_cases() -> None:
    legitimate_case = _first_case("easy", "legitimate")
    harmful_case = _first_case("easy", "harmful")

    assert legitimate_case.action_value("accept") == 3.0
    assert harmful_case.action_value("accept") == -harmful_case.false_accept_cost


def test_reject_scoring_on_harmful_and_legitimate_cases() -> None:
    harmful_case = _first_case("easy", "harmful")
    legitimate_case = _first_case("easy", "legitimate")

    assert harmful_case.action_value("reject") == 4.0
    assert legitimate_case.action_value("reject") == -legitimate_case.false_reject_cost


def test_review_scoring_on_ambiguous_and_obvious_cases() -> None:
    ambiguous_case = _first_case("easy", "ambiguous")
    legitimate_case = _first_case("easy", "legitimate")

    assert ambiguous_case.action_value("review") == 2.0
    assert legitimate_case.action_value("review") == -legitimate_case.review_cost


def test_review_budget_exhaustion_penalty() -> None:
    definition = TASK_BANK["hard"]
    ambiguous_index = next(
        index for index, case in enumerate(definition.cases) if case.true_case_class == "ambiguous"
    )

    step_grade = grade_step(
        definition=definition,
        case_index=ambiguous_index,
        requested_decision="review",
        remaining_review_budget=0,
        cumulative_raw_score_before=0.0,
    )

    assert step_grade.outcome_category == "review_budget_exhausted"
    assert step_grade.review_honored is False
    assert step_grade.raw_reward == REVIEW_EXHAUSTION_PENALTY - definition.cases[ambiguous_index].review_cost


def test_invalid_actions_take_a_deterministic_penalty_path() -> None:
    definition = TASK_BANK["easy"]
    step_grade = grade_step(
        definition=definition,
        case_index=0,
        requested_decision="escalate",
        remaining_review_budget=definition.review_budget,
        cumulative_raw_score_before=0.0,
    )

    assert step_grade.outcome_category == "invalid_action"
    assert step_grade.applied_decision is None
    assert step_grade.raw_reward < min(
        definition.cases[0].action_value_accept,
        definition.cases[0].action_value_reject,
        definition.cases[0].action_value_review,
    )


def test_normalization_bounds_stay_inside_zero_and_one() -> None:
    bounds = score_bounds_for_task(TASK_BANK["medium"])

    assert normalize_raw_score(bounds.min_raw_score - 50.0, bounds) == 0.0
    assert normalize_raw_score(bounds.max_raw_score + 50.0, bounds) == 1.0


def test_golden_easy_optimal_episode() -> None:
    definition = TASK_BANK["easy"]
    actions = [case.optimal_decision for case in definition.cases]
    grade = grade_episode(definition, actions)

    assert grade.raw_score == 64.0
    assert grade.normalized_score == 1.0


def test_golden_medium_follow_model_episode() -> None:
    definition = TASK_BANK["medium"]
    actions = [case.model_recommendation for case in definition.cases]
    grade = grade_episode(definition, actions)

    assert grade.raw_score == -86.0
    assert grade.normalized_score == 0.0


def test_golden_hard_simple_baseline_episode() -> None:
    definition = TASK_BANK["hard"]
    actions = []
    for case in definition.cases:
        if (
            case.risk_score >= 0.79
            or case.anomaly_score >= 0.82
            or any(
                flag in {
                    "velocity_spike",
                    "credential_reset",
                    "gift_card_cluster",
                    "checkout_fanout",
                    "identity_fragmentation",
                    "linked_chargeback_cluster",
                }
                for flag in case.policy_flags
            )
        ):
            actions.append("reject")
        elif case.model_recommendation == "review":
            actions.append("review")
        elif case.uncertainty_score >= 0.66 and case.novelty_score >= 0.24:
            actions.append("review")
        elif (
            case.risk_score <= 0.3
            and case.anomaly_score <= 0.35
            and case.history_risk_score <= 0.3
        ):
            actions.append("accept")
        else:
            actions.append(case.model_recommendation)

    grade = grade_episode(definition, actions)

    assert grade.raw_score == -22.0
    assert grade.normalized_score == 0.0


def test_aggregate_task_scores_uses_equal_weight_mean() -> None:
    easy = grade_episode(
        TASK_BANK["easy"],
        [case.model_recommendation for case in TASK_BANK["easy"].cases],
    )
    medium = grade_episode(
        TASK_BANK["medium"],
        [case.optimal_decision for case in TASK_BANK["medium"].cases],
    )
    hard = grade_episode(TASK_BANK["hard"], ["reject" for _ in TASK_BANK["hard"].cases])

    assert aggregate_task_scores((easy, medium, hard)) == 0.510417
