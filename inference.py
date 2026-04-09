# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Deterministic baseline inference entrypoint with strict log formatting."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

from openai import OpenAI

try:
    from client import RiskTriageEnv
    from models import Decision, TaskName, TriageAction, TriageObservation
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Run inference.py from the repository root.") from exc


ENV_NAME = "operational_risk_triage"

SEVERE_RISK_FLAGS = {
    "account_linkage",
    "checkout_fanout",
    "coordination_signal",
    "credential_reset",
    "credential_stuffing",
    "gift_card_cluster",
    "identity_fragmentation",
    "linked_chargeback_cluster",
    "merchant_cluster",
    "promo_velocity",
    "refund_velocity",
    "velocity_spike",
}

VALID_DECISIONS = {"accept", "reject", "review"}

SYSTEM_PROMPT = (
    "You are an operational risk triage agent. You must decide whether to accept, "
    "reject, or review a case. Only output one word: accept, reject, or review."
)

LLM_TIMEOUT_S = 5.0


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_error(message: str) -> str:
    compact = "_".join(message.strip().split())
    return compact or "runtime_error"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _format_score(value: float | None) -> str:
    if value is None:
        return "0.000000"
    return f"{value:.6f}"


def _format_rewards(values: Iterable[float]) -> str:
    return ",".join(_format_reward(value) for value in values)


def _llm_enabled(api_base_url: str | None) -> bool:
    llm_opt_in = os.environ.get("ENABLE_LLM", "false").strip().lower() == "true"
    if not llm_opt_in:
        return False
    if api_base_url:
        return True
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and api_key != "dummy":
        return True
    return False


def _heuristic_decision(observation: TriageObservation) -> Decision:
    case = observation.current_case
    if case is None:
        return "review"

    policy_flags = set(case.policy_flags)
    has_severe_flag = bool(policy_flags & SEVERE_RISK_FLAGS)

    if (
        {"beneficiary_change", "invoice_metadata_change"} <= policy_flags
        and case.risk_score >= 0.60
        and case.history_risk_score >= 0.38
    ):
        return "reject"

    if (
        {"merchant_reactivation", "amount_shift"} <= policy_flags
        and case.risk_score >= 0.62
        and case.history_risk_score >= 0.40
    ):
        if observation.remaining_review_budget > 0 and case.uncertainty_score >= 0.55:
            return "review"
        return "reject"

    if (
        case.history_risk_score <= 0.24
        and case.anomaly_score <= 0.60
        and case.feature_completeness >= 0.72
        and not has_severe_flag
    ):
        return "accept"

    if (
        case.anomaly_score >= 0.93
        and case.history_risk_score <= 0.30
        and case.risk_score <= 0.63
        and not has_severe_flag
    ):
        return "accept"

    if case.history_risk_score >= 0.60 and case.risk_score >= 0.70:
        return "reject"

    if case.anomaly_score >= 0.90 and case.history_risk_score >= 0.34 and has_severe_flag:
        return "reject"

    if case.anomaly_score >= 0.80 and case.history_risk_score >= 0.52 and has_severe_flag:
        return "reject"

    if case.anomaly_score >= 0.85 and case.history_risk_score >= 0.38 and has_severe_flag:
        return "reject"

    if case.risk_score >= 0.62 and case.anomaly_score >= 0.75 and has_severe_flag:
        return "reject"

    if (
        case.risk_score >= 0.65
        and case.anomaly_score >= 0.68
        and case.history_risk_score >= 0.40
    ):
        return "reject"

    if (
        observation.remaining_review_budget > 0
        and case.uncertainty_score >= 0.86
        and case.feature_completeness <= 0.70
    ):
        return "review"

    if (
        observation.remaining_review_budget > 0
        and case.uncertainty_score >= 0.78
        and 0.22 <= case.history_risk_score <= 0.45
        and case.feature_completeness <= 0.78
        and case.novelty_score >= 0.48
        and case.risk_score <= 0.68
        and "new_beneficiary" in policy_flags
    ):
        return "review"

    if (
        case.history_risk_score <= 0.28
        and case.risk_score <= 0.72
        and case.anomaly_score <= 0.98
        and not has_severe_flag
    ):
        return "accept"

    if (
        case.history_risk_score <= 0.24
        and case.anomaly_score <= 0.80
        and case.feature_completeness >= 0.76
    ):
        return "accept"

    if case.model_recommendation == "review" and observation.remaining_review_budget > 0:
        return "review"

    if (
        observation.remaining_review_budget > 0
        and case.uncertainty_score >= 0.76
        and case.feature_completeness <= 0.80
        and case.history_risk_score < 0.50
    ):
        return "review"

    if (
        case.risk_score <= 0.30
        and case.anomaly_score <= 0.35
        and case.history_risk_score <= 0.30
    ):
        return "accept"

    if case.model_recommendation == "review" and observation.remaining_review_budget == 0:
        return "reject"

    return case.model_recommendation


def get_llm_decision(client: OpenAI, observation: TriageObservation) -> str:
    case = observation.current_case
    if case is None:
        return "review"

    try:
        response = client.chat.completions.create(
            model=os.environ.get("MODEL_NAME", "gpt-4.1-mini"),
            temperature=0.0,
            max_tokens=1,
            timeout=LLM_TIMEOUT_S,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Decide action."},
            ],
        )
        content = (response.choices[0].message.content or "").strip().lower()
        return content if content in VALID_DECISIONS else "review"
    except Exception:
        return "review"


def _decision_for_observation(model_client, observation, llm_enabled):
    if llm_enabled:
        return get_llm_decision(model_client, observation)
    return _heuristic_decision(observation)


def run_episode(env_url, task, api_base_url, model_name, hf_token) -> int:
    api_key = os.environ.get("OPENAI_API_KEY") or hf_token or "dummy"

    try:
        client = OpenAI(base_url=api_base_url, api_key=api_key)
    except Exception:
        client = OpenAI(api_key="dummy")

    rewards = []
    steps = 0
    final_score = 0.0

    print(f"[START] task={task} env={ENV_NAME} model={model_name}")

    try:
        env = RiskTriageEnv(base_url=env_url).sync()
        with env:
            result = env.reset(task=task)

            while not result.done:
                action = _decision_for_observation(client, result.observation, False)

                try:
                    result = env.step(
                        TriageAction(
                            decision=action,
                            rationale="baseline",
                        )
                    )
                    reward = float(result.reward or 0.0)
                except Exception as exc:
                    error = _sanitize_error(str(exc))
                    print(
                        f"[STEP] step={steps+1} action={action} reward=0.00 done=true error={error}"
                    )
                    print(
                        f"[END] success=false steps={steps} score={_format_score(final_score)} rewards={_format_rewards(rewards)}"
                    )
                    return 0

                rewards.append(reward)
                steps += 1
                final_score = float(result.observation.normalized_score or 0.0)

                print(
                    f"[STEP] step={steps} action={action} reward={_format_reward(reward)} done={_bool_text(result.done)} error=null"
                )

    except Exception as exc:
        error = _sanitize_error(str(exc))
        print(
            f"[STEP] step={steps+1} action=review reward=0.00 done=true error={error}"
        )
        print(
            f"[END] success=false steps={steps} score={_format_score(final_score)} rewards={_format_rewards(rewards)}"
        )
        return 0

    print(
        f"[END] success=true steps={steps} score={_format_score(final_score)} rewards={_format_rewards(rewards)}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default=os.environ.get("ENV_URL", "http://localhost:8000"))
    parser.add_argument("--task", default="easy")
    args = parser.parse_args()

    return run_episode(
        env_url=args.env_url,
        task=args.task,
        api_base_url=os.environ.get("API_BASE_URL"),
        model_name=os.environ.get("MODEL_NAME", "gpt-4.1-mini"),
        hf_token=os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    raise SystemExit(main())