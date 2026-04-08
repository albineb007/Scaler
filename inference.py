"""Inference runner for the Startup Pivot Agent environment."""

from __future__ import annotations

import asyncio
import os
from typing import Tuple

from openai import OpenAI

try:
    from .models import ActionType, StartupPivotAction, StartupPivotObservation
    from .server.my_env_environment import StartupPivotEnvironment
    from .tasks import ordered_tasks
except ImportError:
    from models import ActionType, StartupPivotAction, StartupPivotObservation
    from server.my_env_environment import StartupPivotEnvironment
    from tasks import ordered_tasks

ENV_NAME = "startup_pivot_agent"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")


class DeterministicPivotPolicy:
    """Deterministic policy with optional OpenAI-generated rationale text."""

    def __init__(self, api_base_url: str, model_name: str, token: str):
        self.model_name = model_name
        self.token = token.strip()
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=self.token or "hf_missing_token",
        )

    @staticmethod
    def _select_action(observation: StartupPivotObservation) -> Tuple[ActionType, str]:
        metrics = observation.metrics
        features = list(observation.metadata.get("features", []))
        feature_count = int(observation.metadata.get("feature_count", len(features)))

        if metrics.clarity < 6.0:
            return (
                ActionType.refine_market,
                "independent local service businesses with repeat customers",
            )
        if metrics.feasibility < 6.0 and feature_count >= 5:
            fallback = features[-1] if features else "non-core analytics module"
            return (ActionType.reduce_scope, fallback)
        if metrics.feasibility < 6.5:
            return (
                ActionType.adjust_pricing,
                "tiered monthly subscription with pilot onboarding",
            )
        if metrics.market_demand < 6.5:
            return (
                ActionType.pivot_problem,
                "help small businesses recover revenue lost from manual follow-up",
            )
        if metrics.scalability < 6.5:
            fallback = features[-1] if features else "custom one-off integrations"
            return (ActionType.remove_feature, fallback)
        if metrics.novelty < 6.0:
            return (ActionType.add_feature, "lightweight referral loop")
        return (
            ActionType.refine_market,
            "multi-location independent operators in one vertical",
        )

    def _build_rationale(
        self,
        observation: StartupPivotObservation,
        action_type: ActionType,
        parameter: str,
    ) -> str:
        if not self.token:
            return "Deterministic heuristic policy."

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                max_tokens=40,
                messages=[
                    {
                        "role": "system",
                        "content": "Give one short rationale sentence for a startup pivot action.",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Metrics={observation.metrics.model_dump()} "
                            f"Action={action_type.value} "
                            f"Parameter={parameter}"
                        ),
                    },
                ],
            )
            content = completion.choices[0].message.content or ""
            return " ".join(content.strip().split())[:300]
        except Exception:
            return "LLM rationale unavailable; deterministic policy used."

    def next_action(self, observation: StartupPivotObservation) -> StartupPivotAction:
        action_type, parameter = self._select_action(observation)
        rationale = self._build_rationale(observation, action_type, parameter)
        return StartupPivotAction(
            action_type=action_type,
            parameter=parameter,
            rationale=rationale,
        )


async def run_task(
    env: StartupPivotEnvironment,
    policy: DeterministicPivotPolicy,
    task_id: str,
) -> float:
    observation = await env.reset_async(task_id=task_id)
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    rewards: list[float] = []
    success = False

    for step_idx in range(1, observation.max_steps + 1):
        error = "none"
        action_name = "none"

        try:
            action = policy.next_action(observation)
            action_name = action.action_type.value
            observation = await env.step_async(action)
            reward = float(observation.reward or 0.0)
            done = bool(observation.done)
        except Exception as exc:
            reward = 0.0
            done = True
            error = " ".join(str(exc).split())[:120]

        rewards.append(round(reward, 4))
        print(
            f"[STEP] step={step_idx} action={action_name} "
            f"reward={reward:.4f} done={str(done).lower()} error={error}"
        )

        if done:
            break

    state = env.state
    score = float(state.final_score or 0.0)
    success = score >= 0.55
    rewards_text = ",".join(f"{value:.4f}" for value in rewards)

    print(
        f"[END] success={str(success).lower()} steps={state.step_count} "
        f"score={score:.4f} rewards={rewards_text}"
    )
    return score


async def async_main() -> None:
    env = StartupPivotEnvironment()
    policy = DeterministicPivotPolicy(API_BASE_URL, MODEL_NAME, HF_TOKEN)

    for task in ordered_tasks():
        await run_task(env, policy, task.task_id)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
