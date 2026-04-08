"""Startup Pivot Agent environment implementation."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..graders import calculate_step_reward, grade_final_score, summarize_feedback
    from ..models import (
        ActionType,
        MetricBundle,
        StartupPivotAction,
        StartupPivotObservation,
        StartupPivotState,
        TaskSpec,
    )
    from ..tasks import load_tasks
except ImportError:
    from graders import calculate_step_reward, grade_final_score, summarize_feedback
    from models import (
        ActionType,
        MetricBundle,
        StartupPivotAction,
        StartupPivotObservation,
        StartupPivotState,
        TaskSpec,
    )
    from tasks import load_tasks


class StartupPivotEnvironment(
    Environment[StartupPivotAction, StartupPivotObservation, StartupPivotState]
):
    """Environment where an agent improves startup ideas through deterministic pivots."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._tasks = load_tasks()
        if not self._tasks:
            raise ValueError("No task definitions found for StartupPivotEnvironment.")

        first_task_id = sorted(self._tasks.keys())[0]
        self._active_task = self._tasks[first_task_id]
        self._initial_feature_count = len(self._active_task.features)
        self._state = StartupPivotState(episode_id=str(uuid4()))
        self._initialize_from_task(self._active_task, episode_id=str(uuid4()))

    @staticmethod
    def _clip_metric(value: float) -> float:
        return max(0.0, min(10.0, round(value, 4)))

    @staticmethod
    def _compose_idea(
        input_idea: str,
        problem_statement: str,
        target_audience: str,
        features: list[str],
        pricing_model: str,
    ) -> str:
        feature_text = ", ".join(features) if features else "no selected features"
        return (
            f"Problem: {problem_statement}. "
            f"Target audience: {target_audience}. "
            f"Current concept: {input_idea}. "
            f"Core features: {feature_text}. "
            f"Pricing: {pricing_model}."
        )

    def _initialize_from_task(self, task: TaskSpec, episode_id: str | None = None) -> None:
        self._initial_feature_count = len(task.features)
        base_metrics = task.initial_metrics.model_copy(deep=True)

        self._state = StartupPivotState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            difficulty=task.difficulty,
            initial_idea=task.input_idea,
            current_startup_idea=self._compose_idea(
                task.input_idea,
                task.problem_statement,
                task.target_audience,
                list(task.features),
                task.pricing_model,
            ),
            problem_statement=task.problem_statement,
            target_audience=task.target_audience,
            pricing_model=task.pricing_model,
            features=list(task.features),
            metrics=base_metrics,
            initial_metrics=base_metrics.model_copy(deep=True),
            feedback_summary=f"Baseline loaded. Goal: {task.expected_improvement}",
            expected_improvement=task.expected_improvement,
            max_steps=task.max_steps,
            done=False,
            final_score=None,
            action_history=[],
        )

    def _apply_metric_adjustments(self, adjustments: dict[str, float]) -> None:
        values = self._state.metrics.as_dict()
        for key, delta in adjustments.items():
            values[key] = self._clip_metric(values[key] + delta)
        self._state.metrics = MetricBundle.from_dict(values)

    def _pop_feature(self, parameter: str) -> str:
        if not self._state.features:
            return ""
        needle = parameter.strip().lower()
        if needle:
            for idx, feature in enumerate(self._state.features):
                if needle in feature.lower():
                    return self._state.features.pop(idx)
        return self._state.features.pop()

    def _apply_action(self, action: StartupPivotAction) -> str:
        parameter = action.parameter.strip()
        action_type = action.action_type

        note = ""
        adjustments: dict[str, float] = {
            "market_demand": 0.0,
            "feasibility": 0.0,
            "scalability": 0.0,
            "clarity": 0.0,
            "novelty": 0.0,
        }

        if action_type == ActionType.refine_market:
            if parameter:
                self._state.target_audience = parameter
            elif "everyone" in self._state.target_audience.lower():
                self._state.target_audience = "independent service businesses in tier-2 cities"
            note = f"Refined market focus to {self._state.target_audience}."
            adjustments.update(
                {
                    "market_demand": 1.1,
                    "feasibility": 0.4,
                    "scalability": 0.2,
                    "clarity": 1.0,
                    "novelty": -0.1,
                }
            )

        elif action_type == ActionType.reduce_scope:
            removed = self._pop_feature(parameter)
            note = (
                f"Reduced scope by removing '{removed}'."
                if removed
                else "Attempted scope reduction but no feature was available."
            )
            adjustments.update(
                {
                    "market_demand": 0.2,
                    "feasibility": 1.1,
                    "scalability": 0.6,
                    "clarity": 0.9,
                    "novelty": -0.2,
                }
            )

        elif action_type == ActionType.add_feature:
            new_feature = parameter or f"feature_{self._state.step_count}"
            if new_feature.lower() not in [item.lower() for item in self._state.features]:
                self._state.features.append(new_feature)
            note = f"Added feature '{new_feature}'."
            adjustments.update(
                {
                    "market_demand": 0.3,
                    "feasibility": -0.55,
                    "scalability": 0.2,
                    "clarity": -0.35,
                    "novelty": 0.7,
                }
            )
            if len(self._state.features) > 5:
                adjustments["feasibility"] -= 0.6
                adjustments["clarity"] -= 0.4
                adjustments["scalability"] -= 0.3

        elif action_type == ActionType.remove_feature:
            removed = self._pop_feature(parameter)
            note = (
                f"Removed feature '{removed}'."
                if removed
                else "Attempted to remove a feature, but none existed."
            )
            adjustments.update(
                {
                    "market_demand": -0.1,
                    "feasibility": 0.8,
                    "scalability": 0.4,
                    "clarity": 0.6,
                    "novelty": -0.2,
                }
            )

        elif action_type == ActionType.adjust_pricing:
            if parameter:
                self._state.pricing_model = parameter
            lower_pricing = self._state.pricing_model.lower()
            note = f"Adjusted pricing to '{self._state.pricing_model}'."

            if "subscription" in lower_pricing or "monthly" in lower_pricing:
                adjustments.update(
                    {
                        "market_demand": 0.5,
                        "feasibility": 0.6,
                        "scalability": 0.5,
                        "clarity": 0.5,
                        "novelty": 0.1,
                    }
                )
            elif "free" in lower_pricing:
                adjustments.update(
                    {
                        "market_demand": 0.8,
                        "feasibility": -0.4,
                        "scalability": -0.3,
                        "clarity": 0.2,
                        "novelty": 0.0,
                    }
                )
            else:
                adjustments.update(
                    {
                        "market_demand": 0.2,
                        "feasibility": 0.3,
                        "scalability": 0.2,
                        "clarity": 0.3,
                        "novelty": 0.0,
                    }
                )

        elif action_type == ActionType.pivot_problem:
            if parameter:
                self._state.problem_statement = parameter
            lower_problem = self._state.problem_statement.lower()
            note = f"Pivoted problem statement to '{self._state.problem_statement}'."

            unrealistic_tokens = (
                "everyone",
                "all industries",
                "instantly",
                "overnight",
                "100x",
            )
            if any(token in lower_problem for token in unrealistic_tokens):
                adjustments.update(
                    {
                        "market_demand": -0.6,
                        "feasibility": -1.3,
                        "scalability": -0.5,
                        "clarity": -0.9,
                        "novelty": 0.2,
                    }
                )
            else:
                adjustments.update(
                    {
                        "market_demand": 0.9,
                        "feasibility": 0.3,
                        "scalability": 0.4,
                        "clarity": 0.8,
                        "novelty": 1.1,
                    }
                )

        self._apply_metric_adjustments(adjustments)
        self._state.current_startup_idea = self._compose_idea(
            self._state.initial_idea,
            self._state.problem_statement,
            self._state.target_audience,
            self._state.features,
            self._state.pricing_model,
        )
        return note

    def _build_observation(self, reward: float, done: bool) -> StartupPivotObservation:
        return StartupPivotObservation(
            current_startup_idea=self._state.current_startup_idea,
            metrics=self._state.metrics.model_copy(deep=True),
            feedback_summary=self._state.feedback_summary,
            task_id=self._state.task_id,
            step_index=self._state.step_count,
            max_steps=self._state.max_steps,
            done=done,
            reward=reward,
            metadata={
                "feature_count": len(self._state.features),
                "features": list(self._state.features),
                "pricing_model": self._state.pricing_model,
                "difficulty": self._state.difficulty,
            },
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> StartupPivotObservation:
        del seed

        requested_task = task_id or kwargs.get("task_id")
        options = kwargs.get("options")
        if requested_task is None and isinstance(options, dict):
            requested_task = options.get("task_id")

        if requested_task is None:
            requested_task = self._active_task.task_id

        if requested_task not in self._tasks:
            raise ValueError(f"Unknown task_id '{requested_task}'.")

        self._active_task = self._tasks[requested_task]
        self._initialize_from_task(self._active_task, episode_id=episode_id or str(uuid4()))
        return self._build_observation(reward=0.0, done=False)

    async def reset_async(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> StartupPivotObservation:
        return self.reset(seed=seed, episode_id=episode_id, task_id=task_id, **kwargs)

    def step(
        self,
        action: StartupPivotAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> StartupPivotObservation:
        del timeout_s, kwargs
        if self._state.done:
            return self._build_observation(reward=0.0, done=True)

        self._state.step_count += 1
        previous_metrics = self._state.metrics.model_copy(deep=True)

        action_note = self._apply_action(action)
        self._state.action_history.append(f"{action.action_type.value}:{action.parameter.strip()}")

        reward = calculate_step_reward(
            previous_metrics,
            self._state.metrics,
            len(self._state.features),
            action.action_type,
        )

        self._state.done = self._state.step_count >= self._state.max_steps
        self._state.feedback_summary = summarize_feedback(
            previous_metrics,
            self._state.metrics,
            action_note,
        )

        if self._state.done:
            self._state.final_score = grade_final_score(
                self._active_task,
                self._state.initial_metrics,
                self._state.metrics,
                self._initial_feature_count,
                len(self._state.features),
            )
            self._state.feedback_summary = (
                f"{self._state.feedback_summary} "
                f"Episode complete with score {self._state.final_score:.4f}."
            )

        return self._build_observation(reward=reward, done=self._state.done)

    async def step_async(
        self,
        action: StartupPivotAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> StartupPivotObservation:
        return self.step(action=action, timeout_s=timeout_s, **kwargs)

    async def state_async(self) -> StartupPivotState:
        """Async accessor mirroring the synchronous state property."""
        return self._state

    @property
    def state(self) -> StartupPivotState:
        return self._state
