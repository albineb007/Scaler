"""Startup Pivot Agent OpenEnv client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    MetricBundle,
    StartupPivotAction,
    StartupPivotObservation,
    StartupPivotState,
)


class StartupPivotEnv(EnvClient[StartupPivotAction, StartupPivotObservation, StartupPivotState]):
    """Client for the Startup Pivot Agent environment."""

    def _step_payload(self, action: StartupPivotAction) -> Dict:
        """Convert StartupPivotAction to a JSON payload for step calls."""
        return {
            "action_type": action.action_type.value,
            "parameter": action.parameter,
            "rationale": action.rationale,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[StartupPivotObservation]:
        """Parse server response payload into a typed StepResult."""
        obs_data = payload.get("observation", {})
        metrics = MetricBundle.from_dict(obs_data.get("metrics", {}))

        observation = StartupPivotObservation(
            current_startup_idea=obs_data.get("current_startup_idea", ""),
            metrics=metrics,
            feedback_summary=obs_data.get("feedback_summary", ""),
            task_id=obs_data.get("task_id", ""),
            step_index=obs_data.get("step_index", 0),
            max_steps=obs_data.get("max_steps", 1),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> StartupPivotState:
        """Parse server state payload into StartupPivotState."""
        metrics = MetricBundle.from_dict(payload.get("metrics", {}))
        initial_metrics = MetricBundle.from_dict(payload.get("initial_metrics", {}))

        return StartupPivotState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "task_easy"),
            difficulty=payload.get("difficulty", "easy"),
            initial_idea=payload.get("initial_idea", ""),
            current_startup_idea=payload.get("current_startup_idea", ""),
            problem_statement=payload.get("problem_statement", ""),
            target_audience=payload.get("target_audience", ""),
            pricing_model=payload.get("pricing_model", ""),
            features=list(payload.get("features", [])),
            metrics=metrics,
            initial_metrics=initial_metrics,
            feedback_summary=payload.get("feedback_summary", ""),
            expected_improvement=payload.get("expected_improvement", ""),
            max_steps=payload.get("max_steps", 8),
            done=payload.get("done", False),
            final_score=payload.get("final_score"),
            action_history=list(payload.get("action_history", [])),
        )
