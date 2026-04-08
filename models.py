"""Typed models for the Startup Pivot Agent OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


class MetricBundle(BaseModel):
    """Five startup quality metrics scored from 0 to 10."""

    market_demand: float = Field(..., ge=0.0, le=10.0)
    feasibility: float = Field(..., ge=0.0, le=10.0)
    scalability: float = Field(..., ge=0.0, le=10.0)
    clarity: float = Field(..., ge=0.0, le=10.0)
    novelty: float = Field(..., ge=0.0, le=10.0)

    def mean(self) -> float:
        return (
            self.market_demand
            + self.feasibility
            + self.scalability
            + self.clarity
            + self.novelty
        ) / 5.0

    def as_dict(self) -> dict[str, float]:
        return {
            "market_demand": self.market_demand,
            "feasibility": self.feasibility,
            "scalability": self.scalability,
            "clarity": self.clarity,
            "novelty": self.novelty,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "MetricBundle":
        return cls(
            market_demand=float(values.get("market_demand", 0.0)),
            feasibility=float(values.get("feasibility", 0.0)),
            scalability=float(values.get("scalability", 0.0)),
            clarity=float(values.get("clarity", 0.0)),
            novelty=float(values.get("novelty", 0.0)),
        )


class ActionType(str, Enum):
    """Supported environment actions."""

    refine_market = "refine_market"
    reduce_scope = "reduce_scope"
    add_feature = "add_feature"
    remove_feature = "remove_feature"
    adjust_pricing = "adjust_pricing"
    pivot_problem = "pivot_problem"


class StartupPivotAction(Action):
    """Structured action applied to the startup idea."""

    action_type: ActionType = Field(..., description="Type of startup refinement action")
    parameter: str = Field(
        default="",
        max_length=240,
        description="Action-specific payload (audience, feature, pricing, etc.)",
    )
    rationale: str = Field(
        default="",
        max_length=320,
        description="Optional agent rationale for debugging",
    )


class StartupPivotObservation(Observation):
    """Observation emitted after reset or step."""

    current_startup_idea: str = Field(
        ..., description="Current startup idea text after refinements"
    )
    metrics: MetricBundle = Field(..., description="Current startup quality metrics")
    feedback_summary: str = Field(
        default="",
        description="Deterministic textual feedback summarizing latest change",
    )
    task_id: str = Field(..., description="Current task identifier")
    step_index: int = Field(default=0, ge=0, description="Current step index")
    max_steps: int = Field(default=8, ge=1, description="Episode step limit")


class GradingProfile(BaseModel):
    """Weights for final deterministic grading."""

    average_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    clarity_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    feasibility_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    viability_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    complexity_penalty_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_positive_weights(self) -> "GradingProfile":
        total_positive = (
            self.average_weight
            + self.clarity_weight
            + self.feasibility_weight
            + self.viability_weight
        )
        if total_positive <= 0.0:
            raise ValueError("At least one positive grading weight must be non-zero.")
        return self


class TaskSpec(BaseModel):
    """Single deterministic startup-refinement task."""

    task_id: str = Field(..., min_length=3)
    difficulty: str = Field(..., description="easy | medium | hard")
    input_idea: str = Field(..., min_length=10)
    problem_statement: str = Field(..., min_length=10)
    target_audience: str = Field(..., min_length=3)
    pricing_model: str = Field(..., min_length=3)
    features: List[str] = Field(default_factory=list)
    initial_metrics: MetricBundle
    expected_improvement: str = Field(..., min_length=10)
    max_steps: int = Field(default=8, ge=3, le=20)
    grading_profile: GradingProfile = Field(default_factory=GradingProfile)


class StartupPivotState(State):
    """Internal environment state."""

    task_id: str = Field(default="task_easy")
    difficulty: str = Field(default="easy")
    initial_idea: str = Field(default="")
    current_startup_idea: str = Field(default="")
    problem_statement: str = Field(default="")
    target_audience: str = Field(default="")
    pricing_model: str = Field(default="")
    features: List[str] = Field(default_factory=list)
    metrics: MetricBundle = Field(
        default_factory=lambda: MetricBundle(
            market_demand=5.0,
            feasibility=5.0,
            scalability=5.0,
            clarity=5.0,
            novelty=5.0,
        )
    )
    initial_metrics: MetricBundle = Field(
        default_factory=lambda: MetricBundle(
            market_demand=5.0,
            feasibility=5.0,
            scalability=5.0,
            clarity=5.0,
            novelty=5.0,
        )
    )
    feedback_summary: str = Field(default="")
    expected_improvement: str = Field(default="")
    max_steps: int = Field(default=8, ge=1)
    done: bool = Field(default=False)
    final_score: float | None = Field(default=None, ge=0.0, le=1.0)
    action_history: List[str] = Field(default_factory=list)
