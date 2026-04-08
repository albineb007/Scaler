"""Deterministic reward shaping and final grading utilities."""

from __future__ import annotations

try:
    from .models import ActionType, MetricBundle, TaskSpec
except ImportError:
    from models import ActionType, MetricBundle, TaskSpec

METRIC_KEYS = (
    "market_demand",
    "feasibility",
    "scalability",
    "clarity",
    "novelty",
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def metric_deltas(previous: MetricBundle, current: MetricBundle) -> dict[str, float]:
    """Compute per-metric deltas between two metric bundles."""
    return {
        key: round(getattr(current, key) - getattr(previous, key), 4)
        for key in METRIC_KEYS
    }


def summarize_feedback(
    previous: MetricBundle,
    current: MetricBundle,
    action_note: str,
) -> str:
    """Create deterministic human-readable feedback from metric changes."""
    deltas = metric_deltas(previous, current)
    sorted_by_abs = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)

    strongest_key, strongest_val = sorted_by_abs[0]
    weakest_key, weakest_val = sorted_by_abs[-1]

    return (
        f"{action_note} Strongest shift: {strongest_key} {strongest_val:+.2f}. "
        f"Weakest shift: {weakest_key} {weakest_val:+.2f}."
    )


def calculate_step_reward(
    previous: MetricBundle,
    current: MetricBundle,
    feature_count: int,
    action_type: ActionType,
) -> float:
    """Dense deterministic reward based on metric improvements and complexity penalties."""
    deltas = metric_deltas(previous, current)
    total_delta = sum(deltas.values())
    average_delta = total_delta / len(METRIC_KEYS)

    reward = average_delta * 0.20
    reward += max(0.0, deltas["clarity"]) * 0.10
    reward += max(0.0, deltas["feasibility"]) * 0.09

    # Overly large feature sets reduce viability and training signal quality.
    if feature_count > 5:
        reward -= (feature_count - 5) * 0.08

    if action_type == ActionType.add_feature and feature_count >= 5:
        reward -= 0.08

    if total_delta <= 0.0:
        reward -= 0.12

    return round(_clamp(reward, -1.0, 1.0), 4)


def grade_final_score(
    task: TaskSpec,
    initial_metrics: MetricBundle,
    final_metrics: MetricBundle,
    initial_feature_count: int,
    final_feature_count: int,
) -> float:
    """Calculate a deterministic normalized task score in [0.0, 1.0]."""
    profile = task.grading_profile

    average_gain = _clamp((final_metrics.mean() - initial_metrics.mean()) / 10.0, 0.0, 1.0)
    clarity_gain = _clamp(
        (final_metrics.clarity - initial_metrics.clarity) / 10.0,
        0.0,
        1.0,
    )
    feasibility_gain = _clamp(
        (final_metrics.feasibility - initial_metrics.feasibility) / 10.0,
        0.0,
        1.0,
    )
    final_quality = _clamp(final_metrics.mean() / 10.0, 0.0, 1.0)

    viability_gate = 1.0 if min(
        final_metrics.market_demand,
        final_metrics.feasibility,
        final_metrics.clarity,
    ) >= 6.0 else 0.0

    complexity_growth = max(0, final_feature_count - (initial_feature_count + 2))
    complexity_penalty = _clamp(complexity_growth / 4.0, 0.0, 1.0)

    raw_score = (
        0.35 * final_quality
        + profile.average_weight * average_gain
        + profile.clarity_weight * clarity_gain
        + profile.feasibility_weight * feasibility_gain
        + profile.viability_weight * viability_gate
        - profile.complexity_penalty_weight * complexity_penalty
    )

    return round(_clamp(raw_score, 0.0, 1.0), 4)
