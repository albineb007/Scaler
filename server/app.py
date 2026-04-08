"""FastAPI application for the Startup Pivot Agent OpenEnv environment."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import gradio as gr
import pandas as pd
from openai import OpenAI

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server import web_interface as openenv_web_interface
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import StartupPivotAction, StartupPivotObservation
    from ..tasks import ordered_tasks
    from .my_env_environment import StartupPivotEnvironment
except ImportError:
    from models import StartupPivotAction, StartupPivotObservation
    from tasks import ordered_tasks
    from server.my_env_environment import StartupPivotEnvironment


ACTION_CHOICES = [
    "refine_market",
    "reduce_scope",
    "add_feature",
    "remove_feature",
    "adjust_pricing",
    "pivot_problem",
]
METRIC_KEYS = ["market_demand", "feasibility", "scalability", "clarity", "novelty"]
METRIC_LABELS = {
    "market_demand": "Market Demand",
    "feasibility": "Feasibility",
    "scalability": "Scalability",
    "clarity": "Clarity",
    "novelty": "Novelty",
}
METRIC_COLORS = {
    "market_demand": "#4cc9f0",
    "feasibility": "#2ec4b6",
    "scalability": "#f4a261",
    "clarity": "#90be6d",
    "novelty": "#e76f51",
}
METRIC_COLOR_BY_LABEL = {
        METRIC_LABELS[key]: value for key, value in METRIC_COLORS.items()
}

CUSTOM_UI_CSS = """
footer, .footer, .gradio-footer {
  display: none !important;
}
.gradio-container {
    background: linear-gradient(180deg, #081524 0%, #0a1b2d 100%) !important;
}
.hero-card {
    border: 1px solid rgba(76, 201, 240, 0.20);
    background: #0b2034;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.hero-card h1 {
  margin: 0;
    font-size: 1.2rem;
  letter-spacing: 0.02em;
}
.hero-card p {
    margin: 6px 0 0;
    opacity: 0.92;
}
.top-ai-pane,
.control-pane,
.result-pane {
    border: 1px solid rgba(144, 190, 109, 0.20);
    border-radius: 12px;
    background: #0b2034;
}
.top-ai-pane,
.control-pane,
.result-pane {
    padding: 10px !important;
}
.compact-row {
    gap: 8px;
}
.gradio-container .gr-button {
    min-height: 38px;
}
"""

EVALUATION_HELP_MD = """
### Evaluation Metrics Used

- Market Demand: Is this problem valuable enough to solve?
- Feasibility: Can a small team realistically build and ship it?
- Scalability: Can it grow without extreme custom effort?
- Clarity: Is the pitch and positioning easy to understand?
- Novelty: Is it differentiated enough to matter?

### Scoring Signals

- Final quality (mean of 5 metrics) contributes strongly.
- Gains in average quality, clarity, and feasibility are rewarded.
- Viability gate is positive when demand, feasibility, and clarity stay healthy.
- Excessive feature bloat adds a complexity penalty.
"""

AI_SUMMARY_DEFAULT_MD = """
### AI Input Summary

- Free text summary: n/a
- Model summary: n/a
- Last suggested decision: n/a
"""

AI_STATS_DEFAULT_MD = """
### AI Stats

- Mode: n/a
- Model: n/a
- Input chars/words/tokens: n/a
- Model output tokens: n/a
- Latency: n/a
"""


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_action(payload: dict[str, Any]) -> tuple[str, str, str]:
    action = str(payload.get("action_type", "")).strip()
    if action not in ACTION_CHOICES:
        action = "refine_market"

    parameter = str(payload.get("parameter", "")).strip()[:240]
    rationale = str(payload.get("rationale", "")).strip()[:320]
    if not rationale:
        rationale = "Generated from natural-language goal."

    return action, parameter, rationale


def _heuristic_action(goal_text: str) -> tuple[str, str, str]:
    text = goal_text.lower()

    if any(token in text for token in ["price", "pricing", "subscription", "monthly"]):
        return (
            "adjust_pricing",
            "tiered monthly subscription with pilot onboarding",
            "Clarify monetization while keeping onboarding practical.",
        )
    if any(token in text for token in ["target", "audience", "market", "niche"]):
        return (
            "refine_market",
            "independent local service businesses with repeat customers",
            "Sharpen target audience to improve clarity and focus.",
        )
    if any(token in text for token in ["simpl", "scope", "reduce", "cut", "remove"]):
        return (
            "reduce_scope",
            "non-core analytics module",
            "Reduce implementation complexity to increase feasibility.",
        )
    if any(token in text for token in ["feature", "differ", "novel", "innovation"]):
        return (
            "add_feature",
            "lightweight referral loop",
            "Add one focused differentiator without overloading complexity.",
        )
    if any(token in text for token in ["problem", "pivot", "position"]):
        return (
            "pivot_problem",
            "help small businesses recover revenue lost from manual follow-up",
            "Reframe the core problem to one with immediate business value.",
        )

    return (
        "refine_market",
        "multi-location independent operators in one vertical",
        "Start by narrowing the market to increase strategic clarity.",
    )


def _summarize_free_text(text: str, max_words: int = 22) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "n/a"
    words = cleaned.split(" ")
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]) + " ..."


def _estimate_tokens(text: str) -> int:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    return max(1, int(len(cleaned) / 4))


def _single_line(text: Any) -> str:
    return " ".join(str(text or "").split()) or "n/a"


def _build_ai_action(
    goal_text: str,
    state_data: dict[str, Any],
    hf_token: str,
    model_name: str,
) -> tuple[str, str, str, str, dict[str, Any]]:
    goal = (goal_text or "").strip()
    selected_model = (
        (model_name or "").strip()
        or os.getenv("HF_MODEL_NAME", "").strip()
        or os.getenv("MODEL_NAME", "").strip()
        or "openai/gpt-4.1-mini"
    )

    base_meta = {
        "mode": "heuristic",
        "model": selected_model,
        "input_summary": _summarize_free_text(goal),
        "model_summary": "n/a",
        "input_chars": len(goal),
        "input_words": len(goal.split()) if goal else 0,
        "input_est_tokens": _estimate_tokens(goal),
        "output_tokens": None,
        "latency_ms": 0,
    }

    if not goal:
        action, parameter, rationale = _heuristic_action("")
        meta = dict(base_meta)
        meta["model_summary"] = "Goal was empty, so a safe default strategy was selected."
        return (
            action,
            parameter,
            rationale,
            "Describe your goal first to generate an action.",
            meta,
        )

    token = (hf_token or "").strip() or os.getenv("HF_TOKEN", "").strip()

    if not token:
        action, parameter, rationale = _heuristic_action(goal)
        meta = dict(base_meta)
        meta["model_summary"] = rationale
        return (
            action,
            parameter,
            rationale,
            "HF token not provided. Used local heuristic mapping.",
            meta,
        )

    prompt_state = {
        "task_id": state_data.get("task_id"),
        "step_count": state_data.get("step_count"),
        "metrics": state_data.get("metrics", {}),
        "feature_count": len(state_data.get("features", []) or []),
        "pricing_model": state_data.get("pricing_model", ""),
        "target_audience": state_data.get("target_audience", ""),
    }

    try:
        started = time.perf_counter()
        client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=token,
        )
        completion = client.chat.completions.create(
            model=selected_model,
            temperature=0,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Map user intent to a deterministic environment action. "
                        "Return only JSON with keys action_type, parameter, rationale, reasoning_summary. "
                        f"Allowed action_type values: {', '.join(ACTION_CHOICES)}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "goal": goal,
                            "current_state": prompt_state,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        content = completion.choices[0].message.content or ""
        payload = _extract_first_json_object(content)
        if payload is None:
            raise ValueError("Could not parse JSON action from model response.")

        action, parameter, rationale = _normalize_action(payload)
        model_summary = _single_line(payload.get("reasoning_summary", ""))
        if model_summary == "n/a":
            model_summary = _single_line(rationale)

        usage = getattr(completion, "usage", None)
        output_tokens = None
        if usage is not None:
            output_tokens = getattr(usage, "completion_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(usage, "total_tokens", None)

        meta = dict(base_meta)
        meta.update(
            {
                "mode": "model",
                "model_summary": model_summary,
                "output_tokens": output_tokens,
                "latency_ms": elapsed_ms,
            }
        )
        return (
            action,
            parameter,
            rationale,
            f"AI suggestion generated with {selected_model}.",
            meta,
        )
    except Exception as exc:
        action, parameter, rationale = _heuristic_action(goal)
        meta = dict(base_meta)
        meta["model_summary"] = rationale
        return (
            action,
            parameter,
            rationale,
            f"AI suggestion failed ({type(exc).__name__}). Used heuristic mapping.",
            meta,
        )


def _build_ai_result_panels(
    action: str,
    parameter: str,
    rationale: str,
    ai_meta: dict[str, Any],
) -> tuple[str, str]:
    summary_md = (
        "### AI Input Summary\n\n"
        f"- Free text summary: {_single_line(ai_meta.get('input_summary'))}\n"
        f"- Model summary: {_single_line(ai_meta.get('model_summary'))}\n"
        f"- Suggested decision: **{_single_line(action)}**\n"
        f"- Suggested details: {_single_line(parameter)}\n"
        f"- Suggested why: {_single_line(rationale)}\n"
    )

    output_tokens = ai_meta.get("output_tokens")
    token_text = "n/a" if output_tokens in (None, "") else str(output_tokens)
    stats_md = (
        "### AI Stats\n\n"
        f"- Mode: **{_single_line(ai_meta.get('mode'))}**\n"
        f"- Model: **{_single_line(ai_meta.get('model'))}**\n"
        f"- Input chars: **{int(ai_meta.get('input_chars', 0))}**\n"
        f"- Input words: **{int(ai_meta.get('input_words', 0))}**\n"
        f"- Estimated input tokens: **{int(ai_meta.get('input_est_tokens', 0))}**\n"
        f"- Model output tokens: **{token_text}**\n"
        f"- Latency: **{int(ai_meta.get('latency_ms', 0))} ms**\n"
    )

    return summary_md, stats_md


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    observation = payload.get("observation", {}) if isinstance(payload, dict) else {}
    metrics = observation.get("metrics", {}) if isinstance(observation, dict) else {}

    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _history_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["step", *METRIC_KEYS, "average"]
    if not history:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(history)
    for key in METRIC_KEYS:
        if key not in df.columns:
            df[key] = None

    df["average"] = df[METRIC_KEYS].mean(axis=1).round(3)
    return df[columns]


def _append_history(
    history: list[dict[str, Any]],
    step_index: int,
    metrics: dict[str, float],
) -> list[dict[str, Any]]:
    if not metrics:
        return history

    point = {"step": int(step_index)}
    for key in METRIC_KEYS:
        point[key] = float(metrics.get(key, 0.0))

    updated = list(history)
    if updated and int(updated[-1].get("step", -1)) == int(step_index):
        updated[-1] = point
    else:
        updated.append(point)
    return updated


def _empty_trend_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["step", "metric", "score"])


def _empty_snapshot_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [METRIC_LABELS[key] for key in METRIC_KEYS],
            "score": [0.0 for _ in METRIC_KEYS],
        }
    )


def _build_trend_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return _empty_trend_dataframe()

    frame = pd.DataFrame(history).sort_values("step")
    long_frame = frame.melt(
        id_vars=["step"],
        value_vars=METRIC_KEYS,
        var_name="metric",
        value_name="score",
    )
    long_frame["metric"] = long_frame["metric"].map(METRIC_LABELS)
    return long_frame[["step", "metric", "score"]]


def _build_snapshot_dataframe(history: list[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        return _empty_snapshot_dataframe()

    latest = pd.DataFrame(history).sort_values("step").iloc[-1]
    return pd.DataFrame(
        {
            "metric": [METRIC_LABELS[key] for key in METRIC_KEYS],
            "score": [float(latest.get(key, 0.0)) for key in METRIC_KEYS],
        }
    )


def _format_observation(payload: dict[str, Any]) -> str:
    observation = payload.get("observation", {})
    metrics = observation.get("metrics", {})
    task_id = observation.get("task_id", "unknown")
    step_index = observation.get("step_index", 0)
    max_steps = observation.get("max_steps", "?")
    done = payload.get("done", False)
    reward = payload.get("reward")
    feedback = observation.get("feedback_summary", "")

    metric_lines = []
    for key in METRIC_KEYS:
        if key in metrics:
            metric_lines.append(f"- {METRIC_LABELS[key]}: **{float(metrics[key]):.2f}**")

    reward_text = "n/a" if reward is None else f"{float(reward):.4f}"
    done_text = "yes" if done else "no"

    return (
        "## Strategy Console\n\n"
        f"- Task: **{task_id}**\n"
        f"- Step: **{step_index}/{max_steps}**\n"
        f"- Reward: **{reward_text}**\n"
        f"- Episode complete: **{done_text}**\n\n"
        "### Metrics\n"
        + "\n".join(metric_lines)
        + "\n\n### Feedback\n"
        + (feedback or "No feedback yet.")
    )


def _build_score_markdown(
    metrics: dict[str, float],
    payload: dict[str, Any],
    state_data: dict[str, Any] | None,
) -> str:
    if not metrics:
        return EVALUATION_HELP_MD

    average_quality = sum(metrics.values()) / len(metrics)
    viability_gate = min(
        metrics.get("market_demand", 0.0),
        metrics.get("feasibility", 0.0),
        metrics.get("clarity", 0.0),
    )
    reward = payload.get("reward")
    reward_text = "n/a" if reward is None else f"{float(reward):.4f}"

    final_score = None
    if isinstance(state_data, dict):
        raw_score = state_data.get("final_score")
        if raw_score is not None:
            try:
                final_score = float(raw_score)
            except (TypeError, ValueError):
                final_score = None

    final_score_line = (
        f"- Final episode score: **{final_score:.4f}**\n"
        if final_score is not None
        else "- Final episode score: **pending** (available when episode ends)\n"
    )

    return (
        "### Evaluation Snapshot\n\n"
        f"- Average quality (mean of 5 metrics): **{average_quality:.3f}/10**\n"
        f"- Viability gate (min of demand, feasibility, clarity): **{viability_gate:.3f}/10**\n"
        f"- Latest reward: **{reward_text}**\n"
        + final_score_line
        + "\n### Scoring Signals\n"
        "- + Final quality contributes heavily.\n"
        "- + Improvements in average, clarity, and feasibility are rewarded.\n"
        "- + Viability gate contributes when core metrics are strong.\n"
        "- - Feature bloat can apply complexity penalty.\n"
    )


def _default_task_choices() -> tuple[list[tuple[str, str]], str]:
    try:
        tasks = ordered_tasks()
        choices = [
            (f"{task.task_id} ({task.difficulty})", task.task_id)
            for task in tasks
        ]
        default_value = choices[0][1] if choices else "task_easy"
        return choices, default_value
    except Exception:
        fallback = [
            ("task_easy (easy)", "task_easy"),
            ("task_medium (medium)", "task_medium"),
            ("task_hard (hard)", "task_hard"),
        ]
        return fallback, "task_easy"


def build_custom_gradio_app(
    web_manager: Any,
    action_fields: list[dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str = "OpenEnv Environment",
    quick_start_md: str | None = None,
) -> gr.Blocks:
    del action_fields, is_chat_env

    display_title = (
        getattr(metadata, "title", None)
        or getattr(metadata, "name", None)
        or title
    )
    task_choices, default_task = _default_task_choices()

    async def _run_step(action_data: dict[str, Any], history: list[dict[str, Any]]):
        try:
            data = await web_manager.step_environment(action_data)
            state_data = web_manager.get_state()

            metrics = _extract_metrics(data)
            step_index = int(data.get("observation", {}).get("step_index", 0))
            new_history = _append_history(history or [], step_index, metrics)

            observation_md = _format_observation(data)
            raw_json = json.dumps(data, indent=2)
            status = "Decision applied."
            if data.get("done"):
                status = "Decision applied. Episode finished."

            table = _history_dataframe(new_history)
            trend_data = _build_trend_dataframe(new_history)
            snapshot_data = _build_snapshot_dataframe(new_history)
            score_md = _build_score_markdown(metrics, data, state_data)

            return (
                observation_md,
                raw_json,
                status,
                new_history,
                table,
                trend_data,
                snapshot_data,
                score_md,
            )
        except Exception as exc:
            existing_history = history or []
            return (
                "",
                "",
                f"Error: {exc}",
                existing_history,
                _history_dataframe(existing_history),
                _build_trend_dataframe(existing_history),
                _build_snapshot_dataframe(existing_history),
                EVALUATION_HELP_MD,
            )

    async def reset_env(task_id: str):
        try:
            request = {"task_id": task_id} if task_id else None
            data = await web_manager.reset_environment(request)
            observation_md = _format_observation(data)
            raw_json = json.dumps(data, indent=2)

            state_data = web_manager.get_state()
            metrics = _extract_metrics(data)
            step_index = int(data.get("observation", {}).get("step_index", 0))
            history = _append_history([], step_index, metrics)

            table = _history_dataframe(history)
            trend_data = _build_trend_dataframe(history)
            snapshot_data = _build_snapshot_dataframe(history)
            score_md = _build_score_markdown(metrics, data, state_data)
            status = f"Episode reset for scenario '{task_id}'."
            return (
                observation_md,
                raw_json,
                status,
                history,
                table,
                trend_data,
                snapshot_data,
                score_md,
                AI_SUMMARY_DEFAULT_MD,
                AI_STATS_DEFAULT_MD,
            )
        except Exception as exc:
            return (
                "",
                "",
                f"Error: {exc}",
                [],
                _history_dataframe([]),
                _empty_trend_dataframe(),
                _empty_snapshot_dataframe(),
                EVALUATION_HELP_MD,
                AI_SUMMARY_DEFAULT_MD,
                AI_STATS_DEFAULT_MD,
            )

    def suggest_action(goal_text: str, hf_token: str, model_name: str):
        try:
            state_data = web_manager.get_state()
        except Exception:
            state_data = {}

        action, parameter, rationale, status, ai_meta = _build_ai_action(
            goal_text,
            state_data,
            hf_token,
            model_name,
        )
        summary_md, stats_md = _build_ai_result_panels(
            action,
            parameter,
            rationale,
            ai_meta,
        )
        return action, parameter, rationale, status, summary_md, stats_md

    async def apply_step(
        decision: str,
        details: str,
        why: str,
        history: list[dict[str, Any]],
        ai_summary: str,
        ai_stats: str,
    ):
        chosen_action = (decision or "").strip() or "refine_market"
        if chosen_action not in ACTION_CHOICES:
            chosen_action = "refine_market"

        action_data = {
            "action_type": chosen_action,
            "parameter": (details or "").strip(),
            "rationale": (why or "").strip(),
        }

        (
            observation_md,
            raw_json,
            status,
            new_history,
            table,
            trend_data,
            snapshot_data,
            score_md,
        ) = await _run_step(action_data, history)

        return (
            observation_md,
            raw_json,
            status,
            new_history,
            table,
            trend_data,
            snapshot_data,
            score_md,
            ai_summary,
            ai_stats,
        )

    async def ai_step(
        goal_text: str,
        hf_token: str,
        model_name: str,
        history: list[dict[str, Any]],
    ):
        try:
            state_data = web_manager.get_state()
        except Exception:
            state_data = {}

        action, parameter, rationale, ai_status, ai_meta = _build_ai_action(
            goal_text,
            state_data,
            hf_token,
            model_name,
        )

        ai_summary_md, ai_stats_md = _build_ai_result_panels(
            action,
            parameter,
            rationale,
            ai_meta,
        )

        (
            observation_md,
            raw_json,
            step_status,
            new_history,
            table,
            trend_data,
            snapshot_data,
            score_md,
        ) = await _run_step(
            {
                "action_type": action,
                "parameter": parameter,
                "rationale": rationale,
            },
            history,
        )
        status = f"{ai_status} {step_status}".strip()

        return (
            action,
            parameter,
            rationale,
            observation_md,
            raw_json,
            status,
            new_history,
            table,
            trend_data,
            snapshot_data,
            score_md,
            ai_summary_md,
            ai_stats_md,
        )

    def get_state_view():
        try:
            data = web_manager.get_state()
            return json.dumps(data, indent=2), "Loaded latest environment state."
        except Exception as exc:
            return "", f"Error: {exc}"

    with gr.Blocks(title=display_title) as demo:
        gr.HTML(
            (
                "<div class='hero-card'>"
                f"<h1>{display_title} - AI Strategy Lab</h1>"
                "<p>Simple workflow: Reset a scenario, describe your goal in plain English, "
                "and apply steps while tracking live metrics.</p>"
                "</div>"
            )
        )

        history_state = gr.State(value=[])

        with gr.Row(equal_height=True, elem_classes=["compact-row"]):
            with gr.Column(scale=2, elem_classes=["top-ai-pane"]):
                gr.Markdown("### AI Assistant (Top)")
                with gr.Row(elem_classes=["compact-row"]):
                    hf_token_input = gr.Textbox(
                        label="Hugging Face Token",
                        type="password",
                        placeholder="hf_xxx (optional, falls back to HF_TOKEN env)",
                    )
                    model_input = gr.Textbox(
                        label="Model Name",
                        value=(
                            os.getenv("HF_MODEL_NAME", "")
                            or os.getenv("MODEL_NAME", "")
                            or "openai/gpt-4.1-mini"
                        ),
                    )
                goal_input = gr.Textbox(
                    label="Goal in Plain English",
                    lines=3,
                    placeholder="Example: Make this startup easier to build while keeping strong demand.",
                )
                with gr.Row(elem_classes=["compact-row"]):
                    suggest_btn = gr.Button("Suggest with AI", variant="secondary")
                    ai_step_btn = gr.Button("AI Step", variant="primary")

                ai_summary_md = gr.Markdown(value=AI_SUMMARY_DEFAULT_MD)
                ai_stats_md = gr.Markdown(value=AI_STATS_DEFAULT_MD)

            with gr.Column(scale=1, elem_classes=["control-pane"]):
                scenario_selector = gr.Radio(
                    choices=task_choices,
                    value=default_task,
                    label="Scenario",
                    info="Choose a task first, then click Reset Episode.",
                )
                with gr.Row(elem_classes=["compact-row"]):
                    reset_btn = gr.Button("Reset Episode", variant="secondary")
                    state_btn = gr.Button("Get State", variant="secondary")

                decision_input = gr.Radio(
                    choices=ACTION_CHOICES,
                    value="refine_market",
                    label="Decision",
                    info="No dropdown needed. Pick one strategy move.",
                )
                details_input = gr.Textbox(
                    label="Details",
                    lines=2,
                    placeholder="What specific change should be applied?",
                )
                why_input = gr.Textbox(
                    label="Why",
                    lines=2,
                    placeholder="Short reason for this decision",
                )
                step_btn = gr.Button("Apply Decision", variant="primary")
                status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Row(elem_classes=["compact-row"]):
            with gr.Column(scale=2, elem_classes=["result-pane"]):
                observation_md = gr.Markdown(
                    value="## Strategy Console\n\nPress **Reset Episode** to start.",
                )

            with gr.Column(scale=1, elem_classes=["result-pane"]):
                score_md = gr.Markdown(value=EVALUATION_HELP_MD)

        with gr.Row(elem_classes=["compact-row"]):
            trend_plot = gr.LinePlot(
                value=_empty_trend_dataframe(),
                x="step",
                y="score",
                color="metric",
                title="Metric Trend",
                y_lim=[0, 10],
                x_axis_labels_visible=True,
                color_map=METRIC_COLOR_BY_LABEL,
                label="Metric Trend",
            )
            snapshot_plot = gr.BarPlot(
                value=_empty_snapshot_dataframe(),
                x="metric",
                y="score",
                color="metric",
                title="Current Metric Snapshot",
                y_lim=[0, 10],
                color_map=METRIC_COLOR_BY_LABEL,
                label="Current Metric Snapshot",
            )

        with gr.Row(elem_classes=["compact-row"]):
            metric_table = gr.Dataframe(
                label="Metric History",
                value=_history_dataframe([]),
                interactive=False,
            )
            raw_json = gr.Code(
                label="Raw JSON Response",
                language="json",
                interactive=False,
            )

        with gr.Accordion("Evaluation Metrics Used", open=False):
            gr.Markdown(EVALUATION_HELP_MD)

        if quick_start_md:
            with gr.Accordion("Quick Start", open=False):
                gr.Markdown(quick_start_md)

        with gr.Accordion("README", open=False):
            gr.Markdown(
                "Use Reset first. You can run manual decisions or AI-assisted steps from the top panel."
            )

        reset_btn.click(
            fn=reset_env,
            inputs=[scenario_selector],
            outputs=[
                observation_md,
                raw_json,
                status_box,
                history_state,
                metric_table,
                trend_plot,
                snapshot_plot,
                score_md,
                ai_summary_md,
                ai_stats_md,
            ],
        )

        suggest_btn.click(
            fn=suggest_action,
            inputs=[goal_input, hf_token_input, model_input],
            outputs=[
                decision_input,
                details_input,
                why_input,
                status_box,
                ai_summary_md,
                ai_stats_md,
            ],
        )

        step_btn.click(
            fn=apply_step,
            inputs=[
                decision_input,
                details_input,
                why_input,
                history_state,
                ai_summary_md,
                ai_stats_md,
            ],
            outputs=[
                observation_md,
                raw_json,
                status_box,
                history_state,
                metric_table,
                trend_plot,
                snapshot_plot,
                score_md,
                ai_summary_md,
                ai_stats_md,
            ],
        )

        ai_step_btn.click(
            fn=ai_step,
            inputs=[goal_input, hf_token_input, model_input, history_state],
            outputs=[
                decision_input,
                details_input,
                why_input,
                observation_md,
                raw_json,
                status_box,
                history_state,
                metric_table,
                trend_plot,
                snapshot_plot,
                score_md,
                ai_summary_md,
                ai_stats_md,
            ],
        )

        state_btn.click(
            fn=get_state_view,
            outputs=[raw_json, status_box],
        )

    return demo


openenv_web_interface.OPENENV_GRADIO_CSS = (
    openenv_web_interface.OPENENV_GRADIO_CSS + "\n" + CUSTOM_UI_CSS
)
openenv_web_interface.build_gradio_app = build_custom_gradio_app


app = create_app(
    StartupPivotEnvironment,
    StartupPivotAction,
    StartupPivotObservation,
    env_name="startup_pivot_agent",
    max_concurrent_envs=2,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the environment server locally."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
