"""Task loading utilities for Startup Pivot Agent."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .models import TaskSpec
except ImportError:
    from models import TaskSpec

DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "tasks" / "startup_tasks.json"
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def load_tasks(tasks_path: Path | None = None) -> dict[str, TaskSpec]:
    """Load all task specifications from JSON."""
    source = tasks_path or DEFAULT_TASKS_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks", [])

    tasks: dict[str, TaskSpec] = {}
    for item in raw_tasks:
        task = TaskSpec(**item)
        tasks[task.task_id] = task

    return tasks


def ordered_tasks(tasks_path: Path | None = None) -> list[TaskSpec]:
    """Return tasks sorted by difficulty and id for deterministic evaluation."""
    tasks = load_tasks(tasks_path)
    return sorted(
        tasks.values(),
        key=lambda item: (DIFFICULTY_ORDER.get(item.difficulty, 99), item.task_id),
    )
