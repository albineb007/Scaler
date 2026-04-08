---
title: Startup Pivot Agent
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - startup
---

# Startup Pivot Agent (RL Environment)

## Project Description
Startup Pivot Agent is a deterministic OpenEnv reinforcement learning environment where an agent improves weak startup ideas step by step. The environment simulates real strategy decisions used by founders and product teams: narrowing target market, reducing scope, changing pricing, and pivoting the problem statement.

The agent receives dense rewards after each action and a normalized final score from 0.0 to 1.0 for each episode.

## Real-World Motivation
Early-stage startup ideas often fail because they are too broad, too complex, or not feasible for small teams. This environment turns that challenge into a reproducible RL benchmark:

- Learn sequencing of strategic decisions
- Balance innovation against execution feasibility
- Penalize feature bloat and unclear positioning
- Reward practical improvements in clarity and viability

## Environment API
The environment is implemented with typed Pydantic models and OpenEnv-compatible endpoints.

- Action model: `StartupPivotAction`
- Observation model: `StartupPivotObservation`
- State model: `StartupPivotState`
- Core methods:
  - `reset(...)` and `reset_async(...)`
  - `step(...)` and `step_async(...)`
  - `state` property and `state_async()` helper

## Action Space
Available actions are deterministic and structured:

- `refine_market`: narrow and sharpen target audience
- `reduce_scope`: remove non-core complexity
- `add_feature`: add capability (with feasibility/clarity tradeoffs)
- `remove_feature`: simplify by deleting features
- `adjust_pricing`: update monetization strategy
- `pivot_problem`: redefine the startup problem statement

Each action updates startup text and metrics with deterministic transition logic.

## Observation Structure
Each observation includes:

- `current_startup_idea` (text)
- `metrics`:
  - `market_demand` in [0, 10]
  - `feasibility` in [0, 10]
  - `scalability` in [0, 10]
  - `clarity` in [0, 10]
  - `novelty` in [0, 10]
- `feedback_summary` (deterministic explanation)
- `step_index`, `max_steps`, and metadata

## Reward Logic
Dense reward (per step) is deterministic and shaped as:

- Positive signal from average metric delta
- Additional bonus for clarity improvement
- Additional bonus for feasibility improvement
- Penalty for overcomplication when feature count is high
- Penalty for no improvement

Per-step rewards are clipped to [-1.0, 1.0].

Final score is deterministic, normalized to [0.0, 1.0], and computed by a grader comparing initial vs final metrics with task-specific weighting.

## Tasks
Three mandatory tasks are defined in `tasks/startup_tasks.json`:

1. `task_easy` (Easy): decent idea, optimize slightly
2. `task_medium` (Medium): vague idea, structure and narrow it
3. `task_hard` (Hard): unrealistic concept, transform into viable startup

Each task includes:

- Input idea
- Initial deterministic metrics
- Expected improvement statement
- Deterministic grading profile

## Deterministic Grading
The grader compares initial and final metrics and produces a reproducible score:

- Score range: [0.0, 1.0]
- Inputs: metric gains and complexity growth
- Penalty: excessive feature growth beyond target complexity

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r server/requirements.txt
```

## Run The OpenEnv Server

```bash
set ENABLE_WEB_INTERFACE=true
set HF_TOKEN=hf_xxx
set MODEL_NAME=openai/gpt-4.1-mini
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open the browser playground at `http://127.0.0.1:8000/web`.

### Enhanced Web Playground

The web UI supports both manual and natural-language control:

- **Goal in Plain English**: describe what improvement you want
- **Suggest with AI**: converts your goal into a structured action
- **AI Step**: generates and executes an action in one click
- AI panel is placed at the **top** for faster access

The action inputs use simpler labels:

- **Decision** (previously Action Type)
- **Details** (previously Parameter)
- **Why** (previously Rationale)
- Scenario and Decision use **radio buttons** (no dropdowns)

The dashboard also includes:

- Live metric trend chart across steps
- Current metric snapshot chart
- Metric history table
- Evaluation panel showing scoring signals
- AI Input Summary (free-text summary + model summary)
- AI Stats (mode, model, token estimates, latency)

## Run Inference
The inference runner reads these environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=openai/gpt-4.1-mini
set HF_TOKEN=hf_xxx
python inference.py
```

Logs follow this exact structure:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... score=... rewards=...`

## Expected Baseline Scores
With the included deterministic policy in `inference.py`, current validated scores are:

- Easy: ~0.6292
- Medium: ~0.6484
- Hard: ~0.9223

Exact values are reproducible for fixed task definitions and transition logic.

## Docker
Build and run locally:

```bash
docker build -t startup-pivot-agent -f server/Dockerfile .
docker run --rm -p 8000:8000 startup-pivot-agent
```

Then open `http://127.0.0.1:8000/web` to use the interactive playground.

This Dockerfile is compatible with Hugging Face Spaces Docker runtime.

## Project Structure

```
my_env/
├── __init__.py
├── client.py
├── graders.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── tasks.py
├── tasks/
│   └── startup_tasks.json
└── server/
    ├── __init__.py
    ├── app.py
    ├── my_env_environment.py
    ├── requirements.txt
    └── Dockerfile
```
"# Scaler" 
