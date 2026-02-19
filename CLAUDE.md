# CLAUDE.md — Animus

> Personal AI exocortex with multi-agent orchestration.

## Quick Reference

- **Version**: 0.7.0
- **Python**: >=3.10
- **Package layout**: `animus/` (setuptools, flat layout with `forge/` subpackage)
- **Tests**: `tests/` (pytest, 1703 tests, 96% coverage, fail_under=80)
- **License**: MIT

## Build & Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# CLI
animus                          # Interactive CLI (prompt-toolkit)
```

## Testing

```bash
pytest tests/ -v                           # Full suite
pytest tests/test_forge_units.py -v        # Forge unit tests
pytest tests/test_forge_integration.py -v  # Forge integration tests
pytest tests/test_swarm.py -v              # Swarm unit + integration tests
```

## Linting

```bash
ruff check animus/ tests/                  # Lint
ruff format animus/ tests/                 # Format
ruff check animus/ tests/ && ruff format --check animus/ tests/  # CI check
```

Rules: E, F, I, N, W, UP. E501 ignored (line length handled separately). B008 not enabled.

## Architecture

```
animus/
├── __init__.py              # Version, public API exports
├── __main__.py              # CLI entry point (prompt-toolkit, 40+ /commands)
├── cognitive.py             # CognitiveLayer — LLM interface (Ollama/Anthropic/OpenAI/Mock)
├── config.py                # AnimusConfig — dataclass + YAML + env overrides
├── memory.py                # MemoryLayer — episodic/semantic/procedural (ChromaDB)
├── tools.py                 # ToolRegistry, Tool, ToolResult, 6 built-in tools
├── logging.py               # get_logger(name), rotating file handler
├── entities.py              # Entity memory + relationship graph
├── learning/                # Self-learning system (patterns, preferences, guardrails)
├── proactive.py             # Context nudges, deadline scanning
├── autonomous.py            # Action policies (observe/notify/act/execute)
├── decision.py              # Decision framework
├── tasks.py                 # Task tracker
├── register.py              # Communication register detection + adaptation
├── api.py                   # FastAPI HTTP server (optional)
├── voice.py                 # Whisper STT + TTS (optional)
├── integrations/            # Google, Todoist, filesystem, webhooks, Gorgon
├── sync/                    # Multi-device sync protocol
├── protocols/               # ABCs: intelligence, memory, safety, sync
├── forge/                   # ★ Multi-agent orchestration engine
│   ├── __init__.py          #   Public API (ForgeEngine, WorkflowConfig, etc.)
│   ├── models.py            #   Dataclasses: AgentConfig, GateConfig, WorkflowConfig, StepResult, WorkflowState
│   ├── loader.py            #   YAML workflow parser with validation
│   ├── budget.py            #   Per-agent token + cost ceiling tracking
│   ├── gates.py             #   Quality gate evaluator (safe condition parser, no eval)
│   ├── checkpoint.py        #   SQLite WAL state persistence (pause/resume)
│   ├── agent.py             #   ForgeAgent — CognitiveLayer wrapper with archetype prompts
│   └── engine.py            #   ForgeEngine — sequential orchestration loop
├── swarm/                   # ★ Parallel agent orchestration (stigmergic coordination)
│   ├── __init__.py          #   Public API with lazy imports
│   ├── models.py            #   SwarmConfig, SwarmStage, IntentEntry, SwarmError
│   ├── graph.py             #   DAG analysis + Kahn's topological sort → parallel stages
│   ├── intent.py            #   Thread-safe IntentGraph + IntentResolver (stability-based)
│   └── engine.py            #   SwarmEngine — parallel execution via ThreadPoolExecutor
└── dashboard.py             # Streamlit ops dashboard
```

## Layer Overview

**Core** (`animus/`) — User-facing exocortex. Identity, memory, CLI, voice, integrations.

**Forge** (`animus/forge/`) — Multi-agent orchestration. Declarative YAML workflows, token budgets, quality gates, SQLite checkpoint/resume. Sequential execution MVP. Provider-agnostic via CognitiveLayer.

**Swarm** (`animus/swarm/`) — Parallel agent orchestration. DAG-based stage derivation, ThreadPoolExecutor parallel execution, stigmergic intent graph for conflict detection. Extends Forge (reuses ForgeAgent, BudgetTracker, CheckpointStore, gates). YAML `execution_mode: parallel` triggers Swarm engine.

## Key Patterns

- **Dataclasses** throughout (not Pydantic) for models — `AgentConfig`, `ToolResult`, `ModelConfig`, etc.
- **CognitiveLayer.think()** is sync. Forge agents delegate to it. `think_with_tools()` for tool-using agents.
- **ModelConfig.mock()** for testing — deterministic responses via `default_response` + `response_map` dict.
- **get_logger("name")** from `animus.logging` — structured logging, rotating file handler.
- **ToolResult(tool_name, success, output, error)** — standard tool response pattern.
- **Config**: `AnimusConfig.load()` → YAML file + env var overrides via `__post_init__`.
- **Forge YAML loader**: `load_workflow_str(yaml)` validates agent names unique, input refs point to prior agents, gate refs valid, revise requires target.
- **Gate conditions**: Safe parser (no eval). Supports `field >= N`, `field contains "text"`, `field.length >= N`, `true`/`false`. JSON field access via dot notation.
- **Forge checkpoints**: SQLite WAL, `workflows` + `step_results` tables. `ForgeEngine(checkpoint_dir=path)` enables persistence.
- **Budget tracking**: `BudgetTracker.from_config(workflow)`. Per-agent token limits + workflow cost ceiling. Raises `BudgetExhaustedError`.
- **Archetype prompts**: 6 built-in (researcher, writer, reviewer, producer, editor, analyst). Custom via `system_prompt` field.

## Forge YAML Schema

```yaml
name: my_workflow
description: What this workflow does
provider: ollama          # Default LLM provider
model: llama3:8b          # Default model
max_cost_usd: 1.0         # Workflow cost ceiling

agents:
  - name: researcher
    archetype: researcher
    budget_tokens: 5000
    outputs: [brief]

  - name: writer
    archetype: writer
    budget_tokens: 8000
    inputs: [researcher.brief]   # agent.output reference
    outputs: [draft]

gates:
  - name: quality_check
    after: writer
    type: automated             # or "human"
    pass_condition: "true"      # safe expression
    on_fail: halt               # halt, skip, or revise
```

## Swarm YAML Schema

```yaml
name: parallel_workflow
execution_mode: parallel    # Enables SwarmEngine (default: sequential → ForgeEngine)
provider: ollama
model: llama3:8b
max_cost_usd: 1.0

agents:
  - name: researcher
    archetype: researcher
    outputs: [brief]

  - name: fact_checker
    archetype: analyst
    outputs: [facts]

  - name: writer
    archetype: writer
    inputs: [researcher.brief, fact_checker.facts]  # Forward refs OK in parallel mode
    outputs: [draft]
# Stages derived from DAG: [researcher, fact_checker] → [writer]
```

## Optional Dependencies

```bash
pip install animus[anthropic]     # Claude API
pip install animus[openai]        # OpenAI / compatible endpoints
pip install animus[api]           # FastAPI HTTP server
pip install animus[voice]         # Whisper + TTS
pip install animus[integrations]  # Google, Todoist
pip install animus[gorgon]        # Gorgon HTTP client (httpx)
pip install animus[sync]          # Multi-device sync (websockets)
```

## Conventions

- Type hints throughout (mypy advisory, `ignore_missing_imports = true`)
- Dataclasses for models, not Pydantic (despite YAML config saying otherwise)
- `from __future__ import annotations` in test files
- Ruff for all linting/formatting
- Conventional commits
- Tests in flat `tests/` directory (not mirrored subdirectories)
