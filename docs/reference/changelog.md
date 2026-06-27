# Changelog

All notable changes to the Animus monorepo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Quorum v2 Week 1: EventLog bitemporal + signal bridge

- **Bitemporal-lite on `CoordinationEvent`** — `valid_from` (world-time) and `recorded_at` (observation-time) fields ported from memboot's pattern. Backward-compatible: `timestamp` preserved as alias.
- **Idempotent SQLite migration** — `EventLog._migrate()` adds the two columns and backfills both from `timestamp` on legacy databases; safe to re-run.
- **`EventLog.query()` bitemporal range filters** — `valid_from_since` / `valid_from_until` for world-time windows; `since` / `until` continue to filter by observation-time.
- **Signal bridge** — `EventLog(signal_bus=bus)` mirrors every recorded event onto the bus as `Signal(signal_type=f"event.{event_type.value}")`. Best-effort; failures swallowed so bus issues never break `record()`.
- **4 of 5 mutation sites wired** to `EventLog.record()` — closes Constitutional Principle P3 (Transparency) on `INTENT_PUBLISHED` (resolver), `VOTE_CAST` (triumvirate), `DECISION_MADE` (triumvirate), `MARKER_LEFT` (stigmergy). `SCORE_UPDATED` and `INTENT_RESOLVED` defer to Week 3-4 scorer registry.
- Quorum test count: 920 → 957 (+37 tests across 4 new test files).

See `docs/specs/quorum_v2_week1_event_log_extension.md` and ADL-20260510-001.

## [2.2.0] - 2026-03-05

### Added
- **Dual-model routing** — Claude as brain (planning, code gen) + Ollama as hands (summarization, formatting)
- **Task classification** — `classify_task()` heuristic routes HEAVY vs LIGHT tasks to appropriate model
- **Autonomous build pipeline** — 4-agent Forge workflow (planner → coder → verifier → fixer) with quality gates and $2.00 budget cap
- **Constrained tool selection** — Numbered menu + key:value parsing for reliable Ollama tool use
- **Task outcome tracking** — Records outcomes in MemoryLayer, recalls similar past tasks, detects failure patterns, tracks success rates
- **New archetypes** — `planner`, `coder`, `verifier` added to ForgeAgent
- **New slash commands** — `/build`, `/model`, `/stats` in chat agent
- **`create_local_think_tool()`** — Lets Claude offload cheap subtasks to Ollama during agentic loop
- Core test count: 1879 → 2046 (+167 tests)

### Changed
- `think_with_tools()` dispatch: non-Anthropic models now route to constrained loop instead of markdown loop
- `chat.py` fully wired to Animus Core (CognitiveLayer, ToolRegistry, MemoryLayer, TaskOutcomeTracker)

## [2.0.0] - 2026-02-20

### Added
- **Monorepo consolidation** — Gorgon and Convergent merged into Animus as Forge and Quorum packages
- Four independently installable packages: Core, Forge, Quorum, Bootstrap
- Monorepo CI pipeline with per-package test jobs, security scanning, and benchmarks
- Bootstrap package (v0.5.0): install daemon, onboarding wizard, dashboard, gateway, intelligence layer, persona system

### Changed
- Forge: `test_ai` module renamed to `animus_forge`
- Quorum: keeps `convergentAI` PyPI name, imports as `convergent`
- Core: lightweight forge/swarm sub-engines for embedded orchestration

### Fixed
- OAuth test patch target and race condition
- Forge async test failures (stale from prior codebase)

## [1.0.0] - 2026-02-20

### Added
- Forge/Swarm revise gate loop-back with `ReviseRequestedError`
- Register translation (LLM-based formal/casual/technical)
- Native Anthropic tool_use in cognitive layer
- `/workflow` command in chat.py for running Forge YAML workflows

### Fixed
- OAuth test patching correct target
- `datetime.utcnow()` deprecation warnings

## [0.7.0] - 2026-02-19

### Added
- Swarm parallel agent orchestration with stigmergic coordination
- DAG analysis via Kahn's topological sort with parallel execution stages
- Stage-level atomic checkpoints
- Coverage hardening: 584 to 1475 tests (46% to 91%)

### Fixed
- Gitleaks shallow clone issue (fetch-depth: 0)
- Python 3.10 compatibility (`datetime.UTC` to `timezone.utc`)
- 30 CodeQL alerts resolved to 0

## [0.6.0] - 2026-02-19

### Added
- Forge multi-agent orchestration MVP
- YAML workflows with agent archetypes, token budgets, quality gates
- SQLite WAL checkpoints for workflow state
- Google Calendar and Gorgon integrations
- Sync protocol and learning system
