# Changelog

All notable changes to the Animus monorepo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — Truth Baseline (2026-06-27)

- **`scripts/truth-baseline.py`** — `check_test_count` now fails when the shell command exits non-zero, even when `expected` is not set in the TOML. Previously, missing `python` in the environment produced a silent false positive (0 tests reported as PASS).
- **`truth-baseline.toml`** — added `expected = 1` and `op = ">="` to `core_tests` and `forge_tests` checks, plus `python3` fallback when `python` is unavailable.
- **New `version_alignment` check** — reads all `packages/*/pyproject.toml` and `pwa/package.json` versions, reports mismatches. `contracts/` documented as exception (no manifest). Surfaces actual misalignment: core 2.3.0, forge 1.9.0, bootstrap 0.8.0, kernel 0.1.0, quorum 1.2.0, types 0.1.0, pwa 0.1.0.

### Added — Schema Compiler (2026-06-27)

- **`scripts/compile_schemas.py`** — New script using `datamodel-code-generator` to auto-generate Pydantic v2 models from `packages/contracts/*.schema.json`.
- **Generated 20 schema modules** in `packages/types/src/animus_types/` (`action.py`, `event.py`, `assessment.py`, etc.) plus `common.py` with shared enums (`ArtifactType`, `SubjectDomain`, `SecurityClass`, etc.) and base `Common` model.
- **`packages/types/src/animus_types/__init__.py`** — Updated to re-export all generated models + legacy types.
- **`packages/types/tests/test_schemas.py`** — 22 tests: importable check for all 20 schemas + round-trip validation for `action` and `event`.
- **`packages/contracts/pyproject.toml`** — New package manifest, making contracts a real installable package.

### Added — Scaffold Cleanup (2026-06-27)

- **Root `contracts/` deleted** — Empty directory removed; canonical schemas live in `packages/contracts/`.
- **READMEs added** to `apps/`, `modules/`, `database/`, `infra/`, `evidence/releases/` — each explains what belongs there, boundary vs `packages/`, and owner.
- **ADRs written**: ADR-002 (Audience-Based Docs Tree), ADR-003 (Schema Compiler), ADR-004 (Truth Baseline Fix), ADR-005 (Kernel Extraction).
- **`truth-baseline.toml`** updated: removed `contracts/` from `root_architecture_dirs`, added `root_architecture_readmes` and `adr_coverage` checks.

### Added — Documentation Reorganization (2026-06)

- **Audience-based docs tree** — `docs/` reorganized from flat graveyard (44 files, no index) into structured lanes:
  - `getting-started/` — quickstart, installation, concepts
  - `architecture/` — overview (rewritten for v2.1, 8 planes), packages, decisions, standards
  - `packages/` — per-package documentation with version matrix
  - `contributing/` — setup, workflow, debugging
  - `operators/` — deployment, configuration, monitoring, troubleshooting
  - `reference/` — glossary, FAQ, changelog, security, whitepapers
  - `roadmap/` — current priorities and plans
- **14 new placeholder files** written with minimal viable content across all doc lanes
- **docs-validate.py** — CI validation script for internal links, anchor references, trailing whitespace, and redirect stubs
- **Test count reconciliation** — Core 2,865, Forge 10,304, Bootstrap 2,048, Quorum 961 (total 16,178)
- **Gorgon→Forge naming cleanup** in `packages/forge/` (tests/README, skills/README, 6 SKILL.md files)
- **Bootstrap config reference expanded** in `docs/operators/configuration.md` — full schema covering identity, ollama, gateway, intelligence, channels, self-improvement, proactive, personas
- **Package READMEs written** for Quorum, PWA, Contracts (previously missing)
- **Whitepaper index** populated at `docs/reference/whitepapers/README.md`
- **MkDocs site** — Material theme, dark/light mode, search, code copy buttons. Deploys to GitHub Pages via `.github/workflows/docs-deploy.yml` on every push to `main` that touches docs

### Changed

- `docs/architecture/overview.md` — rewritten from aspirational 4-layer hardware diagram to v2.1 reality (8 technical planes, no stale hardware references)
- `.github/workflows/ci.yml` — Docs Validation job now calls `scripts/docs-validate.py` instead of inline Python/shell
- Root `README.md` — ASCII tree expanded to show all 8 packages, nav links updated to new docs tree

### Fixed

- 6 broken internal links in `docs/reference/faq.md` and `glossary.md` (path depth errors)
- Trailing whitespace in 29 session-modified files
- Stale test counts understated by ~11% across `docs/packages/README.md` and root README

---

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
