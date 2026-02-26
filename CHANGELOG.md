# Changelog

All notable changes to the Animus monorepo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
