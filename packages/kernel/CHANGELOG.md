# Changelog

All notable changes to `animus-kernel` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-21

### Added
- Schema Validation Gate: `scripts/validate_schemas.py` with 5-check CI job (well-formed Draft 2020-12, unique `$id`, filename alignment, dangling `$ref` detection, `$schema` declaration).
- Fleet-wide OPSEC pre-push hook now tracked in notes repo (false-positive fix for `docs/cookbook/` regex-pattern docs).

### Fixed
- Declared missing third-party dependencies (`pydantic-settings`, `jsonschema`, `pyyaml`) in `pyproject.toml` that caused import failures in fresh venvs.

## [0.1.0] - 2026-07-19

### Added
- Initial extraction of Tier 1 primitives from `packages/forge/` and `packages/core/`.
- Memory subsystem with HOT/WARM/COLD tiering (`memory.layer`, `memory.tier`, `memory.stores_local`, `memory.stores_base`, `memory.redaction`).
- Durable memory store backed by SQLAlchemy with bitemporal tracking.
- Local memory store with JSON file persistence and substring search.
- Chroma memory store for vector-based semantic search.
- Provider abstraction layer (`providers.router`, `providers.manager`, `providers.ollama_provider`, `providers.mock_provider`).
- Budget management (`budget.manager`, `budget.models`, `budget.core_bridge`).
- Workflow executor with checkpoint/resume (`executor.checkpoint`, `executor.composer`, `executor.auto_parallel`).
- Safety gates for PII and suspicious code detection (`safety.gates`).
- Resilience patterns: bulkheads, fallbacks, circuit breakers, concurrency limits (`resilience.bulkhead`, `resilience.fallback`, `resilience.concurrency`).
- Rate limiting with token-bucket and quota management (`ratelimit.limiter`, `ratelimit.quota`).
- Metrics collection and cost tracking (`metrics.collector`, `metrics.cost_tracker`, `metrics.debt_monitor`).
- Sandbox orchestration for isolated build-test-lint-rollback cycles (`sandbox.orchestrator`, `sandbox.analyzer`, `sandbox.approval`).
- Server mode with FastAPI (`server.app`).
- Head/session management with context pruning and quality gates (`head.context_manager`, `head.checkpoint`, `head.daemon`).
- State management with SQLite backends (`state.backends`, `state.checkpoint`, `state.agent_context`).
- Tools registry and filesystem operations (`tools.registry`, `tools.filesystem`).
- Builder terminal agent and command runner (`builder.terminal_agent`, `builder.command_runner`).
- 357 unit tests covering memory stores, types, redaction, tier management, and integration workflows.
- PyPI publication workflow via OIDC trusted publishing.

### Changed
- Unified `__init__.py` docstring to reflect all implemented submodules.
- README updated with PyPI install instructions.

### Fixed
- `pyproject.toml` optional dependency syntax corrected from Poetry-style `extra ==` to PEP 621 `[project.optional-dependencies]`.

[Unreleased]: https://github.com/your-org/animus/compare/kernel-v0.1.1...HEAD
[0.1.1]: https://github.com/your-org/animus/releases/tag/kernel-v0.1.1
[0.1.0]: https://github.com/your-org/animus/releases/tag/kernel-v0.1.0
