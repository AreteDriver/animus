# CLAUDE.md — quorum

## Project Overview

Multi-agent coherence and coordination for AI systems

## Current State

- **Version**: 1.2.0
- **Language**: Python
- **Files**: 79 across 2 languages
- **Lines**: 26,474

## Architecture

```
quorum/
├── .benchmarks/
├── benches/
├── python/
│   └── convergent/
├── src/
├── tests/
├── .gitleaks.toml
├── CLAUDE.md
├── Cargo.lock
├── Cargo.toml
├── pyproject.toml
```

## Tech Stack

- **Language**: Python, Rust
- **Framework**: rust
- **Package Manager**: cargo, pip
- **Linters**: clippy, ruff
- **Formatters**: ruff
- **Type Checkers**: mypy
- **Test Frameworks**: cargo test, pytest

## Coding Standards

- **Naming**: snake_case
- **Quote Style**: double quotes
- **Type Hints**: present
- **Docstrings**: google style
- **Imports**: absolute
- **Path Handling**: pathlib
- **Line Length (p95)**: 76 characters

## Common Commands

```bash
# test
pytest tests/ -v
# lint
ruff check src/ tests/
# format
ruff format src/ tests/
# type check
mypy src/
# coverage
pytest --cov=src/ tests/
```

## Anti-Patterns (Do NOT Do)

- Do NOT commit secrets, API keys, or credentials
- Do NOT skip writing tests for new code
- Do NOT use `.unwrap()` in production code — use proper error handling
- Do NOT use `unsafe` without a safety comment
- Do NOT clone when a reference will do
- Do NOT use `os.path` — use `pathlib.Path` everywhere
- Do NOT use bare `except:` — catch specific exceptions
- Do NOT use mutable default arguments
- Do NOT use `print()` for logging — use the `logging` module

## Dependencies

### Core
- pyo3
- rusqlite
- serde
- serde_json
- uuid
- chrono

### Dev
- pytest
- pytest-asyncio
- pytest-benchmark
- ruff
- mypy

## Domain Context

### Key Models/Classes
- `Adjustment`
- `AdjustmentKind`
- `AgentAction`
- `AgentBranch`
- `AgentIdentity`
- `AgentLog`
- `AlwaysHardFail`
- `AnthropicSemanticMatcher`
- `AsyncBackendWrapper`
- `AsyncGraphBackend`
- `AuthService`
- `BenchmarkMetrics`
- `BenchmarkSuite`
- `Budget`
- `CommandGate`

### Enums/Constants
- `ABSTAIN`
- `ADD_EVIDENCE`
- `ANY`
- `APPEND_ONLY`
- `APPROVE`
- `APPROVED`
- `AUTO_RESOLVE`
- `AdjustmentKind`
- `BLOCK`
- `BLOCKED_BY_CONFLICT`

### Outstanding Items
- **TODO**: Add benchmarks for intent graph operations (`benches/intent_graph.rs`)

## Git Conventions

- Commit messages: Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- Branch naming: `feat/description`, `fix/description`
- Run tests before committing
