# CLAUDE.md — core

## Project Overview

An exocortex architecture for personal cognitive sovereignty

## Current State

- **Version**: 1.0.0
- **Language**: Python
- **Files**: 106 across 1 languages
- **Lines**: 48,110

## Architecture

```
core/
├── .benchmarks/
├── animus/
│   ├── forge/
│   ├── integrations/
│   ├── learning/
│   ├── protocols/
│   ├── swarm/
│   └── sync/
├── configs/
│   ├── examples/
│   └── media_engine/
├── tests/
├── pyproject.toml
```

## Tech Stack

- **Language**: Python
- **Framework**: fastapi
- **Package Manager**: pip
- **Linters**: ruff
- **Formatters**: ruff
- **Type Checkers**: mypy
- **Test Frameworks**: pytest

## Coding Standards

- **Naming**: snake_case
- **Quote Style**: double quotes
- **Type Hints**: present
- **Docstrings**: google style
- **Imports**: absolute
- **Path Handling**: pathlib
- **Line Length (p95)**: 78 characters
- **Error Handling**: Custom exception classes present

## Common Commands

```bash
# test
pytest tests/ -v
# lint
ruff check animus/ tests/
# format
ruff format animus/ tests/
# type check
mypy animus/
# coverage
pytest --cov=animus/ tests/
# animus
animus.__main__:main
```

## Anti-Patterns (Do NOT Do)

- Do NOT commit secrets, API keys, or credentials
- Do NOT skip writing tests for new code
- Do NOT use synchronous database calls in async endpoints
- Do NOT return raw dicts — use Pydantic response models
- Do NOT use `os.path` — use `pathlib.Path` everywhere
- Do NOT use bare `except:` — catch specific exceptions
- Do NOT use mutable default arguments
- Do NOT use `print()` for logging — use the `logging` module

## Dependencies

### Core
- ollama
- chromadb
- pyyaml
- pydantic
- rich
- prompt-toolkit

### Dev
- pytest
- pytest-benchmark
- pytest-cov
- ruff
- mypy
- pre-commit
- httpx

## Domain Context

### Key Models/Classes
- `APIConfig`
- `APIServer`
- `ActionLevel`
- `ActionLog`
- `ActionPolicy`
- `ActionStatus`
- `AgentConfig`
- `AnimusConfig`
- `AnthropicModel`
- `AppState`
- `ApprovalManager`
- `ApprovalRequest`
- `ApprovalRequirement`
- `ApprovalStatus`
- `AuthType`

### API Endpoints
- `/autonomous/actions`
- `/autonomous/actions/{action_id}/approve`
- `/autonomous/actions/{action_id}/deny`
- `/autonomous/pending`
- `/autonomous/stats`
- `/brief`
- `/chat`
- `/dashboard`
- `/dashboard/data`
- `/decide`
- `/entities`
- `/entities/search`
- `/entities/stats`
- `/entities/{entity_id}`
- `/entities/{entity_id}/timeline`

### Enums/Constants
- `ACCESS`
- `ACT`
- `ACTIVE`
- `ANTHROPIC`
- `API_KEY`
- `APPROVE`
- `APPROVED`
- `AUTH`
- `AUTH_FAIL`
- `AUTH_OK`

## Git Conventions

- Commit messages: Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- Branch naming: `feat/description`, `fix/description`
- Run tests before committing
