# CLAUDE.md — forge

## Project Overview

Multi-agent orchestration framework for production AI workflows

## Current State

- **Version**: 1.2.0
- **Language**: Python
- **Files**: 567 across 2 languages
- **Lines**: 191,125

## Architecture

```
forge/
├── .benchmarks/
├── config/
├── migrations/
├── skills/
│   ├── browser/
│   ├── email/
│   ├── integrations/
│   └── system/
├── src/
│   └── animus_forge/
├── tests/
│   ├── dashboard/
│   └── tui/
├── workflows/
│   └── examples/
├── pyproject.toml
```

## Tech Stack

- **Language**: Python, SQL
- **Framework**: fastapi
- **Package Manager**: pip
- **Linters**: ruff
- **Formatters**: ruff
- **Test Frameworks**: pytest

## Coding Standards

- **Naming**: snake_case
- **Quote Style**: double quotes
- **Type Hints**: present
- **Docstrings**: google style
- **Imports**: absolute
- **Path Handling**: pathlib
- **Line Length (p95)**: 77 characters
- **Error Handling**: Custom exception classes present

## Common Commands

```bash
# test
pytest tests/ -v
# lint
ruff check src/ tests/
# format
ruff format src/ tests/
# coverage
pytest --cov=src/ tests/
# animus-forge
animus_forge.cli:app
```

## Anti-Patterns (Do NOT Do)

- Do NOT commit secrets, API keys, or credentials
- Do NOT skip writing tests for new code
- Do NOT use `os.path` — use `pathlib.Path` everywhere
- Do NOT use bare `except:` — catch specific exceptions
- Do NOT use mutable default arguments
- Do NOT use `print()` for logging — use the `logging` module
- Do NOT use synchronous database calls in async endpoints
- Do NOT return raw dicts — use Pydantic response models

## Dependencies

### Core
- openai
- anthropic
- fastapi
- uvicorn
- streamlit
- google-auth
- google-auth-oauthlib
- google-auth-httplib2
- google-api-python-client
- notion-client
- PyGithub
- pydantic
- pydantic-settings
- python-dotenv
- aiofiles

### Dev
- pytest
- pytest-cov
- pytest-asyncio
- pytest-benchmark
- ruff

## Domain Context

### Key Models/Classes
- `ABTestManager`
- `AIHandlersMixin`
- `APIClientMetricsCollector`
- `APIError`
- `APIException`
- `APIKeyCreate`
- `APIKeyCreateRequest`
- `APIKeyInfo`
- `APIKeyStatus`
- `ActionType`
- `AdaptiveAllocation`
- `AdaptiveRateLimitConfig`
- `AdaptiveRateLimitState`
- `AgentContext`
- `AgentContract`

### API Endpoints
- `/`
- `/agents`
- `/agents/{agent_id}`
- `/auth/login`
- `/budgets`
- `/budgets/summary`
- `/budgets/{budget_id}`
- `/budgets/{budget_id}/add-usage`
- `/budgets/{budget_id}/reset`
- `/credentials`
- `/credentials/{credential_id}`
- `/cycles`
- `/dashboard/budget`
- `/dashboard/recent-executions`
- `/dashboard/stats`

### Enums/Constants
- `ABSTAIN`
- `ACKNOWLEDGED`
- `ACTIVE`
- `ADMIN`
- `AI_PROVIDER`
- `ALERT`
- `ALLOW`
- `ANALYST`
- `ANALYTICS`
- `ANALYZE`

### Outstanding Items
- **TODO**: Fix self-improve coverage test artifact (`tests/test_self_improve_coverage.py`)

## Git Conventions

- Commit messages: Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- Branch naming: `feat/description`, `fix/description`
- Run tests before committing
