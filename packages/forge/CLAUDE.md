# CLAUDE.md — forge

## Project Overview

Multi-agent orchestration framework for production AI workflows

## Current State

- **Version**: 1.7.0
- **Language**: Python
- **Files**: 638 across 2 languages
- **Lines**: 239,179

## Architecture

```
forge/
├── .benchmarks/
├── .gorgon/
│   └── snapshots/
├── config/
├── logs/
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
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── token.json
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
- Do NOT use synchronous database calls in async endpoints
- Do NOT return raw dicts — use Pydantic response models
- Do NOT use `os.path` — use `pathlib.Path` everywhere
- Do NOT use bare `except:` — catch specific exceptions
- Do NOT use mutable default arguments
- Do NOT use `print()` for logging — use the `logging` module

## Dependencies

### Core
- openai
- 
- anthropic
- 
- fastapi
- 
- uvicorn
- 
- streamlit
- 
- google-auth
- 
- google-auth-oauthlib
- 
- google-auth-httplib2

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
- `AgentConfig`
- `AgentContext`

### Domain Terms
- AI
- Animus Bootstrap
- Animus Core
- Animus Forge Multi
- Animus Monorepo
- Animus Quorum
- Improvement Pipeline The
- MCP
- MIT
- Model Context Protocol

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
- `ACT`
- `ACTIVE`
- `ADMIN`
- `AGENT`
- `AI_PROVIDER`
- `ALERT`
- `ALLOW`
- `ANALYST`

### Outstanding Items
- **TODO**: fix this bug\n") (`tests/test_self_improve_coverage.py`)
- **TODO**: handle division by zero\n" (`tests/test_self_improve_ollama_integration.py`)

## AI Skills

**Installed**: 122 skills in `~/.claude/skills/`
- `a11y`, `accessibility-checker`, `agent-teams-orchestrator`, `align-debug`, `api-client`, `api-docs`, `api-tester`, `apple-dev-best-practices`, `arch`, `backup`, `brand-voice-architect`, `build`, `changelog`, `ci`, `cicd-pipeline`
- ... and 107 more

**Recommended bundles**: `api-integration`, `full-stack-dev`

**Recommended skills** (not yet installed):
- `api-integration`
- `full-stack-dev`

## Git Conventions

- Commit messages: Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- Branch naming: `feat/description`, `fix/description`
- Run tests before committing
