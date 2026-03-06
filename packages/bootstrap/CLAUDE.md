# CLAUDE.md — bootstrap

## Project Overview

Animus install daemon, onboarding wizard, and local dashboard

## Current State

- **Version**: 0.5.0
- **Language**: Python
- **Files**: 199 across 3 languages
- **Lines**: 41,702

## Architecture

```
bootstrap/
├── .benchmarks/
├── src/
│   └── animus_bootstrap/
├── tests/
├── CLAUDE.md
├── PHASE2_GATEWAY.md
├── PHASE3_INTELLIGENCE.md
├── PHASE4_PERSONAS.md
├── README.md
├── pyproject.toml
```

## Tech Stack

- **Language**: Python, HTML, CSS
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
- **Line Length (p95)**: 74 characters

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
# animus-bootstrap
animus_bootstrap.cli:app
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
- typer
- 
- rich
- 
- fastapi
- 
- uvicorn

### Dev
- pytest
- pytest-cov
- pytest-asyncio
- ruff
- mypy
- respx

## Domain Context

### Key Models/Classes
- `ActionConfig`
- `AnimusConfig`
- `AnimusInstaller`
- `AnimusMemoryBackend`
- `AnimusRuntime`
- `AnimusSection`
- `AnimusSupervisor`
- `AnimusUpdater`
- `AnimusWizard`
- `AnthropicBackend`
- `ApiSection`
- `Attachment`
- `AutomationEngine`
- `AutomationResult`
- `AutomationRule`

### Domain Terms
- AI
- Animus Bootstrap
- Animus Core
- Animus Ecosystem
- Animus Forge
- Animus Quorum
- CONTEXT
- Configuration Config
- Feedback Signal Thumbs
- GOALS

### API Endpoints
- `/`
- `/activity`
- `/api/feedback`
- `/automations`
- `/channels`
- `/channels/{channel_name}/toggle`
- `/config`
- `/conversations`
- `/conversations/messages`
- `/feedback`
- `/forge`
- `/health`
- `/identity`
- `/identity/edit/{filename}`
- `/identity/view/{filename}`

### Enums/Constants
- `API_URL`
- `APPROVE`
- `AUTO`
- `DENY`
- `_AGENT_LABEL`
- `_API_BASE`
- `_CONFIG_FILE`
- `_DEFAULT_FORGE_HOST`
- `_DEFAULT_HOST`
- `_DEFAULT_PATH`

## Git Conventions

- Commit messages: Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- Branch naming: `feat/description`, `fix/description`
- Run tests before committing
