# CLAUDE.md — bootstrap

## Project Overview

Animus Bootstrap — install daemon, onboarding wizard, local dashboard, message gateway, intelligence layer, persona system, and identity/self-improvement infrastructure. The first running piece of Animus and the foundation for self-improvement.

## Current State

- **Version**: 0.5.0
- **Language**: Python
- **Files**: 180 across 3 languages
- **Lines**: 35,172
- **Tests**: 1403, 94% coverage (fail_under=95)
- **Tools**: 31 (8 built-in + 4 identity + MCP bridge)
- **Dashboard Pages**: 15+ (HTMX, localhost:7700)

## Architecture

```
bootstrap/
├── src/animus_bootstrap/
│   ├── config/           # Pydantic config, TOML read/write, chmod 600
│   ├── daemon/           # OS detection, systemd/launchd, supervisor, updater
│   │   └── platforms/    # Linux, macOS, Windows service modules
│   ├── setup/            # Rich TUI wizard (9 steps)
│   │   └── steps/        # welcome, identity, identity_files, api_keys, forge, memory, device, sovereignty, launch
│   ├── identity/         # 6 identity files, CORE_VALUES.md lock, Jinja2 templates
│   │   └── templates/    # *.md.j2 templates for wizard generation
│   ├── gateway/          # Message gateway (8 channel adapters, cognitive backends)
│   │   ├── channels/     # telegram, discord, slack, matrix, whatsapp, signal, email, webchat
│   │   └── middleware/   # auth, rate limiting, logging
│   ├── intelligence/     # Memory, tools, proactive engine, automations
│   │   ├── memory_backends/  # SQLite FTS5, ChromaDB (with fallback)
│   │   ├── tools/builtin/    # 31 tools (shell, web, code, memory, identity, etc.)
│   │   ├── proactive/checks/ # morning_brief, task_nudge, calendar, reflection
│   │   ├── automations/      # trigger/condition/action pipeline
│   │   └── feedback.py       # Thumbs up/down store (SQLite WAL)
│   ├── personas/         # PersonaEngine, voice presets, domain routing
│   └── dashboard/        # FastAPI + HTMX + Tailwind (dark theme)
│       ├── routers/      # home, config, memory, logs, tools, identity, proposals, feedback, etc.
│       ├── templates/    # Jinja2 HTML templates
│       └── static/css/   # Custom styles
├── tests/                # 1403 tests
└── pyproject.toml
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
- **Line Length (p95)**: 75 characters

## Common Commands

```bash
# test (use venv — has all deps including pytest-asyncio)
.venv/bin/python -m pytest tests/ -v
# lint + format
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
# auto-fix
.venv/bin/python -m ruff check . --fix && .venv/bin/python -m ruff format .
# coverage
.venv/bin/python -m pytest tests/ --cov=src/animus_bootstrap --cov-fail-under=95
# install dev
pip install -e ".[dev]"
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
- rich
- fastapi
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
- Animus Bootstrap Animus
- Animus Core
- Animus Ecosystem
- Animus Forge
- Animus Quorum
- Configuration Config
- MIT
- Platform Support

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

## Critical Design Constraints

- **CORE_VALUES.md is immutable** — Animus can never write to this file. `IdentityFileManager.write()` raises `PermissionError` for locked files. `identity_write` tool returns graceful error (not exception). Only editable via dashboard (`write_locked()`) or wizard.
- **20% change threshold** — Identity file changes >20% of file size become proposals (not direct writes). Proposals require human approval in dashboard `/proposals` page.
- **Local-first** — No data leaves machine without explicit user action. No telemetry by default. Single-user architecture. Works offline (Ollama local).
- **chmod 600 on config** — API keys file-permission protected.
- **asyncio.run() poisoning** — Test files must use `asyncio.new_event_loop()` + `set_event_loop()` pattern, NOT `asyncio.run()`, or downstream tests using `get_event_loop()` will fail in suite.

## Git Conventions

- Commit messages: Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)
- Branch naming: `feat/description`, `fix/description`
- Run tests before committing
