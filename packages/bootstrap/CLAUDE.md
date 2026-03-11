# CLAUDE.md — Animus Bootstrap

## Project Overview

Animus install daemon, onboarding wizard, local dashboard, message gateway, intelligence layer, persona system, and self-improvement loop.

## Current State

- **Version**: 0.8.0
- **Language**: Python (+ HTML/CSS templates)
- **Source files**: 126 Python modules, 17K LOC
- **Test files**: 49 test modules, 25K LOC
- **Tests**: 1,841 passed, 35 skipped

## Architecture

```
bootstrap/
├── src/animus_bootstrap/
│   ├── cli.py                    # Typer CLI (personas, automations, tools)
│   ├── runtime.py                # AnimusRuntime — central boot orchestrator
│   ├── config/                   # TOML config (schema, manager, defaults)
│   ├── daemon/                   # systemd/launchd/Windows service management
│   ├── dashboard/                # FastAPI + HTMX dashboard (localhost:7700)
│   │   ├── app.py               # Lifespan, WebSocket, health endpoint
│   │   ├── routers/             # 18 page routers
│   │   ├── static/              # CSS/JS
│   │   └── templates/           # Jinja2 HTML
│   ├── gateway/                  # Message gateway
│   │   ├── channels/            # 8 adapters (Telegram/Discord/Slack/Matrix/WhatsApp/Signal/Email/WebChat)
│   │   ├── cognitive.py         # LLM backends (Anthropic/Ollama/DualOllama/Forge)
│   │   ├── router.py            # MessageRouter (broadcast, channel registry)
│   │   ├── session.py           # SessionManager (SQLite)
│   │   └── models.py            # GatewayMessage, GatewayResponse
│   ├── identity/                 # Identity file management (LEARNED.md, proposals)
│   ├── intelligence/
│   │   ├── router.py            # IntelligentRouter (memory + tool loop)
│   │   ├── memory.py            # MemoryManager
│   │   ├── feedback.py          # FeedbackStore (thumbs up/down)
│   │   ├── memory_backends/     # SQLite FTS5, ChromaDB, AnimusCore
│   │   ├── tools/
│   │   │   ├── executor.py      # ToolExecutor (permissions, history, approval)
│   │   │   ├── history_store.py # Persistent tool history (SQLite)
│   │   │   ├── mcp_bridge.py    # MCP tool auto-discovery
│   │   │   └── builtin/         # 37 built-in tools
│   │   │       ├── self_improve.py      # analyze, propose, apply, measure, rollback
│   │   │       ├── improvement_store.py # SQLite proposals + impact metrics
│   │   │       ├── sandbox.py           # Safe TOML config + identity changes with backup
│   │   │       ├── code_edit.py         # read/write/patch/list
│   │   │       ├── forge_ctl.py         # status/start/stop/invoke
│   │   │       ├── timer_ctl.py         # timer CRUD
│   │   │       ├── task_ctl.py          # task CRUD
│   │   │       ├── identity_tools.py    # view/edit identity files
│   │   │       ├── memory_tools.py      # store/search/recall
│   │   │       └── gateway_tools.py     # send/broadcast
│   │   ├── proactive/
│   │   │   ├── engine.py        # ProactiveEngine (scheduler, quiet hours, nudges)
│   │   │   └── checks/          # 6 built-in checks
│   │   │       ├── self_heal.py     # Auto-detect tool failures → propose improvements
│   │   │       ├── reflection.py    # Feedback → LEARNED.md (3 AM daily)
│   │   │       ├── morning_brief.py
│   │   │       ├── tasks.py
│   │   │       ├── calendar.py
│   │   │       └── verdict_sync.py
│   │   └── automations/         # Trigger/condition/action pipeline (SQLite)
│   ├── personas/                 # PersonaEngine, VoiceConfig, KnowledgeDomainRouter
│   ├── installer.py             # One-command install
│   └── wizard.py                # Rich-based setup wizard (8 steps)
├── tests/                        # 49 test modules
└── pyproject.toml
```

## Key Concepts

### AnimusRuntime (runtime.py)
Central orchestrator. Boots all components in order:
1. Identity manager → 2. Session manager → 3. Cognitive backend → 4. Memory → 5. Tool executor (+ history store, improvement store, sandbox) → 6. Automation engine → 7. Persona engine → 8. Router → 9. Proactive engine → 10. Feedback store → 11. Channel adapters

### Self-Improvement Loop (Phase 5)
```
Tool failures → self_heal detects (every 6h)
  → auto-proposes improvement with AI analysis
  → apply_improvement executes via sandbox (TOML config / identity files)
  → backup created, baseline metrics captured
  → config hot-reloaded at runtime (no restart)
  → measure_impact compares before/after → score -100 to +100
  → regression? → rollback_improvement restores backup + reloads
```

### Config
- Format: TOML at `~/.config/animus/config.toml`
- Schema: Pydantic models in `config/schema.py`
- Hot-reload: `runtime.reload_config()` updates tool timeout, max calls, quiet hours, system prompt

### ChannelAdapter Protocol
Runtime-checkable: `connect()`, `disconnect()`, `send_message(GatewayResponse)`, `on_message(callback)`, `health_check()`

## Common Commands

```bash
# test
pytest tests/ -v
# lint
ruff check src/ tests/ && ruff format src/ tests/
# run dashboard
python -m animus_bootstrap.dashboard.app
# run daemon
python -m animus_bootstrap.daemon
# health check
curl http://localhost:7700/health
```

## Non-Negotiables

- SQLite `check_same_thread=False` for all stores (FastAPI cross-thread)
- All SQLite stores use WAL mode
- asyncio.to_thread() for sync-to-async bridges (AnimusMemoryBackend)
- `pytestmark = pytest.mark.skipif()` for optional deps (animus-core, chromadb, pytest-benchmark)
- patch.object for chromadb mock tests (module-level import caching)
- Identity changes go through IdentityProposalManager (20% threshold)
- Improvement proposals require sandbox execution (backup + rollback)

## Dashboard Endpoints

| Route | Purpose |
|-------|---------|
| `/` | Home |
| `/conversations` | Message feed |
| `/channels` | Channel adapter status |
| `/config` | Config editor |
| `/memory` | Memory search |
| `/tools` | Tool registry + approval |
| `/automations` | Automation rules |
| `/activity` | Proactive engine log |
| `/personas` | Persona management |
| `/routing` | Knowledge domain routing |
| `/self-mod` | Improvement proposals |
| `/forge` | Forge integration |
| `/identity` | Identity files |
| `/feedback` | Feedback stats |
| `/tasks` | Task management |
| `/timers` | Timer management |
| `/logs` | System logs |
| `/health` | JSON health status |
| `/ws/chat` | WebSocket chat |

## Anti-Patterns

- Do NOT use `os.path` — use `pathlib.Path`
- Do NOT use bare `except:` — catch specific exceptions
- Do NOT use `print()` for logging — use `logging`
- Do NOT call sync DB from async — use `asyncio.to_thread()`
- Do NOT bypass sandbox for config changes — always backup first
- Do NOT hardcode config paths — use `ConfigManager`
