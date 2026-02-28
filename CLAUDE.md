# CLAUDE.md — Animus Monorepo

> Personal AI exocortex with multi-agent orchestration and coordination protocol.

## Quick Reference

- **Version**: 2.0.0
- **Python**: >=3.10 (Core), >=3.11 (Bootstrap), >=3.12 (Forge)
- **Layout**: Multi-package monorepo under `packages/`
- **Tests**: 10,900+ across 4 packages
- **License**: MIT

## Monorepo Structure

```
animus/
├── packages/
│   ├── core/                    # Animus Core — exocortex, identity, memory, CLI
│   │   ├── animus/              # Python package: import animus
│   │   ├── tests/               # 1736 tests, 95% coverage
│   │   └── pyproject.toml
│   ├── forge/                   # Animus Forge — multi-agent orchestration (was: Gorgon)
│   │   ├── src/animus_forge/    # Python package: import animus_forge
│   │   ├── tests/               # 6731 tests, 85% coverage
│   │   ├── skills/              # Skill definitions (YAML + docs)
│   │   ├── migrations/          # 14 SQL migrations
│   │   ├── workflows/           # YAML workflow definitions
│   │   └── pyproject.toml
│   ├── quorum/                  # Animus Quorum — coordination protocol (was: Convergent)
│   │   ├── python/convergent/   # Python package: import convergent
│   │   ├── src/                 # Rust PyO3 (optional)
│   │   ├── tests/               # 906 tests, 97% coverage
│   │   └── pyproject.toml
│   └── bootstrap/               # Animus Bootstrap — install daemon, wizard, dashboard + gateway + intelligence
│       ├── src/animus_bootstrap/ # Python package: import animus_bootstrap
│       ├── tests/               # 1606 tests, 96% coverage
│       └── pyproject.toml
├── docs/
│   └── whitepapers/             # Architecture whitepapers (all 3 layers)
├── .github/workflows/           # CI: lint + test-core + test-quorum + test-forge + test-bootstrap + security
└── pyproject.toml               # Workspace root (dev scripts only)
```

## Independent Installation

Each package can be installed and used separately:

```bash
pip install -e packages/quorum/                # Just coordination protocol
pip install -e packages/forge/                 # Just orchestration engine
pip install -e packages/core/                  # Just exocortex
pip install -e "packages/bootstrap/[dev]"      # Just bootstrap (daemon + wizard + dashboard)
pip install -e "packages/quorum/[dev]" -e "packages/forge/[dev]" -e "packages/core/[dev]" -e "packages/bootstrap/[dev]"  # Everything
```

## Testing

```bash
# Individual packages
cd packages/core && pytest tests/ -v
cd packages/forge && pytest tests/ -v  # MUST run from forge dir (skills/workflows use relative paths)
cd packages/quorum && PYTHONPATH=python pytest tests/ -v
cd packages/bootstrap && pytest tests/ -v

# From root (core + quorum work from root, forge needs working-directory)
pytest packages/core/tests/ -v
PYTHONPATH=packages/quorum/python pytest packages/quorum/tests/ -v
pytest packages/bootstrap/tests/ -v
```

## Linting

```bash
ruff check packages/ && ruff format --check packages/
```

Each package has its own ruff config in its pyproject.toml.

## Package Identity

| Package | PyPI Name | Import | Dependencies |
|---------|-----------|--------|--------------|
| Core | animus | `import animus` | ollama, chromadb, pyyaml, pydantic, rich |
| Forge | animus-forge | `import animus_forge` | openai, anthropic, fastapi, convergentai, ... |
| Quorum | convergentAI | `import convergent` | zero production deps |
| Bootstrap | animus-bootstrap | `import animus_bootstrap` | typer, rich, fastapi, httpx, pydantic |

## Cross-Package Dependencies

- **Quorum** has zero deps — pure library
- **Forge** depends on Quorum (`convergentai ^1.1.0`)
- **Core** optionally depends on Forge for orchestration features
- **Bootstrap** is standalone — connects to Forge via HTTP (optional)

## Layer Overview

**Core** (`packages/core/animus/`) — User-facing exocortex. Identity, memory, tools, CLI (40+ commands), voice, integrations (Google, Todoist, filesystem, webhooks). Lightweight embedded workflow engine (forge/swarm submodules).

**Forge** (`packages/forge/src/animus_forge/`) — Production orchestration engine. Workflow executor with mixins (AI, MCP, queue, graph), provider abstraction (6 providers), persistent budget management, streaming execution logs, eval framework, MCP tool execution, consensus voting, API + CLI + TUI + dashboard.

**Quorum** (`packages/quorum/python/convergent/`) — Coordination protocol library. Intent graph, constraints, contracts, economics, versioned graph, triumvirate voting, stigmergy, flocking, signal bus, phi-weighted scoring, GorgonBridge integration, health dashboard, cycle detection, event log. Optional Rust PyO3 for performance.

**Bootstrap** (`packages/bootstrap/src/animus_bootstrap/`) — Install daemon, onboarding wizard, local dashboard, message gateway, intelligence layer, and persona system. Phase 1: one-command install, Rich-based setup wizard (8 steps), FastAPI+HTMX dashboard at localhost:7700, systemd/launchd service management, auto-updater. Phase 2: message gateway with 8 channel adapters (Telegram, Discord, Slack, Matrix, WhatsApp, Signal, Email, WebChat), cross-channel sessions, cognitive backends (Anthropic/Ollama/Forge), middleware (auth/ratelimit/logging). Phase 3: intelligence layer with memory integration (SQLite FTS5 + ChromaDB/Animus stubs), tool executor (8 built-in tools + MCP bridge + permission system), proactive engine (scheduler, quiet hours, 3 built-in checks), automation pipeline (triggers/conditions/actions with SQLite persistence), IntelligentRouter (memory-enriched + tool loop), intelligence dashboard (/tools, /automations, /activity). Phase 4: persona & voice layer with PersonaEngine (registry + channel routing), VoiceConfig (6 presets + time shifts), KnowledgeDomainRouter (9 domains), ContextAdapter (time/channel/mood), SQLite persona persistence, persona dashboard (/personas, /routing). Runtime wiring (AnimusRuntime orchestrator, lifespan, health endpoint). Native Anthropic tool_use (CognitiveResponse, ToolCall, multi-turn cognitive loop).

## Key Files

### Core
- `packages/core/animus/__main__.py` — CLI entry point (prompt-toolkit)
- `packages/core/animus/cognitive.py` — LLM interface (Ollama/Anthropic/OpenAI/Mock)
- `packages/core/animus/memory.py` — Episodic/semantic/procedural (ChromaDB)
- `packages/core/animus/forge/engine.py` — Lightweight sequential orchestration
- `packages/core/animus/swarm/engine.py` — Lightweight parallel orchestration

### Forge
- `packages/forge/src/animus_forge/api.py` — FastAPI app
- `packages/forge/src/animus_forge/cli/main.py` — Typer CLI
- `packages/forge/src/animus_forge/workflow/executor_core.py` — Workflow executor
- `packages/forge/src/animus_forge/agents/supervisor.py` — SupervisorAgent

### Quorum
- `packages/quorum/python/convergent/intent.py` — Core Intent model
- `packages/quorum/python/convergent/triumvirate.py` — Voting engine
- `packages/quorum/python/convergent/gorgon_bridge.py` — Integration bridge

## Conventions

- Type hints throughout
- Conventional commits
- Ruff for linting/formatting (per-package configs)
- pytest for testing
- Dataclasses in Core, Pydantic in Forge, dataclasses in Quorum
