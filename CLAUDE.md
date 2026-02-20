# CLAUDE.md — Animus Monorepo

> Personal AI exocortex with multi-agent orchestration and coordination protocol.

## Quick Reference

- **Version**: 2.0.0
- **Python**: >=3.10 (Core), >=3.12 (Forge)
- **Layout**: Multi-package monorepo under `packages/`
- **Tests**: 9,373 across 3 packages
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
│   └── quorum/                  # Animus Quorum — coordination protocol (was: Convergent)
│       ├── python/convergent/   # Python package: import convergent
│       ├── src/                 # Rust PyO3 (optional)
│       ├── tests/               # 906 tests, 97% coverage
│       └── pyproject.toml
├── docs/
│   └── whitepapers/             # Architecture whitepapers (all 3 layers)
├── .github/workflows/           # CI: lint + test-core + test-quorum + test-forge + security
└── pyproject.toml               # Workspace root (dev scripts only)
```

## Independent Installation

Each package can be installed and used separately:

```bash
pip install -e packages/quorum/                # Just coordination protocol
pip install -e packages/forge/                 # Just orchestration engine
pip install -e packages/core/                  # Just exocortex
pip install -e "packages/quorum/[dev]" -e "packages/forge/[dev]" -e "packages/core/[dev]"  # Everything
```

## Testing

```bash
# Individual packages
cd packages/core && pytest tests/ -v
cd packages/forge && pytest tests/ -v  # MUST run from forge dir (skills/workflows use relative paths)
cd packages/quorum && PYTHONPATH=python pytest tests/ -v

# From root (core + quorum work from root, forge needs working-directory)
pytest packages/core/tests/ -v
PYTHONPATH=packages/quorum/python pytest packages/quorum/tests/ -v
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

## Cross-Package Dependencies

- **Quorum** has zero deps — pure library
- **Forge** depends on Quorum (`convergentai ^1.1.0`)
- **Core** optionally depends on Forge for orchestration features

## Layer Overview

**Core** (`packages/core/animus/`) — User-facing exocortex. Identity, memory, tools, CLI (40+ commands), voice, integrations (Google, Todoist, filesystem, webhooks). Lightweight embedded workflow engine (forge/swarm submodules).

**Forge** (`packages/forge/src/animus_forge/`) — Production orchestration engine. Workflow executor with mixins (AI, MCP, queue, graph), provider abstraction (6 providers), persistent budget management, streaming execution logs, eval framework, MCP tool execution, consensus voting, API + CLI + TUI + dashboard.

**Quorum** (`packages/quorum/python/convergent/`) — Coordination protocol library. Intent graph, constraints, contracts, economics, versioned graph, triumvirate voting, stigmergy, flocking, signal bus, phi-weighted scoring, GorgonBridge integration, health dashboard, cycle detection, event log. Optional Rust PyO3 for performance.

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
