# Animus

**Multi-agent orchestration framework with budget controls, quality gates, and checkpoint/resume.**

![CI](https://github.com/AreteDriver/animus/workflows/CI/badge.svg)
![CodeQL](https://github.com/AreteDriver/animus/workflows/CodeQL%20Security%20Scan/badge.svg)
![Security](https://github.com/AreteDriver/animus/workflows/Security/badge.svg)
[![PyPI - convergentAI](https://img.shields.io/pypi/v/convergentAI?label=convergentAI&color=blue)](https://pypi.org/project/convergentAI/)
![License](https://img.shields.io/github/license/AreteDriver/animus)

Animus coordinates AI agents across complex workflows — with the operational discipline of a manufacturing line. Every agent has a token budget. Every workflow has a cost ceiling. If a pipeline fails at step 4 of 6, it restarts at step 4, not step 1. Inspired by the Toyota Production System: make cost visible, make waste impossible to ignore.

Four independently-installable packages. 13,700+ tests. Zero vendor lock-in (Claude, OpenAI, Ollama).

**[Architecture](docs/ARCHITECTURE.md)** | **[Roadmap](docs/ROADMAP.md)** | **[Whitepaper](docs/whitepaper.pdf)**

---

## The Architecture

Three-layer stack. Each layer solves exactly one problem and is independently useful.

```
┌─────────────────────────────────────────┐
│           INTERFACE LAYER               │
│   Desktop · Mobile · Wearable · API     │
├─────────────────────────────────────────┤
│           COGNITIVE LAYER               │
│   Reasoning · Forge · Quorum            │
├─────────────────────────────────────────┤
│           MEMORY LAYER                  │
│   Episodic · Semantic · Procedural      │
├─────────────────────────────────────────┤
│           CORE LAYER                    │
│   Identity · Security · Ethics          │
└─────────────────────────────────────────┘
```

---

## Subsystems

### Forge — Workflow Orchestration Engine

Production orchestration for AI agent pipelines. Define workflows in YAML, assign token budgets per agent, set quality gates, and checkpoint state to SQLite for automatic resume on failure. Supports 10 agent archetypes, streaming execution logs, MCP tool execution, and consensus voting.

[`packages/forge/`](packages/forge/) | `import animus_forge`

### Quorum — Agent Coordination Protocol

Decentralized multi-agent coordination without a supervisor bottleneck. Agents read a shared intent graph and self-adjust based on stability scores — no inter-agent messaging required. Includes triumvirate voting, flocking behaviors, and optional Rust PyO3 backend for performance.

[`packages/quorum/`](packages/quorum/) | `import convergent` | [PyPI: convergentAI](https://pypi.org/project/convergentAI/)

### Core — Personal AI Assistant

Persistent memory (episodic, semantic, procedural via ChromaDB), 40+ CLI commands, integrations (Google Calendar, Todoist, filesystem, webhooks), and a cognitive layer supporting Anthropic, OpenAI, and Ollama with native tool use.

[`packages/core/`](packages/core/) | `import animus`

### Bootstrap — System Daemon

One-command install, Rich-based onboarding wizard, FastAPI+HTMX ops dashboard at localhost:7700, systemd/launchd service management. Deploys Animus on new machines with zero manual configuration.

[`packages/bootstrap/`](packages/bootstrap/) | `import animus_bootstrap`

---

## Core Principles

- **Persistence** — context accumulates across sessions, devices, and years
- **Sovereignty** — cryptographic ownership, local-first by default
- **Loyalty** — aligned to you, not a platform's incentives
- **Portability** — moves with you across all devices
- **Model independence** — swap models without losing memory or context

---

## Quickstart

```bash
git clone https://github.com/AreteDriver/animus && cd animus
pip install -e packages/core -e packages/forge -e packages/quorum
pytest packages/core/tests/ packages/quorum/tests/ -q  # 3,000+ tests in seconds
```

To run the Forge test suite (must run from its package directory):

```bash
cd packages/forge && pytest tests/ -q  # 8,800+ tests
```

### Run the CLI

```bash
python -m animus  # Interactive agent with memory, tools, streaming
```

### MCP Server (Claude Code integration)

```bash
pip install animus[mcp]
python -m animus.mcp_server  # 10 tools: memory, tasks, workflows, self-improve
```

Add to `~/.claude/mcp.json`:
```json
{
  "mcpServers": {
    "animus": {
      "command": "python",
      "args": ["-m", "animus.mcp_server"]
    }
  }
}
```

---

## Usage

### Define a workflow in YAML

```yaml
# workflows/code-review.yml
name: code-review
agents:
  - role: researcher
    model: claude-sonnet-4-20250514
    budget: 4000
    task: "Analyze the codebase structure and identify patterns"
  - role: reviewer
    model: claude-sonnet-4-20250514
    budget: 8000
    task: "Review code for correctness, security, and maintainability"
    depends_on: [researcher]
gates:
  - after: reviewer
    check: quality_score >= 0.8
```

### Run it

```bash
# Via CLI
cd packages/forge
animus-forge run workflows/code-review.yml

# Via API (Forge runs as a systemd service on port 8000)
curl -X POST http://localhost:8000/api/v1/workflows/run \
  -H "Content-Type: application/json" \
  -d '{"workflow": "code-review", "params": {"target": "./src"}}'
```

### Use Quorum for agent coordination

```python
from convergent import Intent, IntentGraph

graph = IntentGraph()

# Two agents register their intents
graph.register(Intent(
    agent_id="researcher",
    action="analyze",
    provides=["codebase_map"],
    stability=0.9
))
graph.register(Intent(
    agent_id="reviewer",
    action="review",
    requires=["codebase_map"],
    stability=0.7
))

# Find conflicts and resolve them without a supervisor
overlaps = graph.find_overlapping(agent_id="reviewer")
```

### Self-Improvement

Animus can analyze and improve its own codebase:

```bash
# CLI: analyze and improve
cd packages/forge
gorgon self-improve run --provider ollama --path /path/to/project

# Analyze only (no changes)
gorgon self-improve analyze --focus security

# Record feedback for the reflection loop
animus-bootstrap feedback add up -m "Good response" -c "Accurate and concise"
animus-bootstrap feedback add down -c "Wrong answer, hallucinated API"

# Trigger reflection (reads feedback, updates LEARNED.md)
animus-bootstrap reflect

# View feedback stats
animus-bootstrap feedback stats
```

The self-improve pipeline: analyze → plan → safety check → sandbox test → apply → create PR. Human approval gates at every critical stage. Automatic rollback on test failure.

---

## Repository Structure

```
animus/
├── packages/
│   ├── core/                    # import animus
│   │   ├── animus/              # Identity, memory, cognitive, CLI, integrations
│   │   └── tests/               # 2,108 tests, 97% coverage
│   ├── forge/                   # import animus_forge
│   │   ├── src/animus_forge/    # Executor, agents, API, CLI, TUI, dashboard
│   │   ├── migrations/          # 16 SQL migrations
│   │   ├── workflows/           # YAML workflow definitions
│   │   └── tests/               # 8,894 tests, 97% coverage
│   ├── quorum/                  # import convergent (PyPI: convergentAI)
│   │   ├── python/convergent/   # Intent graph, voting, stigmergy, bridge
│   │   ├── src/                 # Rust PyO3 (optional performance layer)
│   │   └── tests/               # 926 tests, 97% coverage
│   └── bootstrap/               # import animus_bootstrap
│       ├── src/animus_bootstrap/ # Daemon, wizard, dashboard
│       └── tests/               # 1,748 tests, 97% coverage
├── docs/                        # Architecture, roadmap, whitepapers
└── .github/workflows/           # CI: lint, test (per-package), security, CodeQL
```

---

## Status

Active development. Architecture stable. v2.3.0 released.

| Component | Version | Tests | Coverage | Stage |
|-----------|---------|------:|:--------:|-------|
| Core | 2.3.0 | 2,108 | 97% | [Live on PyPI](https://pypi.org/project/animus-core/) — CLI, memory, MCP server |
| Forge | 1.3.0 | 8,894 | 97% | Self-improve pipeline, workflow orchestration |
| Quorum | 1.1.0 | 926 | 97% | [Live on PyPI](https://pypi.org/project/convergentAI/) |
| Bootstrap | 0.5.0 | 1,748 | 97% | Daemon + wizard + dashboard + reflection |

**Total: 13,676 tests across 4 packages.**

---

## Design Principles

**Budget-first execution.** Every agent has a token budget. Every workflow has a cost ceiling. Inspired by Toyota Production System — make cost visible, make waste impossible to ignore.

**No supervisor bottleneck.** The industry default for multi-agent coordination is a supervisor that watches everything. This burns tokens on monitoring and creates a single point of failure. Quorum replaces this with environmental awareness — agents observe shared state and independently converge, the way flocking birds coordinate without a lead bird.

**Checkpoint/resume.** All Forge workflows persist state to SQLite. If a pipeline fails at step 4 of 6, it restarts at step 4. No wasted compute.

**Provider-agnostic.** LLM calls go through a shared interface. Swap Claude for OpenAI or Ollama without touching agent code. Native tool use dispatches by provider.

**Local-first.** Your memory, your identity, your hardware. Nothing leaves unless you configure it to.

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Whitepaper (PDF)](docs/whitepaper.pdf)
- [Roadmap](docs/ROADMAP.md)
- [Whitepapers (Markdown)](docs/whitepapers/)

---

## Community

[Discord](https://discord.gg/fdzQkrt8) — Join the community

---

## License

MIT
