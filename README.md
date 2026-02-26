# Animus

*An exocortex architecture for personal cognitive sovereignty.*

![CI](https://github.com/AreteDriver/animus/workflows/CI/badge.svg)
![CodeQL](https://github.com/AreteDriver/animus/workflows/CodeQL%20Security%20Scan/badge.svg)
![Security](https://github.com/AreteDriver/animus/workflows/Security/badge.svg)
[![PyPI - convergentAI](https://img.shields.io/pypi/v/convergentAI?label=convergentAI&color=blue)](https://pypi.org/project/convergentAI/)
![License](https://img.shields.io/github/license/AreteDriver/animus)

Animus is a framework for sovereign personal AI — persistent, cryptographically owned, model-agnostic, loyal to one user. Not a chatbot. A different kind of cognitive relationship: context that accumulates across sessions, devices, and years, aligned to you rather than a platform's incentives.

**[Read the whitepaper](docs/whitepaper.pdf)** | **[Architecture deep dive](docs/ARCHITECTURE.md)** | **[Roadmap](docs/ROADMAP.md)**

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

### Forge — Multi-Agent Orchestration

Production orchestration engine. Declarative YAML workflows, 10 agent archetypes, per-agent token budgets, quality gates, SQLite checkpoint/resume, streaming execution logs, MCP tool execution, and consensus voting. Built on lean manufacturing principles from the Toyota Production System. Proven in production via the Gorgon Media Engine (480 videos/month, 8 languages, zero human bottleneck).

[`packages/forge/`](packages/forge/) | `import animus_forge`

### Quorum — Agent Coordination Protocol

Decentralized coordination using stigmergy. Agents read a shared intent graph and self-adjust to be compatible with high-stability commitments — no inter-agent messaging, no supervisor bottleneck. Includes triumvirate voting, flocking behaviors, and phi-weighted stability scoring. Optional Rust PyO3 backend for performance.

[`packages/quorum/`](packages/quorum/) | `import convergent` | [PyPI: convergentAI](https://pypi.org/project/convergentAI/)

### Core — Personal Exocortex

User-facing layer. Persistent memory (episodic, semantic, procedural via ChromaDB), identity system, 40+ CLI commands, integrations (Google Calendar, Todoist, filesystem, webhooks), and a cognitive layer supporting Anthropic, OpenAI, and Ollama with native tool use.

[`packages/core/`](packages/core/) | `import animus`

### Bootstrap — System Daemon

Install daemon, setup wizard, and ops dashboard for deploying Animus on new machines. One-command install, Rich-based onboarding wizard, FastAPI+HTMX dashboard at localhost:7700, systemd/launchd service management.

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
pytest packages/core/tests/ packages/quorum/tests/ -q  # 2,600+ tests in seconds
```

To run the Forge test suite (must run from its package directory):

```bash
cd packages/forge && pytest tests/ -q  # 6,700+ tests
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

---

## Repository Structure

```
animus/
├── packages/
│   ├── core/                    # import animus
│   │   ├── animus/              # Identity, memory, cognitive, CLI, integrations
│   │   └── tests/               # 1,736 tests, 95% coverage
│   ├── forge/                   # import animus_forge
│   │   ├── src/animus_forge/    # Executor, agents, API, CLI, TUI, dashboard
│   │   ├── migrations/          # 16 SQL migrations
│   │   ├── workflows/           # YAML workflow definitions
│   │   └── tests/               # 6,731 tests, 85% coverage
│   ├── quorum/                  # import convergent (PyPI: convergentAI)
│   │   ├── python/convergent/   # Intent graph, voting, stigmergy, bridge
│   │   ├── src/                 # Rust PyO3 (optional performance layer)
│   │   └── tests/               # 906 tests, 97% coverage
│   └── bootstrap/               # import animus_bootstrap
│       ├── src/animus_bootstrap/ # Daemon, wizard, dashboard
│       └── tests/               # 1,115 tests, 94% coverage
├── docs/                        # Architecture, roadmap, whitepapers
└── .github/workflows/           # CI: lint, test (per-package), security, CodeQL
```

---

## Status

Active development. Architecture complete. Implementation in progress.

| Component | Version | Tests | Coverage | Stage |
|-----------|---------|------:|:--------:|-------|
| Core | 2.0.0 | 1,736 | 95% | Active — CLI, memory, integrations |
| Forge | 2.0.0 | 6,731 | 85% | Production deployed (systemd service) |
| Quorum | 1.1.0 | 906 | 97% | [Live on PyPI](https://pypi.org/project/convergentAI/) |
| Bootstrap | 0.5.0 | 1,115 | 94% | Active — daemon + wizard + dashboard |

**Total: 10,488 tests across 4 packages.**

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

## License

MIT
