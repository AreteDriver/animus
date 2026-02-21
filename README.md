# Animus

![CI](https://github.com/AreteDriver/animus/workflows/CI/badge.svg)
![CodeQL](https://github.com/AreteDriver/animus/workflows/CodeQL%20Security%20Scan/badge.svg)
![Security](https://github.com/AreteDriver/animus/workflows/Security/badge.svg)
[![PyPI - convergentAI](https://img.shields.io/pypi/v/convergentAI?label=convergentAI&color=blue)](https://pypi.org/project/convergentAI/)
![License](https://img.shields.io/github/license/AreteDriver/animus)

**Personal AI exocortex with production-grade multi-agent orchestration.**

Four independently installable packages: a personal AI interface, a workflow orchestration engine, a decentralized agent coordination protocol, and a system bootstrap daemon. 9,600+ tests. v2.0.0.

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

## What This Does

Animus breaks complex tasks into multi-agent workflows with built-in cost controls, quality gates, and checkpoint/resume. Agents coordinate through a shared intent graph instead of expensive supervisor patterns — O(n) reads instead of O(n^2) messages.

---

## Architecture

```
┌──────────────────────────────────────────┐
│              ANIMUS CORE                  │
│  Identity · Memory · Learning · CLI       │
│  import animus          1,736 tests  95%  │
├──────────────────────────────────────────┤
│              ANIMUS FORGE                 │
│  Workflows · Budgets · Quality Gates      │
│  import animus_forge    6,731 tests  85%  │
├──────────────────────────────────────────┤
│              ANIMUS QUORUM                │
│  Intent Graph · Consensus · Stigmergy     │
│  import convergent        906 tests  97%  │
└──────────────────────────────────────────┘
         + Bootstrap (daemon, wizard, dashboard)
```

**Core** — User-facing exocortex. Persistent memory (episodic, semantic, procedural via ChromaDB), identity system, 40+ CLI commands, integrations (Google Calendar, Todoist, filesystem, webhooks), and a cognitive layer supporting Anthropic, OpenAI, and Ollama with native tool use.

**Forge** — Headless orchestration engine. Declarative YAML workflows, 10 agent archetypes, per-agent token budgets, quality gates, SQLite checkpoint/resume, FastAPI backend, streaming execution logs, MCP tool execution, eval framework, and consensus voting. Deployed as a systemd service.

**Quorum** — Decentralized coordination protocol. Agents read a shared intent graph and self-adjust to be compatible with high-stability commitments — no inter-agent messaging, no supervisor bottleneck. Includes triumvirate voting, stigmergy, flocking behaviors, and phi-weighted stability scoring. Optional Rust PyO3 backend for performance. [Published on PyPI as `convergentAI`](https://pypi.org/project/convergentAI/).

**Bootstrap** — Install daemon, setup wizard, and ops dashboard for deploying Animus on new machines.

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
│   │   ├── configs/             # YAML configs
│   │   └── tests/               # 1,736 tests
│   ├── forge/                   # import animus_forge
│   │   ├── src/animus_forge/    # Executor, agents, API, CLI, TUI, dashboard
│   │   ├── migrations/          # 14 SQL migrations
│   │   ├── workflows/           # YAML workflow definitions
│   │   ├── skills/              # Agent skill definitions
│   │   └── tests/               # 6,731 tests
│   ├── quorum/                  # import convergent (PyPI: convergentAI)
│   │   ├── python/convergent/   # Intent graph, voting, stigmergy, bridge
│   │   ├── src/                 # Rust PyO3 (optional performance layer)
│   │   └── tests/               # 906 tests
│   └── bootstrap/               # import animus_bootstrap
│       ├── src/animus_bootstrap/ # Daemon, wizard, dashboard
│       └── tests/               # 287 tests
├── docs/whitepapers/            # Architecture whitepapers
└── .github/workflows/           # CI: lint, test (per-package), security, CodeQL
```

---

## Design Principles

**Budget-first execution.** Every agent has a token budget. Every workflow has a cost ceiling. Inspired by Toyota Production System — make cost visible, make waste impossible to ignore.

**No supervisor bottleneck.** The industry default for multi-agent coordination is a supervisor that watches everything. This burns tokens on monitoring and creates a single point of failure. Quorum replaces this with environmental awareness — agents observe shared state and independently converge, the way flocking birds coordinate without a lead bird.

**Checkpoint/resume.** All Forge workflows persist state to SQLite. If a pipeline fails at step 4 of 6, it restarts at step 4. No wasted compute.

**Provider-agnostic.** LLM calls go through a shared interface. Swap Claude for OpenAI or Ollama without touching agent code. Native tool use dispatches by provider.

**Local-first.** Your memory, your identity, your hardware. Nothing leaves unless you configure it to.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ (Core), Python 3.12+ (Forge), Rust (Quorum optional) |
| State | SQLite WAL (checkpoints, budgets, migrations), ChromaDB (semantic memory) |
| Workflows | Declarative YAML |
| LLM | Claude API, OpenAI, Ollama (native tool use per provider) |
| API | FastAPI (Forge backend, deployed as systemd service) |
| Coordination | PyPI: [convergentAI](https://pypi.org/project/convergentAI/) |

---

## Status

| Package | Version | Tests | Coverage | Stage |
|---------|---------|------:|:--------:|-------|
| Core | 2.0.0 | 1,736 | 95% | Active — CLI, memory, integrations |
| Forge | 2.0.0 | 6,731 | 85% | Active — deployed as systemd service |
| Quorum | 1.1.0 | 906 | 97% | Active — [live on PyPI](https://pypi.org/project/convergentAI/) |
| Bootstrap | 0.1.0 | 287 | 93% | Active — daemon + wizard |

**Total: 9,660+ tests across 4 packages.**

---

## Background

This project grew out of 17+ years of enterprise operations experience, including applying Toyota Production System principles to AI workflow systematization. The orchestration layer (Forge) treats multi-agent execution the way a lean manufacturing line treats production — visible budgets, quality gates at every stage, and waste elimination through checkpoint/resume.

The coordination layer (Quorum) draws from biological systems research. Traditional multi-agent coordination uses either sequential execution (safe but slow) or supervisor patterns (flexible but expensive). Quorum introduces a third option: stigmergic coordination, where agents self-organize through shared environmental state rather than direct communication.

---

## License

MIT
