# CLAUDE.md — Animus

> Personal AI exocortex with industrial-grade multi-agent orchestration.

---

## What Is Animus?

Animus is a three-layer personal AI system designed for cognitive sovereignty — a persistent, private, portable AI assistant that belongs to its user, not a platform. It combines a user-facing exocortex (Core), a multi-agent orchestration engine (Forge), and a stigmergic coordination protocol (Swarm) into a single cohesive architecture.

**This is a monorepo.** Core, Forge, and Swarm are subsystems of one product, not separate projects.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│                 ANIMUS CORE                   │
│  User-facing exocortex                        │
│  Identity · Memory · Multi-device Interface   │
├──────────────────────────────────────────────┤
│                ANIMUS FORGE                   │
│  Multi-agent orchestration engine             │
│  Workflows · Budgets · Quality Gates · YAML   │
├──────────────────────────────────────────────┤
│                ANIMUS SWARM                   │
│  Stigmergic coordination protocol             │
│  Intent Graph · Stability Scoring · Resolver  │
└──────────────────────────────────────────────┘
```

### Layer Responsibilities

**Core** — The "who." Identity, persistent memory (episodic, semantic, procedural), multi-device interface (CLI, voice, desktop, mobile, CarPlay). Translates user intent into structured task definitions. Owns the conversation. This is what the user interacts with.

**Forge** — The "how." Receives situation configs from Core, spins up purpose-built agents from templates, manages execution with token/cost budgets, enforces quality gates, handles checkpoint/resume via SQLite. Declarative YAML pipelines. Provider-agnostic (Claude API, OpenAI, Ollama). This is a headless orchestration service — no user-facing UI. Has an ops/admin dashboard (FastAPI + React) for monitoring, not for end users.

**Swarm** — The "together." When Forge runs agents in parallel, Swarm prevents collisions without a centralized supervisor. Agents read/write to a shared intent graph. Each agent has an intent resolver that checks the graph before major decisions and self-adjusts to be compatible with high-stability decisions from other agents. Based on stigmergy/flocking principles — no inter-agent messaging, just shared environmental awareness. Low token overhead (read-heavy, write-light).

### Data Flow

```
User → Core (interprets intent, has memory/context)
       → Forge API (situation config + parameters)
              → Spins up agent pipeline from YAML definition
              → Agents run (sequential, parallel, or hybrid)
              → Parallel agents coordinate via Swarm intent graph
              → Quality gates validate outputs
       → Core receives results
       → Core presents to user
```

---

## Repository Structure

```
Animus/
├── CLAUDE.md                  ← You are here
├── README.md                  ← Public-facing project overview
├── docs/
│   ├── architecture.md        ← Detailed integration design
│   ├── forge-whitepaper.md    ← Orchestration layer deep-dive
│   └── swarm-whitepaper.md    ← Coordination protocol deep-dive
├── core/
│   ├── memory/                ← Episodic, semantic, procedural memory
│   │   ├── episodic.py        ← Conversation/event history
│   │   ├── semantic.py        ← Knowledge graph / facts
│   │   └── procedural.py      ← Learned patterns / workflows
│   ├── identity/              ← User profile, preferences, sovereignty model
│   └── interface/             ← CLI, voice, desktop, mobile adapters
├── forge/
│   ├── engine.py              ← Core orchestration loop
│   ├── workflows/             ← YAML pipeline definitions
│   │   └── *.yaml             ← Situation-specific pipeline configs
│   ├── agents/                ← Agent archetypes (templates, not hardcoded)
│   │   ├── base.py            ← Base agent contract
│   │   ├── researcher.py
│   │   ├── writer.py
│   │   ├── reviewer.py
│   │   └── publisher.py
│   ├── budget/                ← Token/cost management per-agent and per-workflow
│   ├── gates/                 ← Quality checkpoint definitions
│   ├── checkpoint/            ← SQLite-backed state persistence
│   └── dashboard/             ← FastAPI + React ops monitoring (admin only)
├── swarm/
│   ├── intent_graph.py        ← Shared directed graph of agent decisions
│   ├── stability.py           ← Confidence scoring for intent nodes
│   ├── resolver.py            ← Per-agent intent resolution (read graph → adjust)
│   └── models.py              ← IntentNode, ResolvedIntent, ConsumeInstead, etc.
├── shared/
│   ├── types.py               ← Cross-layer type definitions
│   ├── config.py              ← Shared configuration loading
│   └── llm.py                 ← Provider-agnostic LLM interface
├── configs/
│   ├── media_engine/          ← Gorgon Media Engine situation configs
│   │   ├── story_fire.yaml
│   │   ├── new_eden_whispers.yaml
│   │   └── holmes_wisdom.yaml
│   └── examples/              ← Reference configs for common patterns
├── tests/
│   ├── core/
│   ├── forge/
│   ├── swarm/
│   └── integration/
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## Design Principles

1. **Separation of concerns** — Core knows the user. Forge knows workflows. Swarm knows coordination. They don't bleed into each other.
2. **Situation-driven** — Forge doesn't have hardcoded pipelines. It reads YAML situation configs and spins up the right agents dynamically. New use case = new YAML file, not new code.
3. **Agents are archetypes, not singletons** — `researcher.py` is a template that gets parameterized per-situation. Multiple researcher instances can exist in one pipeline with different configs.
4. **Budget-first** — Every agent has a token budget. Every workflow has a cost ceiling. No runaway inference. This is a TPS/lean manufacturing principle: make cost visible and constrained.
5. **Checkpoint/resume** — All Forge workflows are resumable via SQLite state. If a pipeline fails at step 4 of 6, you restart at step 4, not step 1.
6. **No supervisor bottleneck** — When agents run in parallel, Swarm's intent graph replaces the expensive supervisor pattern. Agents self-coordinate by reading shared state. O(n) reads vs O(n²) messages.
7. **Provider-agnostic** — LLM calls go through `shared/llm.py`. Swap Claude for OpenAI or Ollama without touching agent code.
8. **Local-first sovereignty** — Core's memory and identity stay on the user's hardware. Data doesn't leave unless explicitly configured to.

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python (primary), TypeScript (dashboard) | Python for ML/AI ecosystem, TS for React dashboard |
| State persistence | SQLite | Portable, zero-config, sufficient for single-user system |
| Workflow definition | Declarative YAML | Non-developers can read/modify pipelines. Git-diffable. |
| Agent coordination | Stigmergy (Swarm) | Low token overhead, no supervisor bottleneck, scales linearly |
| LLM interface | Provider-agnostic wrapper | Avoid vendor lock-in. Test with Ollama, deploy with Claude API. |
| Dashboard | FastAPI + React/TypeScript | Portfolio-grade. Ops monitoring only — not user-facing. |
| Memory storage | ChromaDB + SQLite | Vector search for semantic memory, structured for episodic/procedural |
| Deployment | Docker + Docker Compose | Reproducible, single-command deployment |

---

## Naming History

This project consolidates three previously separate repositories:

| Current Name | Previous Name | What It Is |
|-------------|---------------|------------|
| Animus Core | Animus | Personal exocortex (identity, memory, interface) |
| Animus Forge | Gorgon | Multi-agent orchestration framework |
| Animus Swarm | Convergent | Stigmergic coordination protocol |

References to "Gorgon" or "Convergent" in older docs, commits, or conversations refer to Forge and Swarm respectively. The rename reflects the reality that these were always one system.

---

## Active Use Case: Gorgon Media Engine

The first production workload for the full stack. Three AI-powered YouTube channels producing autonomous content:

| Channel | Domain | Content |
|---------|--------|---------|
| Story Fire | World folklore & mythology | Narrative retellings |
| New Eden Whispers | EVE Online lore | In-universe storytelling |
| Holmes Wisdom | Sherlock Holmes | Wisdom & deduction lessons |

**Scale:** 8 languages per channel, ~480 videos/month total, fully autonomous via Core + Forge.

Each channel has a situation config in `configs/media_engine/` that defines the agent pipeline: Research → Script → Voice (TTS) → Video Assembly → QA → Publish.

---

## Development Context

**Owner:** ARETE (AreteDriver on GitHub)
**Primary dev environment:** Mac Studio (planned), currently Linux
**Target deployment:** Local-first on personal hardware
**LLM providers:** Claude API (primary), Ollama (local/dev)

### Current Status

- **Core:** Architecture documented, repo docs drafted. CLI prototype is Phase 1.
- **Forge:** Pre-Phase 2. Rename from AI-Orchestra → Gorgon completed. 8 implementation prompts scaffolded (setup, persistence, YAML, budgets, logging, CLI, agents, CI). FastAPI + React dashboard scoped. Chat/file-upload components being removed (Animus owns those).
- **Swarm:** Whitepaper complete with full architecture spec, worked convergence examples, and Rust/Python implementation details. Code is pre-implementation.

### Build Priority

1. Forge engine (the orchestration core — this enables everything else)
2. Swarm intent graph + resolver (enables parallel execution)
3. Core CLI prototype (basic interface to drive Forge)
4. Core memory layer (persistence across sessions)
5. Dashboard (monitoring/observability)
6. Core multi-device interfaces (mobile, voice, CarPlay — later phases)

---

## Code Style & Conventions

- Type hints everywhere. Use `from __future__ import annotations`.
- Pydantic models for all data contracts between layers.
- Agents implement a base contract (`forge/agents/base.py`). No freelancing.
- YAML configs validated against schemas on load. Fail fast on bad configs.
- Logging via structured logger. No print statements.
- Tests mirror source structure. Integration tests in `tests/integration/`.
- Docstrings on all public functions and classes.

---

## What NOT To Build

- **Chat UI in Forge** — Core owns all user interaction. Forge is headless.
- **File upload in Forge** — Core handles file ingestion and passes data to Forge.
- **Conversational memory in Forge** — Core's memory layer is the single source of truth.
- **A supervisor agent in Swarm** — The whole point is stigmergic coordination without supervisors.
- **Hardcoded pipelines** — Everything goes through YAML situation configs.

---

## Related Projects

| Project | Relationship |
|---------|-------------|
| BenchGoblins | Fantasy sports decision engine. Will use Forge workflows for analysis pipelines. |
| CLAUDE.md Generator | Meta-tool. Used to generate files like this one. |
| EVE_Rebellion | Python arcade game. Standalone — no Animus integration. |
| EVE_Gatekeeper | EVE Online starmap app. Potential Core integration for EVE data queries. |
