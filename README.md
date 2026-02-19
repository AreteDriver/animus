# Animus

**A personal AI exocortex with industrial-grade multi-agent orchestration.**

Animus is a three-layer system that combines a persistent personal AI interface, a dynamic multi-agent workflow engine, and a novel coordination protocol based on stigmergy — the same principle that lets birds flock without a leader.

Most personal AI projects slap memory onto a chatbot. Animus builds the full stack: from coordination primitives through orchestration to a sovereign user interface.

---

## Architecture

```
┌──────────────────────────────────────────┐
│              ANIMUS CORE                  │
│  Identity · Memory · Multi-device UI      │
│  The layer you talk to.                   │
├──────────────────────────────────────────┤
│              ANIMUS FORGE                 │
│  Workflows · Budgets · Quality Gates      │
│  The layer that gets things done.         │
├──────────────────────────────────────────┤
│              ANIMUS SWARM                 │
│  Intent Graph · Stability · Resolution    │
│  The layer that keeps agents aligned.     │
└──────────────────────────────────────────┘
```

**Core** receives your intent and translates it into structured task definitions. It maintains persistent memory (episodic, semantic, procedural) across sessions and devices. It's the only layer users interact with directly.

**Forge** is a headless orchestration engine. It receives situation configs from Core, spins up purpose-built agents from reusable templates, manages token/cost budgets, enforces quality gates, and persists state via SQLite for checkpoint/resume. Pipelines are defined in declarative YAML — new use case, new config file, no new code.

**Swarm** prevents parallel agents from colliding without a centralized supervisor. Each agent reads a shared intent graph before making decisions and self-adjusts to be compatible with high-stability commitments from other agents. No inter-agent messaging. O(n) reads instead of O(n²) messages. Based on biological stigmergy and flocking behaviors.

---

## How It Works

You tell Animus what you need. Animus figures out how to get it done.

```
You → Core: "Produce today's Story Fire episode"

Core (knows your channels, schedule, preferences)
  → Forge API: situation config + parameters

Forge spins up pipeline:
  Research → Script → Voice → Video → QA → Publish
  (parallel where possible, Swarm keeps them aligned)

Forge → Core: results
Core → You: "Episode published. Here's the summary."
```

The same system handles media production, code review pipelines, research workflows, or any multi-step task you define in YAML.

---

## Design Principles

**Situation-driven orchestration.** Forge doesn't have hardcoded pipelines. It reads YAML situation configs and dynamically assembles the right agents with the right tools. Adding a new workflow means writing a config file, not writing code.

**Budget-first execution.** Every agent has a token budget. Every workflow has a cost ceiling. Inspired by Toyota Production System — make cost visible, make waste impossible to ignore.

**No supervisor bottleneck.** The industry default for multi-agent coordination is a supervisor agent that watches everything. This is expensive (burns tokens on monitoring) and creates a single point of failure. Swarm replaces this with environmental awareness — agents observe shared state and independently converge, the way flocking birds coordinate without a lead bird.

**Checkpoint/resume.** All Forge workflows persist state to SQLite. If a pipeline fails at step 4 of 6, it restarts at step 4. No wasted compute.

**Provider-agnostic.** LLM calls go through a shared interface. Swap Claude API for OpenAI or Ollama without touching agent code.

**Local-first sovereignty.** Your memory, your identity, your hardware. Data doesn't leave unless you explicitly configure it to.

---

## Active Workload: Media Engine

The first production deployment of the full stack. Three AI-powered YouTube channels producing autonomous content:

| Channel | Domain | Content |
|---------|--------|---------|
| Story Fire | World folklore & mythology | Narrative retellings |
| New Eden Whispers | EVE Online lore | In-universe storytelling |
| Holmes Wisdom | Sherlock Holmes | Wisdom & deduction lessons |

8 languages per channel. ~480 videos/month. Fully autonomous via Core + Forge.

Each channel has a situation config that defines its agent pipeline. Same orchestration engine, different configs, completely different outputs.

---

## Repository Structure

```
Animus/
├── core/              ← Identity, memory, multi-device interface
│   ├── memory/        ← Episodic, semantic, procedural
│   ├── identity/      ← User profile, preferences
│   └── interface/     ← CLI, voice, desktop, mobile adapters
├── forge/             ← Multi-agent orchestration engine
│   ├── workflows/     ← YAML pipeline definitions
│   ├── agents/        ← Agent archetypes (researcher, writer, reviewer...)
│   ├── budget/        ← Token/cost management
│   ├── gates/         ← Quality checkpoints
│   └── checkpoint/    ← SQLite state persistence
├── swarm/             ← Stigmergic coordination protocol
│   ├── intent_graph/  ← Shared decision graph
│   ├── stability/     ← Confidence scoring
│   └── resolver/      ← Per-agent intent resolution
├── shared/            ← Cross-layer types, config, LLM interface
├── configs/           ← Situation definitions (media engine, etc.)
├── docs/              ← Architecture docs, whitepapers
└── tests/
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python (primary), TypeScript (dashboard) |
| State | SQLite (checkpoints, procedural memory), ChromaDB (semantic memory) |
| Workflows | Declarative YAML |
| LLM | Provider-agnostic (Claude API, OpenAI, Ollama) |
| Dashboard | FastAPI + React (ops monitoring, admin only) |
| Deployment | Docker + Docker Compose |

---

## Status

| Layer | Stage |
|-------|-------|
| Core | Architecture documented. CLI prototype is next. |
| Forge | Pre-Phase 2. Implementation prompts scaffolded. FastAPI backend scoped. |
| Swarm | Whitepaper complete with full architecture spec and worked examples. Pre-implementation. |

---

## Background

This project grew out of 17+ years of enterprise operations experience, including applying Toyota Production System principles to AI workflow systematization. The orchestration layer (Forge) treats multi-agent execution the way a lean manufacturing line treats production — visible budgets, quality gates at every stage, and waste elimination through checkpoint/resume.

The coordination layer (Swarm) draws from biological systems research. Traditional multi-agent coordination uses either sequential execution (safe but slow) or supervisor patterns (flexible but expensive). Swarm introduces a third option: stigmergic coordination, where agents self-organize through shared environmental state rather than direct communication.

---

## Related Projects

| Project | Relationship |
|---------|-------------|
| [BenchGoblins](https://github.com/AreteDriver/BenchGoblins) | Fantasy sports decision engine. Will use Forge workflows for analysis pipelines. |
| [EVE_Collection](https://github.com/AreteDriver/EVE_Collection) | Media Engine's "New Eden Whispers" channel draws from EVE lore. |
| [LinuxTools](https://github.com/AreteDriver/LinuxTools) | Potential future Animus integration for voice-driven tool control. |

---

## License

MIT
