# Animus

*An exocortex architecture for personal cognitive sovereignty*

---

## What is this?

Animus is a framework for building a **personal AI** — one that persists, learns, and serves a single user by design.

Current AI assistants are rented. Your context exists at the discretion of platform providers. Memory is a feature that can be revoked. The relationship resets at their convenience.

Animus explores an alternative: an AI that is **yours**.

---

## The Concept

The idea of a personal guiding intelligence is ancient — daemons, familiars, advisory entities that serve one person's interests across time.

This project translates that concept into modern architecture: a persistent, private, portable AI co-pilot that extends your cognitive capacity without compromising your sovereignty.

---

## Core Principles

- **Persistence** — Context accumulates across sessions, devices, and years
- **Sovereignty** — Your data stays yours. Local-first by default.
- **Loyalty** — Aligned to you, not to a platform's incentives
- **Portability** — Moves with you: desktop, mobile, wearable
- **Growth** — Learns your patterns, priorities, and goals over time
- **Safety** — Cannot harm its user. Guardrails are user-defined but inviolable.

---

## The Stack

Animus is the user-facing layer of a multi-project architecture. Each layer is an independent, open-source project that can be used standalone or composed into the full system.

```
┌──────────────────────────────────┐
│            ANIMUS                │  You are here
│   Identity · Memory · Interface  │
│                                  │
│   The sovereign personal AI.     │
│   Knows you, serves you, is     │
│   yours.                         │
├──────────────────────────────────┤
│            GORGON                │  github.com/AreteDriver/Gorgon
│   Workflows · Budgets · Gates    │
│                                  │
│   Multi-agent orchestration.     │
│   When Animus needs to do        │
│   complex work, Gorgon           │
│   decomposes, delegates, and     │
│   quality-checks.                │
├──────────────────────────────────┤
│          CONVERGENT              │  github.com/AreteDriver/Convergent
│   Intent Graph · Stability       │
│                                  │
│   Parallel agent coordination.   │
│   Agents converge on compatible  │
│   outputs without a supervisor   │
│   bottleneck. Inspired by        │
│   flocking and stigmergy.        │
├──────────────────────────────────┤
│        INFERENCE ENGINE          │
│   Ollama · Claude API · Any LLM  │
│                                  │
│   Model-agnostic. Swap local     │
│   and cloud providers without    │
│   changing a line of code.       │
└──────────────────────────────────┘
```

### How They Connect

**Animus** is the product — the thing you talk to, the thing that remembers you, the thing that runs on your devices. It doesn't do complex work alone.

**Gorgon** is the engine room. When Animus receives a request that requires multiple steps — research a topic, draft a document, review and refine — it hands the task to Gorgon. Gorgon decomposes it into agent roles (planner, builder, tester, reviewer), manages token budgets, enforces quality gates, and checkpoints progress so nothing is lost if a step fails.

**Convergent** is the coordination protocol inside Gorgon. When multiple agents work in parallel, Convergent's intent graph and stability scoring ensure they arrive at compatible outputs — without centralized message passing. Each agent observes a shared landscape and adjusts independently. Coherent results emerge the same way flocking emerges in birds.

**The inference engine** is pluggable. Run a 70B model locally via Ollama for air-gapped sovereignty, or route to Claude API when you need frontier reasoning. Gorgon's router handles tier-based selection automatically — heavy reasoning gets the best available model, simple tasks get the fastest.

### Use Them Independently

Each project stands on its own:

| Project | Standalone Use |
|---------|---------------|
| **Animus** | Personal AI assistant with persistent memory |
| **Gorgon** | General-purpose multi-agent workflow orchestration |
| **Convergent** | Coordination primitive for any parallel agent system |

You don't need Animus to use Gorgon. You don't need Gorgon to use Convergent. But together, they form something greater than the sum.

---

## Architecture

Animus itself is a four-layer system:

```
┌─────────────────────────────────────┐
│           Interface Layer           │
│   (voice, text, wearable, desktop)  │
├─────────────────────────────────────┤
│           Cognitive Layer           │
│ (reasoning, analysis, generation)   │
│         ┌─────────────┐            │
│         │   Gorgon    │  ← orchestration for complex tasks
│         │ Convergent  │  ← parallel coordination
│         └─────────────┘            │
├─────────────────────────────────────┤
│           Memory Layer              │
│ (episodic, semantic, procedural)    │
├─────────────────────────────────────┤
│            Core Layer               │
│  (identity, security, preferences)  │
└─────────────────────────────────────┘
```

### Core Layer

The foundation. Defines *who* this Animus belongs to.

- **Identity** — Cryptographic ownership. This instance serves one user.
- **Preferences** — Communication style, priorities, boundaries
- **Security** — Encryption at rest, access control, authentication
- **Ethics config** — User-defined behavioral constraints

### Memory Layer

What makes it *yours* over time.

- **Episodic memory** — Conversations, events, decisions (what happened)
- **Semantic memory** — Facts, knowledge, learnings (what you know)
- **Procedural memory** — Workflows, habits, patterns (how you do things)
- **Active context** — Current situation, recent threads, live priorities

### Cognitive Layer

The reasoning engine — where Gorgon and Convergent plug in.

- **Model agnostic** — Swap local or cloud LLMs as needed
- **Simple requests** — Direct LLM inference, single-turn
- **Complex requests** — Routed to Gorgon for multi-agent orchestration
- **Tool use** — File access, web search, API calls, device control
- **Register translation** — Adjusts communication to context

### Interface Layer

How you interact across contexts.

- **Desktop** — Full interface, long-form work
- **Mobile** — Voice-first, quick exchanges
- **Wearable** — Minimal, ambient, notification-based
- **API** — Integrations with other tools and services

Seamless handoff: start a thought on desktop, continue on phone. Context follows you.

---

## Data Flow

```
User input (any device)
         │
         ▼
   Interface Layer ──── Captures, normalizes
         │
         ▼
     Core Layer ──────── Authenticates, applies preferences
         │
         ▼
    Memory Layer ─────── Retrieves relevant context
         │
         ▼
   Cognitive Layer ───── Reasons, generates response
    │         │
    │    (complex tasks)
    │         ▼
    │      Gorgon ─────── Decomposes → agents → quality gates
    │         │
    │     Convergent ──── Coordinates parallel agents
    │         │
    ◄─────────┘
         │
         ▼
    Memory Layer ─────── Stores new context, updates patterns
         │
         ▼
   Interface Layer ───── Delivers response
```

---

## Reference Hardware

Animus is designed to run on consumer hardware. The reference deployment:

| Machine | Role | What It Runs |
|---------|------|-------------|
| Mac Studio M4 Max 128GB | Primary workstation | Animus core, Gorgon orchestrator, 70B reasoning model |
| Mac Studio M4 Max 128GB | Inference server | Agent pool (8B-14B models), Gorgon workers |

Connected via Thunderbolt 5 using [exo](https://github.com/exo-explore/exo) for distributed inference. Combined 256GB unified memory runs frontier-class models (DeepSeek V3 671B at 4-bit) fully local, zero API cost.

Animus also runs on a single laptop with smaller models. The architecture scales down gracefully — fewer agents, smaller models, same sovereignty.

---

## What's Buildable Now vs. Aspirational

### Buildable today
- Local LLM with persistent memory (Ollama + ChromaDB)
- Desktop + mobile text interface
- Basic voice integration
- Personal knowledge base with retrieval
- Multi-agent task execution via Gorgon

### Near-term (6-12 months)
- Cross-device sync with encrypted handoff
- Improved local models approaching API quality
- Wearable integrations (existing hardware)
- Proactive notifications and scheduling

### Aspirational
- True seamless multi-device presence
- Real-time ambient awareness
- Minimal form factor (ring, glasses) with full capability
- Self-improving personalization within safety boundaries

---

## Project Status

🚧 **Early Development**

The stack is being built bottom-up:

- [x] **Convergent** — Core library complete (Rust + Python, intent graph, stability scorer)
- [ ] **Gorgon** — Core implementation in progress (contracts, workflows, budget, agents)
- [ ] **Animus** — Architecture defined, implementation follows Gorgon completion

See each project's repo for detailed status.

---

## Philosophy

> "You don't own it. You rent access."

This is the fundamental problem with current AI assistants. Your relationship with the AI — the context it has about you, the patterns it's learned, the history you've built — exists at the pleasure of corporations whose incentives may diverge from yours at any moment.

Animus is an attempt to build something different: an AI that serves you because it's *yours*, not because a company's business model temporarily aligns with your needs.

The goal isn't to replace cloud AI services entirely — they have capabilities that local systems can't match. The goal is **sovereignty**: you control the core, you own the memory, you decide what gets shared and what stays private.

---

## Related Projects

| Project | Description | Repo |
|---------|-------------|------|
| **Gorgon** | Multi-agent orchestration with budget controls, checkpoint/resume, and YAML workflows | [AreteDriver/Gorgon](https://github.com/AreteDriver/Gorgon) |
| **Convergent** | Parallel agent coordination via intent graphs and stigmergy-inspired stability scoring | [AreteDriver/Convergent](https://github.com/AreteDriver/Convergent) |

---

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Implementation Roadmap](docs/ROADMAP.md)
- [Safety & Ethics](docs/SAFETY.md)

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project draws inspiration from:
- Ancient concepts of personal guiding intelligences (daemons, familiars)
- The exocortex concept from transhumanist thought
- Lean manufacturing and Toyota Production System principles
- Open-source AI projects pushing local-first development
- Everyone building toward a future where AI serves individuals, not platforms
