# Architecture Overview

> Animus is a modular, local-first AI exocortex built around 8 technical planes. Each plane is independently useful and composable.

---

## System Diagram (v2.1)

```
┌─────────────────────────────────────────────────────────────┐
│                     Interface Plane                         │
│   CLI · Bootstrap Dashboard · API · MCP Server           │
├─────────────────────────────────────────────────────────────┤
│                    Cognitive Plane                          │
│   Forge (orchestration) · Quorum (coordination)            │
├─────────────────────────────────────────────────────────────┤
│                     Memory Plane                            │
│   Episodic · Semantic · Procedural · Reflection            │
├─────────────────────────────────────────────────────────────┤
│                     Identity Plane                          │
│   Self · Persona · Core Values · Preferences              │
├─────────────────────────────────────────────────────────────┤
│                    Integration Plane                        │
│   Calendar · Tasks · Filesystem · Webhooks · Tools         │
├─────────────────────────────────────────────────────────────┤
│                     Kernel Plane                            │
│   Budget · Executor · Sandbox · Safety · Resume           │
├─────────────────────────────────────────────────────────────┤
│                    Contracts Plane                          │
│   Canonical JSON schemas (20+) — actions, events,        │
│   assessments, memories, feedback                         │
├─────────────────────────────────────────────────────────────┤
│                     Types Plane                             │
│   Shared Python schemas — install first, used by all    │
└─────────────────────────────────────────────────────────────┘
```

*Verified 2026-06-27. See [Package Architecture](packages.md) for dependency graph and version matrix.*

---

## Interface Plane

How you interact with Animus. Local-first by default.

**CLI** (`python -m animus`) — Interactive agent with memory, streaming, and tool use.

**Bootstrap Dashboard** (`localhost:7700`) — FastAPI+HTMX ops UI. Daemon status, Ollama health, memory backend status, identity proposals.

**MCP Server** (`python -m animus.mcp_server`) — 10 tools exposed to Claude Code: memory read/write, task creation, workflow dispatch, self-improve triggers.

**API** — RESTful endpoints for third-party integrations. Forge runs as a service on port 8000.

---

## Cognitive Plane

The reasoning and coordination engines.

### Forge

Multi-agent workflow orchestration (`animus_forge`). Define pipelines in YAML, assign per-agent token budgets, set quality gates, and checkpoint state to SQLite for automatic resume on failure.

- **10 agent archetypes**: researcher, reviewer, writer, tester, security, etc.
- **Budget controls**: Every agent has a token ceiling. Every workflow has a cost cap.
- **Checkpoint/resume**: If a pipeline fails at step 4 of 6, it restarts at step 4.
- **Quality gates**: Threshold checks after each stage. Failures trigger rollback or retry.

### Quorum

Decentralized agent coordination (`convergent`). No supervisor bottleneck — agents read a shared intent graph and self-adjust based on stability scores.

- **Intent graph**: Agents register intents (provides/requires/stability). Conflicts resolved by overlap detection.
- **Triumvirate voting**: Three agents vote on contentious decisions.
- **Flocking**: Agents converge on consensus without central control.

---

## Memory Plane

What makes it *yours* over time.

| Type | What it stores | Backend |
|---|---|---|
| **Episodic** | Conversations, events, decisions | SQLite / ChromaDB |
| **Semantic** | Facts, preferences, relationships | ChromaDB / Weaviate |
| **Procedural** | Workflows, patterns, how you work | SQLite + learned embeddings |
| **Reflection** | Feedback loop outcomes, improvement history | SQLite |

**Implementation**: Default is SQLite for portability. ChromaDB and Weaviate are optional backends via `memory.backend` config.

---

## Identity Plane

Defines *who* this Animus belongs to and what it cannot do.

- **Self** — The persistent identity file. What Animus calls itself, its values, its relationship to the user.
- **Persona** — Adjustable communication style (formal, casual, technical).
- **Core Values** — Immutable. Stored in `CORE_VALUES.md`. Cannot be modified by self-improvement.
- **Preferences** — Learned over time. Communication style, priorities, interaction patterns.

**Guardrails**: Small changes (<20% of file size) are written directly. Larger changes require dashboard approval. Safety rules are immutable — learned behavior cannot override them.

---

## Integration Plane

External tool connections. All optional, user-configured.

| Integration | Purpose | Status |
|---|---|---|
| Google Calendar | Event read/write | Active |
| Todoist | Task sync | Active |
| Filesystem | Local file access | Active |
| Webhooks | Event callbacks | Active |
| Ollama | Local LLM inference | Active |
| Anthropic / OpenAI | Cloud LLM APIs | Optional |

---

## Kernel Plane

The autonomous builder engine (`animus_kernel`). Extracted from Forge for standalone use.

- **Budget Manager** — Token accounting, spend tracking, cost ceiling enforcement.
- **Executor** — Runs workflow steps, handles async DAG scheduling.
- **Sandbox** — Validates changes before application. Test-driven safety.
- **Safety Checks** — Config validation, guardrail enforcement, forbidden skill blocking.
- **Resume** — State persistence to SQLite. Workflow recovery on failure.

---

## Contracts Plane

Canonical JSON schemas (`packages/contracts/`). 20+ schemas define data structures across all subsystems.

Key schemas: `Action`, `Event`, `Assessment`, `Memory`, `Feedback`, `Identity`, `Task`, `Workflow`.

Every package validates inputs and outputs against these schemas. The Contracts package has no runtime dependencies — it is pure JSON.

---

## Types Plane

Shared Python type definitions (`animus_types`). Install this package **first** — all other packages depend on it as a local sibling.

Contains: dataclasses, enums, protocol definitions, and shared constants used across Core, Forge, Bootstrap, Quorum, and Kernel.

---

## Data Flow

### Standard Interaction

```
User input (CLI / Dashboard / MCP)
         │
         ▼
┌─────────────────────┐
│   Identity Plane    │  ← Load persona, apply preferences
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    Memory Plane     │  ← Retrieve relevant context
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Cognitive Plane   │  ← Forge/Quorum reason, generate
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    Memory Plane     │  ← Store new context, update patterns
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Interface Plane   │  ← Deliver response
└─────────────────────┘
```

### Self-Improvement Flow

```
Feedback collected (up/down votes, comments)
         │
         ▼
┌─────────────────────┐
│   Reflection Loop   │  ← Aggregate, identify patterns
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Kernel Plane       │  ← Plan changes, safety check
│   (Sandbox)          │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Apply Changes      │  ← Update docs, code, or identity
│   (Approval Gate)    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Log Change         │  ← Transparent, reversible
└─────────────────────┘
```

---

## Security Model

### Threat Model

| Threat | Mitigation |
|---|---|
| Unauthorized config access | Config files chmod 600 |
| Malicious self-modification | Approval gate for >20% file changes |
| Data exfiltration | Local-first by default; no telemetry |
| Model prompt injection | Guardrails, input sanitization |
| API key leakage | Stored in config, never transmitted |

### Privacy Guarantees

1. **No telemetry without consent** — Nothing phones home by default
2. **Local memory** — SQLite/ChromaDB on your machine
3. **Explicit sharing only** — Data leaves device only when you choose
4. **Audit trail** — You can see everything that happened

See [Reference → Security](../reference/security.md) for full threat model.

---

## Extensibility

### Plugin Architecture

Animus supports extensions for:
- Additional memory backends
- Custom tool integrations
- New agent archetypes in Forge
- Specialized cognitive modules

### API Layer

RESTful API on port 8000 (Forge service):
- Workflow submission and monitoring
- Memory read/write
- Task and feedback management
- MCP tool execution

All API access subject to Identity Plane authentication and guardrails.

---

## Design Principles

**Budget-first execution.** Every agent has a token budget. Every workflow has a cost ceiling. Inspired by Toyota Production System — make cost visible, make waste impossible to ignore.

**No supervisor bottleneck.** Quorum replaces centralized supervision with environmental awareness. Agents observe shared state and independently converge.

**Checkpoint/resume.** All Forge workflows persist state to SQLite. If a pipeline fails at step 4 of 6, it restarts at step 4. No wasted compute.

**Provider-agnostic.** LLM calls go through a shared interface. Swap Claude for OpenAI or Ollama without touching agent code.

**Local-first.** Your memory, your identity, your hardware. Nothing leaves unless you configure it to.

---

## See Also

- [Package Architecture](packages.md) — Dependency graph and version matrix
- [Decisions](decisions/) — Architecture Decision Records (ADRs)
- [Reference → Glossary](../reference/glossary.md) — Domain terms
- [Roadmap](../roadmap/current.md) — Future plans
