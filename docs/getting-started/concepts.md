# Core Concepts

> Mental models for understanding the Animus system.

---

## Exocortex

An **exocortex** is an external cognitive system that augments your biological brain. Animus stores memories, tracks tasks, learns your preferences, and persists context across sessions, devices, and years. It is not a chatbot — it is a persistent intelligence layer that accumulates knowledge about you over time.

## Forge

**Forge** is the multi-agent orchestration engine. It runs YAML-defined workflows, assigns token budgets per agent, sets quality gates, and checkpoints state to SQLite for automatic resume on failure. Think of it as a manufacturing line for AI pipelines.

## Quorum

**Quorum** is the coordination protocol. Agents read a shared intent graph and self-adjust based on stability scores — no inter-agent messaging required. It enables decentralized multi-agent coherence without a supervisor bottleneck.

## Kernel

**Kernel** is the autonomous builder engine. It handles budget management, workflow execution, sandbox validation, safety checks, and multi-agent supervision. Extracted from Forge for standalone use.

## Bootstrap

**Bootstrap** is the install daemon, onboarding wizard, and local dashboard. One command (`animus-bootstrap install`) sets up dependencies, registers a system service, and opens the ops UI at `localhost:7700`.

## Contracts

**Contracts** are canonical JSON schemas (20+) that define data structures across all Animus subsystems: actions, events, assessments, memories, and more. Every package validates against these schemas.

## Types

**Types** is the shared schema package. Install it first so other packages can resolve `animus-types` as a local dependency.

---

## See Also

- [Architecture → Overview](../architecture/overview.md) — System architecture
- [Packages](../packages/README.md) — Per-package documentation
- [Reference → Glossary](../reference/glossary.md) — Full term definitions
