# Glossary

> Domain terms and definitions for the Animus project.

---

## A

**ADR** — Architecture Decision Record. Documents a significant design choice: context, options, decision, tradeoffs. See `docs/architecture/decisions/`.

**ADL** — Arete Decision Log. A higher-level commitment scoped to a project milestone. ADLs live alongside ADRs.

## C

**Contracts** — Canonical JSON schemas that define data structures across Animus subsystems. 20+ schemas in `packages/contracts/`.

**Core** — The personal AI operating environment package (`animus`). Handles memory, CLI, integrations, and the cognitive layer.

**Crucible** — The universal transformation framework for navigating change. Phase detection, failure taxonomy, active/receptive polarity.

## E

**Internal philosophical frame** — *Internal anchor; not used in public positioning.* The agent's self-model and constitutional principles are anchored in the philosophical framing of an external cognitive system that augments biological intelligence (persistent memory, task tracking, preference learning across sessions and devices). This term appears in agent identity files, internal code self-references, and architectural body text. See `BRANDING.md` for the public/private split.

**Operating Environment** — The public-facing framing for Animus. A Mind-class AI operating environment you own: persistent memory, multi-agent orchestration, and autonomous improvement. This is the canonical external surface; the internal philosophical frame (see above) is what informs agent identity and decisions internally.

## F

**Forge** — Multi-agent workflow orchestration engine (`animus_forge`). YAML-defined pipelines with budget controls and checkpoint/resume.

## J

**Jidoka** — Stop and fix before propagating errors. One of the TPS principles applied to software.

## K

**Kaizen** — Continuous improvement philosophy. Applied to Animus via the reflection loop and self-improvement pipeline.

**Kernel** — Autonomous builder engine (`animus_kernel`). Budget, executor, sandbox, safety, multi-agent supervision.

## M

**Mind-class** — The target architecture for Animus v2.1. A persistent, self-improving personal intelligence with memory, planning, and autonomous execution.

## P

**Poka-yoke** — Error prevention built into the system. Used in Forge's safety checks and config validation.

## Q

**Quorum** — Agent coordination protocol (`convergent`). Decentralized multi-agent coherence via shared intent graph.

## T

**TPS** — Toyota Production System. Applied to software via cost visibility, waste elimination, and quality gates.

**Truth Baseline** — Automated validation that documented claims match codebase reality. Run via `scripts/truth-baseline.py`.

**Types** — Shared schema definitions (`animus_types`). Install first as a sibling dependency.

## V

**v2.1** — The canonical executable baseline for Mind-class architecture. 8 technical planes, 22 JSON schemas, PostgreSQL durable core.

---

## See Also

- [Concepts](../getting-started/concepts.md) — Core mental models
- [Architecture → Overview](../architecture/overview.md) — System architecture
