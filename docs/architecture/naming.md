# Architecture Naming Standard

This document standardizes naming across Animus docs and package references.

## Canonical terms

- **Animus Core** (`packages/core`) — memory, identity, tool-use substrate.
- **Animus Forge** (`packages/forge`) — workflow orchestration, budgets, gates, checkpoint/resume.
- **Animus Quorum** (`packages/quorum`) — coordination primitive (intent graph + convergence).
- **Animus Bootstrap** (`packages/bootstrap`) — install/ops/runtime bootstrapping.

## Stack language

Use this stack order consistently:

1. **Core layer** (identity, security, guardrails)
2. **Memory layer** (episodic/semantic/procedural context)
3. **Cognitive/Execution layer** (Forge orchestration + Quorum coordination)
4. **Interface/Ops layer** (CLI/API/UI + Bootstrap service envelope)

## Style rules

- Prefer **Forge** over legacy codenames in user-facing docs.
- Prefer **Quorum is coordination, not orchestration**.
- Prefer **"four-layer stack"** in architecture overviews.
- When discussing CI/CD, use "AI workflow control plane" for Forge to avoid category confusion.
