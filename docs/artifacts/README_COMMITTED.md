# Animus Downloaded Artifacts — COMMITTED FOR MIGRATION

**Status**: MIGRATION IN PROGRESS. These artifacts describe the v2.1 canonical executable
baseline and v2.2 implementation roadmap that **is now the target architecture** for Animus.

**Decision**: ADL-20260618-001 — Commit to v2.1 Mind-class architecture (see below).

**Current codebase trajectory**: Core/Forge/Quorum/Bootstrap + Kernel extraction.
**Target trajectory**: 8 technical planes, 22 JSON Schemas, PostgreSQL durable authority,
adversarial test harness, evidence bundles, public/private repo split.

**Rationale**: The current ad-hoc trajectory (separate packages with divergent versions,
SQLite-only persistence, no unified event model, no schema layer) cannot support a
"Mind"-class system as defined in `The_Culture_Mind_Daily_Function_and_Tooling.pdf`.
A Mind requires persistent context, world model, tool orchestration, uncertainty discipline,
consent/boundaries, developmental intelligence, and independent ethical judgment. The v2.1
baseline was designed to provide exactly these capabilities through:

- **Object registry + event ledger + bitemporal projections** → persistent context
- **8 technical planes** → world model + tool orchestration
- **22 canonical schemas** → uncertainty discipline + legible intervention
- **Policy decision point + governance plane** → consent and boundaries
- **Adversarial test harness + evidence bundles** → developmental intelligence
- **Consciousness quorum + constitutional principles** → independent ethical judgment

**Migration phases**:
1. Scaffold root directories (`apps/`, `modules/`, `contracts/`, `database/`, `infra/`)
2. Extract 22 schemas from artifact into `packages/contracts/` or new directory
3. Align versions to single canonical source (`packages/core` → v2.3.0 baseline)
4. Archive dead packages to `packages/_archive/`
5. Migrate SQLite-dominant packages to PostgreSQL projection model
6. Integrate Mind Action Pack agents/skills/rules into operating model
7. Build evidence bundle automation (`scripts/assemble_evidence_bundle.py`)

**Last reviewed**: 2026-06-18
