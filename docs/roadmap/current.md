# Animus Roadmap

**Project**: Animus — Personal AI exocortex / Mind-class system  
**Classification**: Flagship  
**Version**: 2.3.0 (migrating to v2.1 baseline)  
**Last updated**: 2026-06-18  
**Next review**: 2026-07-18  

---

## Current State

- Core/Forge/Quorum/Bootstrap packages operational
- Kernel extraction complete (172 files, 55K LOC)
- Truth baseline: 8/8 passing ✅
- v2.1 Mind-class architecture **committed** (ADL-20260618-001)
- 20 canonical schemas extracted to `packages/contracts/`
- 8 dead packages archived to `packages/_archive/`
- Root architecture dirs scaffolded: `apps/`, `modules/`, `contracts/`, `database/`, `infra/`, `evidence/releases/`, `adrs/`

---

## Milestones

### Phase 0: Foundation (COMPLETE)
- [x] Truth baseline deployed and passing
- [x] ADL entry for v2.1 commitment
- [x] Dead packages archived
- [x] Version strings aligned
- [x] ADR-001 recorded

### Phase 1: Schema Integration (Q3 2026)
- [ ] Port 20 extracted schemas into `packages/types/` or `packages/contracts/`
- [ ] Build schema validation layer
- [ ] Add JSON Schema CI gate
- [ ] Document schema usage patterns

### Phase 2: Durable Core (Q3-Q4 2026)
- [ ] Implement object registry + event ledger + bitemporal projections
- [ ] Default to PostgreSQL, SQLite as projection option
- [ ] Wire `packages/kernel/` into the durable core model
- [ ] Add traceability linter (requirement-to-test mapping)

### Phase 3: Evidence & Governance (Q4 2026)
- [ ] Build `scripts/assemble_evidence_bundle.py`
- [ ] Create `evidence/releases/` structure with mandatory reports
- [ ] Integrate adversarial test harness (property-based + fault-injection)
- [ ] Implement policy decision point + governance plane

### Phase 4: Public/Private Split (Q1 2027)
- [ ] Initialize private repo for owner-specific data
- [ ] Move synthetic fixtures to public repo
- [ ] Document boundary and migration path

### Phase 5: Mind Action Pack Integration (Q1-Q2 2027)
- [ ] Port `.claude/agents/`, `.claude/skills/`, `.claude/rules/` from artifacts
- [ ] Validate against current operating model
- [ ] Document agent runtime expectations

---

## Prioritized Next Actions

1. **Port schemas** — move `packages/contracts/*.schema.json` into proper importable package
2. **Build schema compiler** — validate all current code objects against v2.1 schemas
3. **PostgreSQL spike** — verify `packages/kernel/` PostgresBackend meets durable core needs
4. **Evidence bundle script** — start with minimal manifest + build_provenance only

---

## Blockers

- None. Migration is unblocked.

## Definition of Done (Phase 1)

- All 20 schemas importable from Python
- CI validates JSON Schema compliance
- `docs/ARCHITECTURE.md` updated to reflect plane/module model
