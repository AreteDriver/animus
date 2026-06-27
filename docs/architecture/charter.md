# Animus Project Charter

**Version**: 1.0  
**Date**: 2026-06-18  
**Classification**: Flagship  
**Owner**: AreteDriver  

---

## Purpose

Build a **Mind-class AI exocortex** — a persistent, self-improving personal intelligence system that operates across sessions with memory, planning, and autonomous execution capabilities. Animus is the flagship project of the portfolio and serves as the substrate for all other AI tooling.

## Scope

### In Scope
- Kernel tier (memory, tasks, context management)
- Forge tier (autonomous code improvement)
- Schema-driven architecture (8 technical planes, 22 JSON schemas)
- Durable core (PostgreSQL object registry, event ledger, bitemporal projections)
- Evidence bundles and adversarial test harness
- Public/private repository split

### Out of Scope
- General-purpose consumer product (internal/personal use only)
- Multi-tenant SaaS deployment
- Real-time voice/video processing
- Blockchain integration

## Success Criteria

1. Truth baseline passes 8/8 checks continuously
2. All 20 canonical schemas validate against codebase
3. PostgreSQL durable core operational with traceability linter
4. Evidence bundles assembled automatically per release
5. v2.1 Mind-class architecture fully implemented and documented

## Constraints

- Python 3.11+ only
- Open-source with private data separation
- No external AI API dependencies for core runtime
- Must run on standard Linux workstation

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Schema drift during migration | High | Auto-validation CI gate, ADR-001 |
| PostgreSQL complexity | Medium | SQLite fallback, incremental port |
| Scope creep to general product | Medium | Charter scope boundary, personal-use mandate |

## Authority

- **Decision maker**: AreteDriver
- **Architecture authority**: ADL entries (ADL-20260618-001 for v2.1 commitment)
- **Change control**: ADR process in `adrs/`

## Definition of Done

- Feature implemented with tests
- Schema updated and validated
- Documentation reflects change
- Truth baseline still passes
- Evidence bundle updated if applicable
