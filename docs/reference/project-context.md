# Animus — Project Context

**Classification**: Flagship  
**Version**: 2.3.0 (migrating to v2.1 baseline)  
**Owner**: AreteDriver  
**Repository**: https://github.com/AreteDriver/animus  
**Branch**: main  
**Last updated**: 2026-06-18  
**Next review**: 2026-07-18  

---

## Technology Stack

- **Language**: Python 3.11+
- **Persistence**: SQLite (current), PostgreSQL (target)
- **Web**: FastAPI (planned for public API)
- **Data**: JSON Schema, PostgreSQL bitemporal tables
- **Infra**: Linux workstation, CI via GitHub Actions

## Key Directories

| Path | Purpose |
|------|---------|
| `packages/kernel/` | Memory, tasks, context — extracted 172 files, 55K LOC |
| `packages/forge/` | Autonomous code improvement |
| `packages/quorum/` | Multi-agent orchestration |
| `packages/bootstrap/` | System initialization |
| `packages/contracts/` | 20 canonical JSON schemas for v2.1 |
| `packages/_archive/` | 8 dead packages (arbitrage, bounty, content, faucet, mev, prospector, referral, validator) |
| `apps/` | Application entry points (v2.1 scaffold) |
| `modules/` | Technical plane implementations (v2.1 scaffold) |
| `adrs/` | Architecture Decision Records |
| `docs/` | Project documentation |
| `evidence/releases/` | Evidence bundles per release |

## Current Milestone

v2.1 migration: schema integration, durable core, evidence bundles.

## Quick Links

- [ROADMAP.md](ROADMAP.md)
- [PROJECT_CHARTER.md](PROJECT_CHARTER.md)
- [CLAUDE.md](CLAUDE.md)
- [README.md](README.md)
- [ADL entry](decisions/2026-06.md)
