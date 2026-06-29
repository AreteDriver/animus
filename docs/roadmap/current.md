# Animus Roadmap

**Project**: Animus — Personal AI exocortex / Mind-class system
**Classification**: Flagship
**Version**: 2.3.0 (core), mixed across packages (see [Build Truth](#build-truth))
**Last updated**: 2026-06-27
**Next review**: 2026-07-27

---

## Build Truth

> ⚠️ **The previous roadmap overstated completion. This version is grounded in filesystem reality.**

| Package | Version | Source Files | Tests | Runtime Status |
|---|---|---|---|---|
| `core` | 2.3.0 | 94 Python | 2,832 | CLI + memory + dashboard operational |
| `forge` | 1.9.0 | 352 Python | 10,431 | Workflow orchestration operational |
| `bootstrap` | 0.8.0 | 136 Python | 1,874 | Daemon + onboarding + dashboard operational |
| `kernel` | 0.1.0 | 189 Python | 99 | Extracted from Forge, under-tested |
| `quorum` | 1.2.0 (convergentAI) | 14 Rust + 10 Python | 961 | Rust core + Python bindings, active |
| `types` | 0.1.0 | 4 Python | 67 | Minimal but functional |
| `pwa` | — | 19 TypeScript | 0 | Frontend scaffolded, wiring TBD |
| `contracts` | — | 20 JSON schemas | 0 | Pure JSON, no runtime validation |

**Version alignment**: ❌ Mismatch. Core 2.3.0, Forge 1.9.0, Bootstrap 0.8.0, others 0.1.0–1.2.0. No unified version scheme.

**Truth baseline**: ✅ Honest. **10/11 PASS**, 1 FAIL (`version_alignment`). Previously: 8/11 with 3 FAIL (2 were false positives, 1 was real version misalignment). Test infrastructure fixed: root `pyproject.toml` now sets `pythonpath` for all sibling packages.

---

## Current State (Real)

### What Works

- **Core exocortex**: CLI (`python -m animus`), memory tiers (SQLite/ChromaDB), identity, proactive tasks
- **Bootstrap**: Install daemon, onboarding wizard, FastAPI+HTMX dashboard (`localhost:7700`), Ollama health checks
- **Forge**: Multi-agent YAML workflows, 10 archetypes, token budgets, checkpoint/resume, SQLite state
- **Quorum**: Rust intent graph + Python bindings, stigmergy coordination, signal bus, triumvirate voting
- **Kernel**: Extracted 172 files / 55K LOC from Forge, budget/executor/sandbox/resume all present
- **PWA**: TypeScript frontend scaffolded with Vite, login/chat/personas/status views, service worker
- **Types**: Sensitivity, egress, secrets dataclasses (security-focused)
- **Contracts**: 20 canonical JSON schemas (`action`, `event`, `assessment`, `memory`, `trace`, etc.)

### What Is Scaffolding Only

| Structure | Status | Evidence |
|---|---|---|
| `apps/` | ✅ README only | `apps/README.md` explains what belongs here |
| `modules/` | ✅ README only | `modules/README.md` explains boundary vs packages/ |
| `contracts/` (root) | ✅ Removed | Root dir deleted; `packages/contracts/` holds schemas |
| `database/` | ✅ README only | `database/README.md` describes Alembic + PostgreSQL plan |
| `infra/` | ✅ README only | `infra/README.md` describes Docker Compose + systemd plan |
| `evidence/releases/` | ✅ README only | `evidence/releases/README.md` describes bundle format |
| `adrs/` | ✅ 5 ADRs | ADR-001 through ADR-005 all committed |

---

## Milestones (Revised)

### Phase 0: Foundation — MOSTLY COMPLETE

- [x] Dead packages archived to `packages/_archive/` (8 packages)
- [x] ADR-001 through ADR-005 recorded in `adrs/`
- [x] 20 canonical JSON schemas extracted to `packages/contracts/`
- [x] **Truth baseline fixed** — `check_test_count` now FAILs on non-zero exit codes; `python3` fallback added
- [x] **Version alignment check implemented** — reads all 8 package versions, reports mismatches (still 7 unique versions)
- [x] **Schema importability** — 20 schemas auto-generated as Pydantic v2 models in `packages/types/`
- [x] **Root architecture dirs** — populated with READMEs or removed (root `contracts/` deleted)

**Blockers**: Phase 1 can start. Remaining Phase 0 gap: unify package versions (not urgent).

### Phase 1: Schema Integration (Q3 2026)

- [x] Port 20 JSON schemas into importable Python types in `packages/types/` (via `scripts/compile_schemas.py`)
- [x] Add `pyproject.toml` to `packages/contracts/` (now a real package)
- [ ] Build schema validation layer (JSON Schema → Pydantic or dataclass)
- [ ] Add JSON Schema CI gate to `docs-validate.py` or new CI job
- [ ] Document schema usage patterns in `docs/reference/`
- [ ] Align `packages/quorum/` name (`convergentAI` → `animus-quorum` or document exception)

### Phase 2: Durable Core (Q3–Q4 2026)

- [ ] Implement object registry in `database/` (PostgreSQL default, SQLite projection)
- [ ] Implement event ledger with bitemporal projections
- [ ] Wire `packages/kernel/` into durable core model (it has memory backends but no registry/ledger)
- [ ] Add traceability linter (requirement-to-test mapping)
- [ ] Populate `database/` with migration tooling
- [ ] Populate `infra/` with Docker Compose or systemd deployment manifests

### Phase 3: Evidence & Governance (Q4 2026)

- [ ] Build `scripts/assemble_evidence_bundle.py`
- [ ] Create release evidence bundles in `evidence/releases/`
- [ ] Integrate adversarial test harness (property-based + fault-injection) into Forge evaluation
- [ ] Implement policy decision point + governance plane
- [ ] Add ADR-002 through ADR-005 for decisions made to date

### Phase 4: Public/Private Split (Q1 2027)

- [ ] Initialize private repo for owner-specific data
- [ ] Move synthetic fixtures to public repo
- [ ] Document boundary and migration path
- [ ] Make repo public OR deploy docs to Netlify/Cloudflare Pages (see [MkDocs note](https://github.com/AreteDriver/animus/blob/main/docs/roadmap/current.md#mkdocs-deployment))

### Phase 5: Mind Action Pack Integration (Q1–Q2 2027)

- [ ] Port `.claude/agents/`, `.claude/skills/`, `.claude/rules/` from artifacts into versioned packages
- [ ] Validate against current operating model (v2.1 planes)
- [ ] Document agent runtime expectations in `docs/operators/`

---

## Prioritized Next Actions

1. **Fix truth baseline** — `truth-baseline.toml` test-count checks must fail on command error, not return 0 tests passing
2. **Add version check to truth baseline** — verify all 8 packages have aligned versions or documented exceptions
3. **Schema compiler spike** — auto-generate Python dataclasses from `packages/contracts/*.schema.json` into `packages/types/`
4. **Remove or populate empty scaffolding** — `apps/`, `modules/`, `database/`, `infra/`, `evidence/releases/`
5. **Evidence bundle MVP** — start with `scripts/assemble_evidence_bundle.py` that just collects test output + git commit + schema list

---

## Blockers

| Blocker | Impact | Resolution |
|---|---|---|
| Truth baseline false positives | High | Fix `truth-baseline.toml` test-count logic |
| Schemas not importable | High | Blocks Phase 1 and all downstream validation |
| Kernel under-tested (99 tests for 189 files) | Medium | Add coverage before durable core integration |
| GitHub CI billing blocked | Medium | All CI jobs fail; local-first validation only |
| MkDocs deployment blocked | Low | Repo is private; Pages requires Pro or public repo |

---

## Definition of Done (Phase 0 Completion)

- [x] Truth baseline fails on broken checks, not silently passes
- [ ] All 8 packages have version metadata and alignment is documented
- [x] `packages/types/` imports at least 10 of the 20 contract schemas as Python
- [x] Every root architecture directory has a `README.md` explaining its purpose or is removed
- [x] `scripts/assemble_evidence_bundle.py` exists and produces a manifest

---

## MkDocs Deployment

- **Status**: Site builds clean (0 warnings). Deployment blocked because repo is private.
- **Options**: Make repo public (fastest), upgrade GitHub Pro, or deploy to Netlify/Cloudflare Pages
- **Files**: `mkdocs.yml`, `docs/requirements-docs.txt`, `.github/workflows/docs-deploy.yml` all ready

---

## See Also

- [Architecture Overview](../architecture/overview.md) — 8-plane model
- [Package Architecture](../architecture/packages.md) — Dependency graph and version matrix
- [Decisions](../architecture/decisions/) — ADRs
- [Truth Baseline](https://github.com/AreteDriver/animus/blob/main/truth-baseline.toml) — Source of truth config (needs fix)
