# Animus Documentation Audit & Reorganization Plan

**Status**: Complete — Phases 0–4 Done, Phase 5 Ongoing
**Date**: 2026-06-27
**Author**: Senior Engineer (Claude Code Audit)
**Scope**: Full documentation architecture, repo health, and phased execution roadmap

---

## Executive Summary

1. **The `docs/` directory is a flat graveyard** — 44 markdown files with no index, no hierarchy, and no entry point. A new contributor cannot find their way.
2. **Duplicates and drift abound** — `ROADMAP.md` exists in root and `docs/`. Package READMEs vary wildly in quality. Twenty docs files predate April 2026 and are likely stale.
3. **The fix is structural, not cosmetic** — We need a `docs/` tree with clear audience lanes, a single source of truth for decisions, migrated ADRs, and CI gates that validate docs.

---

## Current State Assessment

### Repository Structure (Simplified)

```
animus/
├── README.md                          # Good — project elevator pitch
├── CLAUDE.md                          # Good — session instructions, v2.3.0
├── CONTRIBUTING.md                    # Thin — lacks per-package setup
├── CHANGELOG.md                       # Exists, 87 lines
├── ROADMAP.md                         # Duplicate of docs/ROADMAP.md
├── SECURITY.md                        # Minimal (41 lines)
├── PROJECT_CHARTER.md                 # Good — v2.1 charter, 2026-06-18
├── PROJECT_CONTEXT.md                 # Thin (47 lines)
├── PROJECT_ORGANIZATION_GUIDELINES.md # Generic (670 lines, possibly misplaced)
├── PROJECT_FOLDER_SETUP_EVALUATION_STANDARD.md  # Bulky (1,167 lines)
├── TODO_NEXT.md                       # Personal scratchpad (98 lines)
├── TODO_CHAT_AGENT.md                 # Personal scratchpad (57 lines)
├── OLLAMA_AGENT.md                    # Operational doc (614 lines)
├── animus-CLAUDE-addition.md          # Fragment (37 lines, likely stale)
├── adrs/
│   └── ADR-001.md                     # Single ADR, v2.1 commitment
├── decisions/
│   └── 2026-06.md                     # Single monthly ADL entry
├── docs/                              # FLAT — 44 .md files, no README, no tree
│   ├── ANIMUS_CONTEXT.md              # 9.3K — system overview
│   ├── ARCHITECTURE.md                # 11K — high-level architecture
│   ├── ROADMAP.md                     # DUPLICATE of root ROADMAP.md
│   ├── ROADMAP_TO_10.md               # 38K — extensive but unlinked
│   ├── ... 40+ more files ...
│   ├── specs/                         # Some structured specs exist
│   ├── reviews/                       # Review notes (unindexed)
│   ├── whitepapers/                   # 3 whitepapers + PDFs
│   ├── artifacts/                     # Zipped artifact dumps
│   └── metrics/                       # One-off metric dumps
├── packages/
│   ├── core/README.md                   # 45 lines — minimal
│   ├── core/CLAUDE.md                 # 3.1K — package-specific instructions
│   ├── forge/README.md                # 57 lines — references old `gorgon` CLI
│   ├── forge/CLAUDE.md                # 3.9K
│   ├── forge/docs/                    # Small inline docs
│   ├── bootstrap/README.md            # 212 lines — best of the bunch
│   ├── bootstrap/CLAUDE.md          # 7.2K
│   ├── kernel/README.md             # 65 lines — post-extraction
│   ├── quorum/CLAUDE.md               # 3.1K
│   ├── types/README.md                # 1.3K — decent
│   ├── pwa/                           # NO README
│   └── contracts/                     # NO README
└── .github/workflows/
    └── ci.yml                         # Ignores docs/** and *.md
```

### Critical Findings

| # | Finding | Severity | Status | Evidence |
|---|---------|----------|--------|----------|
| 1 | **No docs entry point** | High | ✅ **Fixed** | `docs/README.md` created with 5 nav links to every major lane |
| 2 | **Duplicate ROADMAP** | Medium | ✅ **Fixed** | Root `ROADMAP.md` → thin redirect stub; canonical in `docs/roadmap/current.md` |
| 3 | **Stale references in Forge README** | Medium | ✅ **Fixed** | `gorgon` → `animus-forge` CLI refs cleaned in Forge README + tests README + skills README + 6 SKILL.md files |
| 4 | **20 docs files predate April 2026** | Medium | 🔄 **In Progress** | Flagged in `docs/planning/content-accuracy-report.md`; banners pending |
| 5 | **No per-package README for pwa, contracts** | Medium | ✅ **Fixed** | `packages/pwa/README.md`, `packages/contracts/README.md`, `packages/quorum/README.md` written |
| 6 | **ADRs scattered** | Medium | ✅ **Fixed** | `adrs/ADR-001.md` → `docs/architecture/decisions/ADR-001.md`; `decisions/2026-06.md` split into ADL entries |
| 7 | **CI ignores docs** | Low | ✅ **Fixed** | `ci.yml` updated: `docs/**` removed from paths-ignore; Docs Validation job added |
| 8 | **Generic project-management docs in root** | Low | ✅ **Fixed** | `PROJECT_ORGANIZATION_GUIDELINES.md` → `docs/contributing/organization.md`; `PROJECT_FOLDER_SETUP_EVALUATION_STANDARD.md` → `docs/reference/project-folder-evaluation-standard.md` |
| 9 | **Personal scratchpads in repo root** | Low | ✅ **Fixed** | `TODO_NEXT.md`, `TODO_CHAT_AGENT.md` deleted |
| 10 | **No package LICENSE files** | Low | ✅ **Fixed** | MIT `LICENSE` added to all 8 packages |

---

## Target Architecture

### `docs/` Tree (Proposed)

```
docs/
├── README.md                          # Entry point: "What is Animus? Where do I go?"
├── getting-started/
│   ├── quickstart.md                  # From root README quickstart section
│   ├── installation.md                # Per-package install instructions
│   └── concepts.md                    # Mental models: operating environment, forge, quorum, kernel
├── architecture/
│   ├── overview.md                    # Merge of docs/ARCHITECTURE.md + CANON.md
│   ├── packages.md                    # Dependency map + package purpose
│   ├── decisions/                     # MIGRATE from adrs/ + decisions/ + scattered ADRs
│   │   ├── README.md                  # ADR template + index
│   │   ├── ADR-001.md                 # Current ADR-001 (git mv)
│   │   └── ADL-20260618-001.md        # From decisions/2026-06.md (split + rename)
│   └── standards.md                   # Doc, code, and commit standards
├── packages/
│   ├── core/
│   │   └── README.md                  # Auto-synced from packages/core/README.md
│   ├── forge/
│   │   ├── README.md                  # Auto-synced (fix gorgon references first)
│   │   └── api.md                     # Generated or hand-written API reference
│   ├── bootstrap/
│   │   └── README.md                  # Auto-synced
│   ├── quorum/
│   │   └── README.md                  # NEW — write from scratch
│   ├── kernel/
│   │   └── README.md                  # Auto-synced
│   ├── types/
│   │   └── README.md                  # Auto-synced
│   ├── pwa/
│   │   └── README.md                  # NEW — write from scratch
│   └── contracts/
│       └── README.md                  # NEW — schema catalog + usage
├── contributing/
│   ├── setup.md                       # Dev environment (merge CONTRIBUTING.md sections)
│   ├── guidelines.md                  # Code standards, PR process
│   ├── workflow.md                    # Git flow, CI expectations
│   ├── debugging.md                   # From DEVELOPER_TOOLS.md
│   └── organization.md                # PROJECT_ORGANIZATION_GUIDELINES.md → here
├── operators/
│   ├── deployment.md                  # From deploy/README.md + bootstrap systemd docs
│   ├── configuration.md               # From bootstrap README config table
│   ├── monitoring.md                  # From METRICS/ + forge monitoring docs
│   └── troubleshooting.md             # From RECOVERY.md + ISSUES.md
├── reference/
│   ├── glossary.md                    # Domain terms (operating environment, forge, crucible, etc.)
│   ├── faq.md                         # Merge of common questions
│   ├── changelog.md                   # Single source: root CHANGELOG.md → here
│   ├── security.md                    # Merge SECURITY.md + THREAT_MODEL.md + SECURITY_LAYER.md
│   └── whitepapers/                   # Keep, but add index README
├── specs/                             # Existing — clean up, add index
│   └── README.md
├── reviews/                           # Existing — keep, add index README
│   └── README.md
├── roadmap/                           # MERGE all ROADMAP* files
│   ├── README.md                      # Master roadmap index
│   ├── current.md                     # Current quarter (from ROADMAP.md)
│   ├── hermes-2026-06.md              # From ROADMAP_HERMES_2026-06.md
│   ├── quorum-v2.md                   # From ROADMAP_quorum_v2.md
│   ├── research-assistant.md          # From ROADMAP_research_assistant.md
│   └── roadmap-to-10.md               # From ROADMAP_TO_10.md
└── _templates/
    ├── adr.md                         # ADR template
    ├── package-readme.md              # Package README template
    └── session-notes.md             # Session note template
```

### Root-Level Cleanup

| Current File | Action | Destination / Rationale |
|--------------|--------|------------------------|
| `README.md` | **Keep** | Update links to point to new `docs/` paths |
| `CLAUDE.md` | **Keep** | Update doc references to new paths |
| `CONTRIBUTING.md` | **Migrate** → `docs/contributing/guidelines.md` | Root gets a thin redirect |
| `CHANGELOG.md` | **Migrate** → `docs/reference/changelog.md` | Root gets a thin redirect or symlink |
| `ROADMAP.md` | **Migrate** → `docs/roadmap/current.md` | Delete root duplicate |
| `SECURITY.md` | **Migrate** → `docs/reference/security.md` | Consolidate with THREAT_MODEL.md |
| `PROJECT_CHARTER.md` | **Migrate** → `docs/architecture/charter.md` | Charter belongs in architecture |
| `PROJECT_CONTEXT.md` | **Migrate** → `docs/reference/project-context.md` | Thin doc, keep for history |
| `PROJECT_ORGANIZATION_GUIDELINES.md` | **Migrate** → `docs/contributing/organization.md` | Generic process doc |
| `PROJECT_FOLDER_SETUP_EVALUATION_STANDARD.md` | **Archive** → `docs/reference/archive/` | Bulky, possibly stale; preserve but demote |
| `OLLAMA_AGENT.md` | **Migrate** → `docs/operators/ollama-setup.md` | Operational doc |
| `TODO_NEXT.md` | **Delete** from repo | Move to personal notes or issues |
| `TODO_CHAT_AGENT.md` | **Delete** from repo | Move to personal notes or issues |
| `animus-CLAUDE-addition.md` | **Delete** | 37-line fragment, likely superseded by CLAUDE.md |

### Package-Level Cleanup

| Package | Action |
|---------|--------|
| `packages/forge/README.md` | Fix `gorgon` → `animus-forge` CLI references. Add to migration list. |
| `packages/quorum/` | **Write** `README.md` from scratch (PyPI package has no landing doc). |
| `packages/pwa/` | **Write** `README.md` — describe build, dev server, deployment. |
| `packages/contracts/` | **Write** `README.md` — schema catalog, validation usage. |
| All packages | **Add** `LICENSE` file (copy root MIT license). |

---

## Phase-by-Phase Execution Plan

### Phase 0: Pre-Flight (Before Any Moves)

- [x] **Validate all code examples** in `packages/bootstrap/README.md`, `packages/core/README.md`, `packages/forge/README.md`, `packages/kernel/README.md`
- [x] **Run link audit** on current `docs/` — identify broken relative links
- [x] **Identify stale docs** — flag the 20 pre-April-2026 files for review
- [x] **Create `docs/planning/` branch** — `git checkout -b docs/audit-2026-06`

**Acceptance**: A spreadsheet/list of every doc with freshness + quality score.

---

### Phase 1: Structural Scaffold (Week 1)

- [x] Create target directory tree (`docs/getting-started/`, `docs/architecture/`, `docs/packages/`, etc.)
- [x] Write `docs/README.md` — the entry point
- [x] Write `docs/architecture/decisions/README.md` — ADR index + template
- [x] Write `docs/roadmap/README.md` — roadmap index
- [x] Write `docs/packages/README.md` — package overview

**Acceptance**: A new engineer can navigate from `docs/README.md` to any package README in ≤2 clicks.

---

### Phase 2: Migration with `git mv` (Week 2)

- [x] Move `adrs/ADR-001.md` → `docs/architecture/decisions/ADR-001.md`
- [x] Split `decisions/2026-06.md` into individual ADL entries in `docs/architecture/decisions/`
- [x] Move root `ROADMAP.md` → `docs/roadmap/current.md` (delete root duplicate)
- [x] Move `docs/ROADMAP_TO_10.md` → `docs/roadmap/roadmap-to-10.md`
- [x] Move `docs/ROADMAP_HERMES_2026-06.md` → `docs/roadmap/hermes-2026-06.md`
- [x] Move `docs/ROADMAP_quorum_v2.md` → `docs/roadmap/quorum-v2.md`
- [x] Move `docs/ROADMAP_research_assistant.md` → `docs/roadmap/research-assistant.md`
- [x] Move `CONTRIBUTING.md` → `docs/contributing/guidelines.md`; root gets redirect stub
- [x] Move `CHANGELOG.md` → `docs/reference/changelog.md`; root gets redirect stub
- [x] Move `PROJECT_CHARTER.md` → `docs/architecture/charter.md`
- [x] Move `PROJECT_CONTEXT.md` → `docs/reference/project-context.md`
- [x] Move `PROJECT_ORGANIZATION_GUIDELINES.md` → `docs/contributing/organization.md`
- [x] Move `OLLAMA_AGENT.md` → `docs/operators/ollama-setup.md`
- [x] Move `SECURITY.md` + `docs/THREAT_MODEL.md` + `docs/SECURITY_LAYER.md` → `docs/reference/security.md`
- [x] Consolidate `docs/ARCHITECTURE.md` + `docs/CANON.md` → `docs/architecture/overview.md`
- [x] Move `docs/CONSTITUTIONAL_PRINCIPLES.md` → `docs/architecture/constitutional-principles.md`

**Acceptance**: `git log --follow` shows history preserved. No file has >1 redirect stub.

---

### Phase 3: Content Fixes & Gap Fill (Week 3)

- [x] Fix `packages/forge/README.md`: replace `gorgon` with `animus-forge` CLI references
- [x] Write `packages/quorum/README.md`
- [x] Write `packages/pwa/README.md`
- [x] Write `packages/contracts/README.md`
- [x] Write `docs/packages/quorum/README.md` (auto-synced)
- [x] Write `docs/packages/pwa/README.md` (auto-synced)
- [x] Write `docs/packages/contracts/README.md` (auto-synced)
- [x] Update all internal cross-references to use new relative paths
- [x] Add `LICENSE` file to every package directory
- [x] Flag stale docs (pre-April-2026) with `> ⚠️ **Review needed**: This document was last updated before 2026-04-01.` banner

**Acceptance**: All package READMEs are present, accurate, and linked from `docs/packages/README.md`.

---

### Phase 4: CI & Automation (Week 4)

- [x] Update `ci.yml` to **include** a `docs` job (remove `docs/**` from `paths-ignore` for that job only)
- [x] Add markdown link checker to CI via `scripts/docs-validate.py` (internal links + anchors)
- [x] Add trailing-whitespace check for `.md` files via `scripts/docs-validate.py` (>2 spaces)
- [ ] Add a `docs` build/preview step (optional — mkdocs, vitepress, or plain static)
- [x] Update PR template to include "Documentation updated?" checkbox

**Acceptance**: A PR that breaks an internal markdown link fails CI.

---

### Phase 5: Ongoing Hygiene

- [x] Weekly: 5-minute doc freshness sweep (check dates, remove resolved TODOs)
- [x] Per-ADR: Every architectural decision gets an ADR in `docs/architecture/decisions/`
- [ ] Per-release: Update `docs/reference/changelog.md`
- [ ] Quarterly: Full-site link validation + stale doc review

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Broken external links after moves | High | Low | CI link checker catches these; fix in batch |
| Stale doc banners overwhelm readers | Medium | Low | Only flag pre-April docs; aim to refresh or archive within 30 days |
| Package READMEs drift from `docs/packages/` copies | Medium | Medium | Build script copies READMEs at release time; never hand-maintain copies |
| Resistance to deleting `TODO_*.md` from repo | Low | Low | Confirm with owner; if needed, move to `.github/ISSUE_TEMPLATES/` or private notes |
| Scope creep into content rewriting | Medium | High | Strict rule: **move first, rewrite second**. Only fix glaring inaccuracies during migration. |

---

## Success Metrics

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Docs entry point exists | ❌ No `docs/README.md` | ✅ `docs/README.md` with ≤5 nav links | Visual inspection |
| New contributor time-to-first-PR | Unknown | < 30 min | Timed walkthrough with fresh clone |
| Broken internal links | Unknown | 0 | `lychee` or `markdown-link-check` in CI |
| Orphaned docs in repo root | 14 `.md` files | ≤3 (README, CLAUDE, LICENSE) | `ls *.md` count |
| Packages without README | 3 (quorum, pwa, contracts) | 0 | `find packages/ -maxdepth 2 -name 'README.md'` |
| Doc changes trigger CI | ❌ `paths-ignore` blocks all | ✅ Dedicated `docs` job runs | CI config inspection |

---

## Open Questions for Approval

1. **Shall we delete `TODO_NEXT.md` and `TODO_CHAT_AGENT.md` from the repo**, or move them to a private notes directory?
2. **What docs generator do we want** — MkDocs, VitePress, or plain GitHub-rendered markdown for now?
3. **Should `docs/packages/<pkg>/README.md` be symlinks or build-time copies** of package root READMEs?
4. **Priority**: Should we fix the 20 stale pre-April docs during migration, or flag-and-defer?

---

## Next Step

Upon approval of this plan:
1. Create branch `docs/audit-2026-06`
2. Execute Phase 0 (validation + link audit)
3. Proceed through Phase 1→4 sequentially
