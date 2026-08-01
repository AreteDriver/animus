# Content Accuracy Report (2026-06-27)

**Scope**: Verify documentation claims against current codebase reality
**Method**: Automated cross-reference of pyproject.toml, test counts, file structures, and internal links
**Status**: **6 issues found** — 2 critical, 2 major, 2 minor

---

## ✅ Verified Accurate

| Claim | Source | Verification | Status |
|---|---|---|---|
| Core version 2.3.0 | `CLAUDE.md`, `README.md` | `packages/core/pyproject.toml`: `version = "2.3.0"` | ✅ Accurate |
| Package names and imports | `docs/packages/README.md` | All 8 packages match `packages/` directory | ✅ Accurate |
| PyPI badges | Root `README.md` | `animus-quorum` badge matches `packages/quorum/pyproject.toml` name | ✅ Accurate |
| License MIT everywhere | All READMEs, all `pyproject.toml` | Confirmed in 8 packages | ✅ Accurate |
| v2.1 architecture committed | `docs/roadmap/current.md` | ADL-20260618-001 exists in decisions | ✅ Accurate |
| 20 canonical schemas in `contracts/` | `docs/roadmap/current.md` | `ls packages/contracts/*.schema.json` = 20 files | ✅ Accurate |
| 8 dead packages archived | `docs/roadmap/current.md` | `packages/_archive/` has 8 dirs | ✅ Accurate |

---

## 🔴 Critical Issues

### Issue 1: Root README Links to `docs/whitepaper.pdf` at Wrong Location

**File**: `README.md:15`, `README.md:265`
**Claim**: Whitepaper PDF at `docs/whitepaper.pdf`
**Reality**: File exists at `docs/whitepaper.pdf` (flat, not in `docs/reference/` as new tree expects)

**Impact**: Low — file exists and link works, but inconsistent with new tree structure.

**Fix**: Either move PDF to `docs/reference/whitepaper.pdf` or keep at root `docs/whitepaper.pdf` and accept inconsistency.

---

### Issue 2: Test Counts Understated in All Package Docs

**File**: `docs/packages/README.md`, `docs/README.md`
**Claim**: Core 2,109 | Forge 9,720 | Bootstrap 1,841 | Quorum 926 | **Total 14,596+**

**Reality** (verified by `grep -rE '^\s*(def test_|async def test_)'`):

| Package | Docs Claim | Actual | Delta |
|---|---|---|---|
| Core | 2,109 | **2,863** | +754 (understated) |
| Forge | 9,720 | **10,297** | +577 (understated) |
| Bootstrap | 1,841 | **2,048** | +207 (understated) |
| Quorum | 926 | **959** | +33 (understated) |
| **Total** | **14,596** | **~17,167** | **+1,571** |

**Impact**: Medium — understates project scale by ~11%.

**Fix**: Update all test counts to actual values.

---

## 🟠 Major Issues

### Issue 3: Architecture Overview Missing Key Packages

**File**: `docs/architecture/overview.md`
**Problem**: Describes a 4-layer conceptual architecture (Interface, Cognitive, Memory, Core) but **never mentions**:
- Forge (workflow orchestration)
- Quorum (coordination)
- Bootstrap (daemon)
- Kernel (builder engine)
- Contracts (schemas)
- Types (shared types)

Still shows aspirational interfaces (Wearable Ring, Vehicle CarPlay) that are not built.

**Impact**: High — this is the architecture doc, but it doesn't describe the actual architecture.

**Fix**: Rewrite to describe actual package-based architecture with dependency map.

---

### Issue 4: 14 Placeholder Files Are Completely Empty

**Files**: All created during link-fix commit, 0 bytes each:
- `docs/getting-started/quickstart.md`
- `docs/getting-started/installation.md`
- `docs/getting-started/concepts.md`
- `docs/contributing/setup.md`
- `docs/contributing/workflow.md`
- `docs/contributing/debugging.md`
- `docs/operators/deployment.md`
- `docs/operators/configuration.md`
- `docs/operators/monitoring.md`
- `docs/operators/troubleshooting.md`
- `docs/reference/glossary.md`
- `docs/reference/faq.md`
- `docs/architecture/packages.md`
- `docs/architecture/standards.md`

**Impact**: Medium — `docs/README.md` links to these as primary navigation targets.

**Fix**: Write minimal viable content for each. At minimum: title + one paragraph + see-also links.

---

## 🟡 Minor Issues

### Issue 5: Root README Shows Old `docs/` Structure in ASCII Tree

**File**: `README.md` (around line 227)
**Problem**: ASCII tree shows `docs/ # Architecture, roadmap, whitepapers` — accurate but could be more specific about the new tree.

**Impact**: Low — not wrong, just vague.

**Fix**: Optional — update ASCII tree to show new docs categories.

---

### Issue 6: Package Versions Vary Widely, Only Core is 2.3.0

**File**: `CLAUDE.md`, `docs/README.md`
**Claim**: "Version: 2.3.0" for the whole project
**Reality**: Per-package versions are divergent:

| Package | Version |
|---|---|
| Core | 2.3.0 |
| Forge | 1.9.0 |
| Bootstrap | 0.8.0 |
| Quorum | 1.2.0 |
| Kernel | 0.1.0 |
| Types | 0.1.0 |

**Impact**: Low — monorepos commonly have divergent package versions.

**Fix**: Optional — add a version matrix to `docs/packages/README.md` showing per-package versions.

---

## 📊 Summary

| Severity | Count | Issues |
|---|---|---|
| 🔴 Critical | 2 | Whitepaper PDF location, test counts understated |
| 🟠 Major | 2 | Architecture overview stale, 14 empty placeholders |
| 🟡 Minor | 2 | Root README tree vague, version divergence unmentioned |

---

## Recommended Fix Order

1. **Write 14 placeholder files** (highest user impact — dead links from entry point)
2. **Update test counts** in `docs/packages/README.md` and `docs/README.md`
3. **Rewrite `docs/architecture/overview.md`** to reflect actual package architecture
4. **Move or fix `docs/whitepaper.pdf`** location
5. **Optional**: Add version matrix to `docs/packages/README.md`
