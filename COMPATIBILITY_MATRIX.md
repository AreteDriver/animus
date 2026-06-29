# Animus Compatibility Matrix

**Last verified**: 2026-06-27
**Verification method**: `python -m pytest -c pyproject.toml packages/*/tests --collect-only` passes for all packages

## Packages

| Package | PyPI Name | Version | Last verified commit | Notes |
|---|---|---|---|---|
| `core` | `animus-core` | 2.3.0 | `f28ee74` | Root version; CLI + memory operational |
| `forge` | `animus-forge` | 1.9.0 | `f28ee74` | Workflow orchestration; extracted kernel in 2026-03 |
| `bootstrap` | `animus-bootstrap` | 0.8.0 | `f28ee74` | Install daemon + dashboard; depends on `core`, `types` |
| `quorum` | `convergentAI` | 1.2.0 | `f28ee74` | Rust core + Python bindings; name mismatch documented in ADR-001 |
| `types` | `animus-types` | 0.1.0 | `f28ee74` | Zero-prod-deps shared types; now includes 20 generated schema models |
| `kernel` | `animus-kernel` | 0.1.0 | `f28ee74` | Extracted from Forge 2026-03; 99 tests, needs coverage |
| `pwa` | `animus-pwa` (npm) | 0.1.0 | `f28ee74` | TypeScript frontend; no Python tests |
| `contracts` | `animus-contracts` | 0.1.0 | `f28ee74` | Pure JSON schemas; no runtime deps |

## Inter-Package Dependencies

```
core ────────► types (animus-types>=0.1.0)
forge ───────► types
bootstrap ───► core, types
kernel ──────► types (embedded copy; will unify)
quorum ──────► (none — Rust + standalone Python)
pwa ─────────► (none — TypeScript frontend)
contracts ───► (none — pure JSON)
```

## Known Incompatibilities / Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `kernel` at 0.1.0 vs `forge` at 1.9.0 | Medium | Kernel was extracted from Forge 1.9.0; they share code lineage. No breaking API changes since extraction. |
| `quorum` PyPI name `convergentAI` | Low | Name mismatch is historical (Convergent project predecessor). Package imports as `convergent` not `animus_quorum`. Documented in ADR-001. |
| `types` and `kernel` both at 0.1.0 | Low | Different packages, same version — not a conflict but confusing. |
| No SemVer enforcement across packages | Medium | Each package versions independently. Breaking changes in `types` require coordinated bumps in `core`/`forge`/`bootstrap`. |

## When to Bump Versions

- **Major bump (X.y.z)**: Breaking API change in public interface
- **Minor bump (x.Y.z)**: New feature, backward-compatible
- **Patch bump (x.y.Z)**: Bugfix, backward-compatible
- **All packages**: Only when a cross-cutting change affects multiple packages

## Unified Version Target

The roadmap Phase 0 calls for version alignment. No unified version is enforced today. Options:

1. **Keep independent** — each package evolves at its own pace (current state)
2. **Lockstep bump** — all packages share a monorepo version (e.g. 2.4.0 together)
3. **Compatibility ranges** — specify `types>=0.1.0,<1` in downstream packages (partially implemented)

Recommended: Option 3 until Phase 2 Durable Core stabilizes, then evaluate Option 2.

## Verification

Run this command to confirm the matrix is still accurate:

```bash
python scripts/truth-baseline.py
```

If `version_alignment` reports PASS, all versions match. If FAIL, update this file with the new mismatches.
