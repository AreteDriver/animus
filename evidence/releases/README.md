# evidence/releases/

Release evidence bundles — machine-readable proof that every release is safe, tested, and traceable.

## What Goes Here

Each release produces a timestamped directory (`evidence-YYYY-MM-DD-HHMMSS/`) containing:

| File | Purpose |
|---|---|
| `manifest.json` | Bundle metadata: git SHA, timestamp, version, builder identity |
| `test-results.json` | Aggregated pytest output across all packages |
| `schema-validation.json` | JSON Schema compliance report |
| `git-info.txt` | Last 5 commits, branch, dirty/clean status |
| `dependencies.lock` | `pip freeze`, `cargo tree`, `npm ls` output |
| `report.md` | Human-readable summary with pass/fail badges |

## How to Generate

```bash
python scripts/assemble_evidence_bundle.py
```

See `docs/roadmap/current.md` Phase 3 for the full evidence bundle specification.

## Owner

AreteDriver

## Status

Scaffolded — no evidence bundles yet. The `assemble_evidence_bundle.py` script is part of Phase 3 (Evidence & Governance).
