# Installation

> Install Animus packages independently. Each solves one problem and can be used on its own.
>
> Updated 2026-08-08: Pages site deploy activated (see ADL-20260808-001).

---

## Requirements

| Package | Python | Notes |
|---|---|---|
| Core | ≥3.10 | Exocortex engine |
| Forge | ≥3.12 | Orchestration (heavier deps) |
| Bootstrap | ≥3.11 | Daemon + dashboard |
| Quorum | ≥3.10 | Coordination protocol |
| Kernel | ≥3.11 | Standalone builder engine |
| Types | ≥3.10 | Shared schemas (install first) |

## Install All Packages

```bash
# Install shared types FIRST — it's a sibling dependency, not on PyPI
pip install -e packages/types/

# Then install any combination
pip install -e packages/core/
pip install -e packages/forge/
pip install -e packages/quorum/
pip install -e packages/bootstrap/
pip install -e packages/kernel/
```

## Verify

```bash
python -c "import animus; print('Core OK')"
python -c "import animus_forge; print('Forge OK')"
python -c "import animus_bootstrap; print('Bootstrap OK')"
python -c "import convergent; print('Quorum OK')"
python -c "import animus_kernel; print('Kernel OK')"
```

---

## See Also

- [Quickstart](quickstart.md) — Get running in 10 minutes
- [Concepts](concepts.md) — Understand the mental models
- [Operators → Configuration](../operators/configuration.md) — Configure after install
