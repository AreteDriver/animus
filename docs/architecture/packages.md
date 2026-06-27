# Package Architecture

> Dependency map, per-package purpose, and version matrix.

---

## Version Matrix (Verified 2026-06-27)

| Package | Import | Version | Python | PyPI | Tests |
|---|---|---|---|---|---|
| [Core](../packages/core/) | `import animus` | 2.3.0 | ≥3.10 | ✅ | ~2,863 |
| [Forge](../packages/forge/) | `import animus_forge` | 1.9.0 | ≥3.12 | ✅ | ~10,297 |
| [Bootstrap](../packages/bootstrap/) | `import animus_bootstrap` | 0.8.0 | ≥3.11 | ✅ | ~2,048 |
| [Quorum](../packages/quorum/) | `import convergent` | 1.2.0 | ≥3.10 | ✅ (as `convergentAI`) | ~959 |
| [Kernel](../packages/kernel/) | `import animus_kernel` | 0.1.0 | ≥3.11 | Local only | — |
| [Types](../packages/types/) | `import animus_types` | 0.1.0 | ≥3.10 | Local only | — |
| [PWA](../packages/pwa/) | N/A | — | Node | N/A | — |
| [Contracts](../packages/contracts/) | N/A | — | JSON | N/A | — |

## Dependency Graph

```
              ┌─────────────┐
              │    Types    │
              └──────┬──────┘
                     │
    ┌────────┬───────┼───────┬────────┐
    │        │       │       │        │
    ▼        ▼       ▼       ▼        ▼
┌───────┐ ┌──────┐ ┌────┐ ┌────────┐ ┌──────┐
│ Core  │ │Forge │ │Boot│ │ Quorum │ │Kernel│
└───┬───┘ └──┬───┘ └──┬─┘ └────────┘ └──────┘
    │        │        │
    └────────┴────────┘
              │
              ▼
        ┌───────────┐
        │ Contracts │
        └───────────┘
```

**Arrows**: Dependency → Dependent. Types is the foundation. Forge optionally depends on Core. Bootstrap optionally depends on Core and Forge.

## Installation Order

```bash
# 1. Types first (sibling dependency)
pip install -e packages/types/

# 2. Any combination
pip install -e packages/core/
pip install -e packages/forge/
pip install -e packages/quorum/
pip install -e packages/bootstrap/
pip install -e packages/kernel/
```

---

## See Also

- [Packages](../packages/) — Per-package documentation
- [Getting Started → Installation](../getting-started/installation.md)
- [Architecture → Overview](overview.md)
