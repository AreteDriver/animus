# Packages

> Animus is a monorepo of independently-installable packages. Each solves one problem and can be used on its own.

---

## Overview

```
┌─────────────────────────────────────────┐
│           INTERFACE LAYER               │
│   PWA · Bootstrap Dashboard · API       │
├─────────────────────────────────────────┤
│           COGNITIVE LAYER               │
│   Forge (orchestration) · Quorum (coord)│
├─────────────────────────────────────────┤
│           MEMORY LAYER                  │
│   Core (episodic/semantic/procedural)   │
├─────────────────────────────────────────┤
│           CORE LAYER                    │
│   Kernel (budget, executor, sandbox)    │
│   Contracts (canonical schemas)         │
│   Types (shared interfaces)             │
└─────────────────────────────────────────┘
```

---

## Package Directory

| Package | Import | Purpose | Tests | Coverage |
|---|---|---|---|---|
| [Core](core/README.md) | `import animus` | Personal AI exocortex — memory, CLI, integrations | 2,865 | 97% |
| [Forge](forge/README.md) | `import animus_forge` | Multi-agent workflow orchestration | 10,304 | 97% |
| [Bootstrap](bootstrap/README.md) | `import animus_bootstrap` | Install daemon, wizard, dashboard | 2,048 | 97% |
| [Quorum](quorum/README.md) | `import convergent` | Decentralized agent coordination | 961 | 97% |
| [Kernel](kernel/README.md) | `import animus_kernel` | Autonomous builder engine (standalone) | — | — |
| [Types](types/README.md) | `import animus_types` | Shared schemas and type definitions | — | — |
| [PWA](pwa/README.md) | N/A | Progressive web app interface | — | — |
| [Contracts](contracts/README.md) | N/A | Canonical JSON schemas for v2.1 | — | — |

---

## Installation

Each package can be installed independently:

```bash
# Install shared types first (local sibling, not on PyPI)
pip install -e packages/types/

# Then any combination
pip install -e packages/core/
pip install -e packages/forge/
pip install -e packages/quorum/
pip install -e packages/bootstrap/
pip install -e packages/kernel/
```

See [Getting Started → Installation](../getting-started/installation.md) for detailed per-package setup.

---

## Package READMEs

Each package directory above links to the package's own documentation lane. The canonical README lives in `packages/<name>/README.md` and is auto-synced here.

---

## Dependency Graph

```
types ←── core
      ←── forge
      ←── quorum
      ←── bootstrap
      ←── kernel

forge ←── core (optional, for memory)
bootstrap ←── core, forge (optional)
pwa ←── bootstrap (API client)
contracts ←── all (schema validation)
```

*Arrows point from dependency to dependent.*
