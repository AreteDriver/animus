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
| [Core](core/) | `import animus` | Personal AI exocortex — memory, CLI, integrations | 2,109 | 97% |
| [Forge](forge/) | `import animus_forge` | Multi-agent workflow orchestration | 9,720 | 97% |
| [Bootstrap](bootstrap/) | `import animus_bootstrap` | Install daemon, wizard, dashboard | 1,841 | 97% |
| [Quorum](quorum/) | `import convergent` | Decentralized agent coordination | 926 | 97% |
| [Kernel](kernel/) | `import animus_kernel` | Autonomous builder engine (standalone) | — | — |
| [Types](types/) | `import animus_types` | Shared schemas and type definitions | — | — |
| [PWA](pwa/) | N/A | Progressive web app interface | — | — |
| [Contracts](contracts/) | N/A | Canonical JSON schemas for v2.1 | — | — |

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
