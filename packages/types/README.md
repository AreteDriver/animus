# animus-types

Shared type definitions for the Animus monorepo. Zero production
dependencies. Pure library.

## Why this package exists

The monorepo has a strict cross-package dependency direction:

- **Quorum** has zero deps
- **Forge** depends on Quorum
- **Core** optionally depends on Forge for orchestration features
- **Bootstrap** is standalone — connects to Forge via HTTP

When Stage 3.C hardening introduced `Sensitivity` (in Core) and the
follow-on tier-aware dispatch needed the same enum in Forge providers,
Forge couldn't import from Core without creating a cycle. The first
solution was vendoring — a sync-by-hand duplicate.

This package consolidates those types so every package can import the
same definitions without violating the dep direction. New shared types
land here; vendored copies in Core/Forge re-export from here.

## What's here

- `Sensitivity` — four-tier disclosure classification:
  `PUBLIC | PERSONAL | CONFIDENTIAL | SECRET`

## Adding a new shared type

Only put a type here if:

1. Two or more packages need it as a literal (not just structurally), AND
2. The type has zero behavioral logic — pure data shape

If it has behavior, keep it in the originating package and export a
Protocol or a string-based interface instead.

## Install

```bash
pip install -e packages/types/[dev]
```
