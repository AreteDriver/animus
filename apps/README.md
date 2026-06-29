# apps/

Standalone applications that consume the Animus monorepo packages.

## What Goes Here

- **CLI applications** (`animus-cli`, future citizen-zero interface)
- **Dashboard frontends** (FastAPI + HTMX or React-based admin UIs)
- **Mobile / PWA wrappers** (installable, offline-capable clients)
- **Integration daemons** (calendar sync, webhook listeners, bridge services)

## Boundary vs packages/

`packages/` contains **library code** — reusable, installable, importable.
`apps/` contains **executable targets** — they depend on packages but are not themselves reusable libraries.

## Owner

AreteDriver

## Status

Scaffolded — no applications yet. When an app is ready, it gets its own directory here with a `pyproject.toml` or `package.json`.
