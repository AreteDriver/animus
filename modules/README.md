# modules/

Cross-package shared modules that are too small for their own package but too large to inline.

## What Goes Here

- **Logging utilities** — structured JSON logging, context propagation
- **Configuration loaders** — TOML/YAML/JSON config with environment overrides
- **Auth primitives** — JWT validation, bearer token parsing, TLS helpers
- **Middleware** — FastAPI/HTMX middleware shared across bootstrap and API surfaces
- **Testing fixtures** — synthetic data generators, mock backends, test utilities

## Boundary vs packages/

`packages/` are **independently versioned** and installable.
`modules/` are **internal shared code** — imported via relative paths or editable installs, not published to PyPI.

## Owner

AreteDriver

## Status

Scaffolded — no modules extracted yet. When a cross-cutting concern emerges in multiple packages, it gets extracted here.
