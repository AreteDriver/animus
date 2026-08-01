# REL-02 — Kernel and Contracts Packaging Repair

## Scope

Repair the critical packaging discrepancies identified in REL-00:

- D1: `animus-kernel` missing runtime dependency on `animus-types` and vendoring a shadow copy.
- D2: `animus-contracts` missing runtime dependency on `animus-types` and shipping no JSON schemas.
- D5: `animus-core[all]` referencing the wrong distribution name (`animus` instead of `animus-core`).

## Changes

### `animus-kernel`

- `packages/kernel/pyproject.toml`: added `animus-types>=0.1.0,<1` to runtime dependencies.
- Removed the vendored `packages/kernel/src/animus_types/` directory so kernel honestly depends on the sibling package.
- Updated `publish-kernel.yml` and `publish-kernel-testpypi.yml` to install `animus-types` in the same transaction as the kernel editable install.
- Updated `ci.yml` `test-kernel` job to install `animus-types` alongside kernel.

### `animus-contracts`

- `packages/contracts/pyproject.toml`: added `animus-types>=0.1.0,<1` to runtime dependencies; removed the incorrect `artifacts = ["*.schema.json"]` hatchling setting.
- Moved all `*.schema.json` files from `packages/contracts/` into `packages/contracts/src/animus_contracts/schemas/` so they are packaged with the wheel.
- Updated `src/animus_contracts/__init__.py` and `src/animus_contracts/validator.py` to load schemas from the packaged `schemas/` directory.
- Updated `tests/test_all_schemas.py` and `tests/test_new_schemas.py` to use `from animus_contracts import SCHEMAS_DIR`.
- Removed the duplicate root-level `packages/contracts/test_new_schemas.py`.
- Updated `packages/types/tests/test_schemas.py`, `scripts/validate_schemas.py`, `scripts/validate_contracts.py`, and `scripts/compile_schemas.py` to point at the new schema location.
- Updated `ci.yml` `schema-validate` and `contracts` jobs to install `animus-types` and `animus-contracts` together.

### `animus-core`

- `packages/core/pyproject.toml`: corrected `extras[all]` from `animus[...]` to `animus-core[...]`.

### Cross-package SBOM workflow

- `ci.yml` `sbom` job now installs `animus-types` for packages that depend on it (`core`, `kernel`, `forge`, `contracts`).

## Verification

### Local dev environment

- `pytest packages/contracts/tests/` — 82 passed.
- `pytest packages/kernel/tests/test_memory_types.py packages/kernel/tests/test_redaction.py` — 38 passed.
- `pytest packages/types/tests/test_schemas.py` — 27 passed.
- `pytest tests/integration/` — 14 passed.
- `python scripts/validate_schemas.py` — PASS (25 schemas valid).
- `python scripts/validate_contracts.py` — PASS.

### Clean-room wheelhouse

Built fresh wheels for `types`, `contracts`, `kernel`, and `core` into a flat wheelhouse. A clean venv installed all four with full dependency resolution from PyPI. Verified:

- `import animus_types, animus_contracts, animus_kernel, animus` succeeds.
- `from animus_contracts import SCHEMAS_DIR` reports 25 packaged schemas.
- `animus_kernel.memory.types`, `animus_kernel.providers.base`, and `animus_kernel.network.egress` import successfully.
- Installing the local `animus_core` wheel and then resolving `animus-core[all]` no longer pulls the unrelated PyPI package `animus-0.0.2`.

## Remaining notes

- The previously published PyPI version of `animus-core` (2.3.0) still carries the old `extras[all]` metadata. A new release is required for public installs to benefit from the fix.
- `animus-types` is not yet published to PyPI. After this change, `animus-kernel` and `animus-contracts` wheels honestly declare a dependency that is only resolvable from the local monorepo or from PyPI once `animus-types` is published.

## Discrepancy status

| ID  | Package        | Severity | REL-02 status |
|-----|----------------|----------|---------------|
| D1  | animus-kernel  | critical | closed        |
| D2  | animus-contracts | critical | closed      |
| D5  | animus-core    | critical | closed        |
