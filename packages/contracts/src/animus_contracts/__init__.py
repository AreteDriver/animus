"""Animus Contracts — canonical JSON schemas and runtime validation."""

from __future__ import annotations

from pathlib import Path

from animus_contracts.validator import ValidationError, validate, validate_with_schema

_pkg_dir = Path(__file__).resolve().parent
# In the built wheel, schemas live next to the Python files (force-include).
# In the monorepo checkout, they live at the package root (two levels up).
SCHEMAS_DIR = (
    _pkg_dir
    if (_pkg_dir / "action.schema.json").exists()
    else _pkg_dir.parent.parent
)

__all__ = ["SCHEMAS_DIR", "validate", "validate_with_schema", "ValidationError"]
