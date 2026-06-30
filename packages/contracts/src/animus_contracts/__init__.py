"""Animus Contracts — canonical JSON schemas and runtime validation."""

from __future__ import annotations

from pathlib import Path

from animus_contracts.validator import ValidationError, validate, validate_with_schema

_pkg_dir = Path(__file__).resolve().parent
SCHEMAS_DIR = _pkg_dir.parent.parent  # canonical: next to pyproject.toml

__all__ = ["SCHEMAS_DIR", "validate", "validate_with_schema", "ValidationError"]
