"""Animus Contracts — canonical JSON schemas and runtime validation."""

from __future__ import annotations

from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("animus-contracts")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+dev"

from animus_contracts.validator import ValidationError, validate, validate_with_schema

_pkg_dir = Path(__file__).resolve().parent
SCHEMAS_DIR = _pkg_dir / "schemas"

__all__ = ["SCHEMAS_DIR", "validate", "validate_with_schema", "ValidationError", "__version__"]
