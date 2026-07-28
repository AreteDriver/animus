"""Shared exception types for Animus.

These live in ``animus_types`` (the bottom of the dependency stack) so that
any layer can catch or raise them without creating circular / reverse
dependencies.
"""

from __future__ import annotations


class ValidationError(Exception):
    """Raised when a payload fails schema validation.

    Attributes:
        schema_name: The schema that was requested (e.g. ``"action"``).
        errors: Human-readable list of validation failures.
    """

    def __init__(self, schema_name: str, errors: list[str]) -> None:
        self.schema_name = schema_name
        self.errors = errors
        super().__init__(f"Validation failed for schema '{schema_name}': {errors}")
