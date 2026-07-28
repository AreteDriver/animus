"""Runtime JSON Schema validator for Animus contracts.

Loads all ``*.schema.json`` files discovered next to this module and builds a
resolver so cross-schema ``$ref`` declarations resolve correctly.

Usage::

    from animus_contracts import validate, ValidationError

    try:
        validate(data, "action")
    except ValidationError as exc:
        print(exc.errors)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from animus_types import ValidationError
from jsonschema import Draft202012Validator

logger = logging.getLogger(__name__)

_pkg_dir = Path(__file__).resolve().parent
SCHEMAS_DIR = _pkg_dir.parent.parent  # canonical: next to pyproject.toml


def _load_schemas() -> dict[str, Any]:
    """Load every ``*.schema.json`` in the package directory into a URI store."""
    store: dict[str, Any] = {}
    for path in SCHEMAS_DIR.glob("*.schema.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON schema: %s", path)
            continue

        schema_id = raw.get("$id")
        if not schema_id:
            logger.warning("Schema %s missing $id — skipping", path)
            continue

        store[schema_id] = raw

    return store


# Module-level singleton — schemas are loaded once on first import.
_SCHEMA_STORE = _load_schemas()


def _get_registry():
    """Build a referencing.Registry for cross-schema $ref resolution."""
    try:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        registry = Registry()
        for uri, contents in _SCHEMA_STORE.items():
            resource = Resource(contents=contents, specification=DRAFT202012)
            registry = registry.with_resource(uri, resource)
        return registry
    except ImportError:
        return None


_REGISTRY = _get_registry()


def _make_validator(schema_uri: str, schema: dict[str, Any]) -> Draft202012Validator:
    """Build a Draft202012Validator with cross-schema $ref resolution."""
    if _REGISTRY is not None:
        validator_cls = Draft202012Validator
        return validator_cls(schema, registry=_REGISTRY)

    # Fallback for older jsonschema without referencing support
    from jsonschema import RefResolver  # type: ignore[attr-defined]

    resolver = RefResolver(base_uri=schema_uri, referrer=schema, store=_SCHEMA_STORE)
    return Draft202012Validator(schema, resolver=resolver)


def validate(data: dict[str, Any], schema_name: str) -> None:
    """Validate *data* against the named schema.

    Args:
        data: The JSON-like payload to validate.
        schema_name: Basename of the schema file without the ``.schema.json``
            suffix (e.g. ``"action"``).

    Raises:
        ValidationError: When the payload does not conform to the schema.
    """
    schema_uri = f"https://animus.local/schemas/{schema_name}.schema.json"
    schema = _SCHEMA_STORE.get(schema_uri)
    if not schema:
        raise ValidationError(schema_name, [f"Schema '{schema_name}' not found"])

    validator = _make_validator(schema_uri, schema)
    errors = list(validator.iter_errors(data))
    if errors:
        messages = [f"{e.json_path}: {e.message}" for e in errors]
        raise ValidationError(schema_name, messages)


def validate_with_schema(data: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate *data* against an inline schema dict (no registry lookup).

    Args:
        data: The JSON-like payload to validate.
        schema: A JSON Schema object.

    Raises:
        ValidationError: When the payload does not conform to the schema.
    """
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    if errors:
        messages = [f"{e.json_path}: {e.message}" for e in errors]
        schema_title = schema.get("title", "inline")
        raise ValidationError(schema_title, messages)
