"""Structural semantic matching utilities — deterministic, no LLM.

Provides normalization and comparison functions for interface names,
type signatures, and constraint targets. Used by InterfaceSpec and
Constraint to upgrade from exact-string matching to structural matching.
"""

from __future__ import annotations

import re

# Known suffixes to strip for name normalization
_NAME_SUFFIXES = ("Model", "Service", "Handler", "Controller", "Spec", "Interface")

# Type alias map for normalization
_TYPE_ALIASES: dict[str, str] = {
    "UUID": "uuid",
    "uuid": "uuid",
    "str": "str",
    "String": "str",
    "string": "str",
    "int": "int",
    "i32": "int",
    "i64": "int",
    "i128": "int",
    "u32": "int",
    "u64": "int",
    "float": "float",
    "f32": "float",
    "f64": "float",
    "bool": "bool",
    "boolean": "bool",
}

# Regex to split CamelCase into tokens
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)")


def normalize_name(name: str) -> str:
    """Normalize an interface name for comparison.

    Lowercase, strip known suffixes (Model, Service, etc.),
    split CamelCase into tokens.

    Examples:
        "UserModel" -> "user"
        "AuthService" -> "auth"
        "MealPlanService" -> "meal plan"
        "User" -> "user"
    """
    if not name:
        return ""

    # Strip known suffixes
    stripped = name
    for suffix in _NAME_SUFFIXES:
        if stripped.endswith(suffix) and len(stripped) > len(suffix):
            stripped = stripped[: -len(suffix)]
            break

    # Split CamelCase into tokens
    tokens = _CAMEL_RE.findall(stripped)
    if not tokens:
        return stripped.lower()

    return " ".join(t.lower() for t in tokens)


def names_overlap(a: str, b: str) -> bool:
    """Check if two names refer to the same concept.

    Returns True if normalized names are equal, one is a prefix
    of the other, or one contains the other.
    """
    if not a or not b:
        return False

    na = normalize_name(a)
    nb = normalize_name(b)

    if na == nb:
        return True

    # Prefix match
    if na.startswith(nb) or nb.startswith(na):
        return True

    # Containment match
    return bool(na in nb or nb in na)


def normalize_type(t: str) -> str:
    """Normalize a type string for comparison.

    Handles aliases (UUID<->uuid, String<->str, i64<->int),
    Optional[X] -> X, list[X]<->Vec<X><->List[X].
    """
    t = t.strip()
    if not t:
        return ""

    # Handle Optional[X] -> X
    if t.startswith("Optional[") and t.endswith("]"):
        t = t[9:-1]

    # Handle X | None or None | X
    if " | " in t:
        parts = [p.strip() for p in t.split(" | ") if p.strip() != "None"]
        t = parts[0] if parts else ""

    # Handle generic containers: list[X], List[X], Vec<X>
    container_match = re.match(r"(?:list|List|Vec)\s*[\[<]\s*(.+?)\s*[\]>]", t)
    if container_match:
        inner = normalize_type(container_match.group(1))
        return f"list[{inner}]"

    # Direct alias lookup
    return _TYPE_ALIASES.get(t, t.lower())


def parse_signature(sig: str) -> dict[str, str]:
    """Parse "field: type, field: type" into {field: type} dict.

    Returns empty dict for empty/unparseable signatures.
    """
    if not sig or not sig.strip():
        return {}

    result: dict[str, str] = {}
    for part in sig.split(","):
        part = part.strip()
        if ":" in part:
            field, type_str = part.split(":", 1)
            result[field.strip()] = type_str.strip()
    return result


def signatures_compatible(a: str, b: str) -> bool:
    """Check if signature b is compatible with signature a.

    Compatible if b's fields are a superset of a's fields
    with normalized types. Empty a is compatible with anything.
    """
    fields_a = parse_signature(a)
    fields_b = parse_signature(b)

    if not fields_a:
        return True

    for field, type_a in fields_a.items():
        if field not in fields_b:
            return False
        if normalize_type(type_a) != normalize_type(fields_b[field]):
            return False

    return True


def normalize_constraint_target(target: str) -> str:
    """Normalize a constraint target for comparison.

    Lowercase, strip "model"/"service" suffix, replace
    underscores/hyphens with spaces, collapse whitespace.

    Examples:
        "User Model" -> "user"
        "user_model" -> "user"
        "User model" -> "user"
        "user-service" -> "user"
    """
    if not target:
        return ""

    # Lowercase
    t = target.lower()

    # Replace underscores and hyphens with spaces
    t = t.replace("_", " ").replace("-", " ")

    # Collapse whitespace
    t = " ".join(t.split())

    # Strip known suffixes
    for suffix in ("model", "service"):
        if t.endswith(f" {suffix}"):
            t = t[: -(len(suffix) + 1)]
        elif t == suffix:
            break  # don't strip if it's the entire string

    return t.strip()
