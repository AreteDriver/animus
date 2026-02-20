"""Quality gate evaluation with safe condition parsing.

No eval() — uses a manual parser for security. Supports:
  - output.field >= N  (numeric: >=, <=, >, <, ==, !=)
  - output contains "text"
  - output.length >= N
  - true / false
"""

import json
import re

from animus.forge.models import ForgeError, GateConfig
from animus.logging import get_logger

logger = get_logger("forge.gates")

_NUMERIC_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def evaluate_gate(
    gate: GateConfig,
    outputs: dict[str, str],
) -> tuple[bool, str]:
    """Evaluate a quality gate against collected outputs.

    Args:
        gate: Gate configuration.
        outputs: Flat dict of all outputs from prior steps.
            Keys are "agent_name.output_name" or just "output_name".

    Returns:
        (passed, reason) — reason is empty string on success.

    Raises:
        ForgeError: If the condition syntax is invalid.
    """
    if gate.type == "human":
        return True, ""

    condition = gate.pass_condition.strip()

    if not condition or condition.lower() == "true":
        return True, ""

    if condition.lower() == "false":
        return False, f"Gate {gate.name!r} always fails (condition='false')"

    # Pattern: output.field OP value
    # Pattern: output OP value
    # Pattern: output contains "text"
    # Pattern: output.length OP N

    return _evaluate_condition(condition, outputs, gate.name)


def _evaluate_condition(
    condition: str,
    outputs: dict[str, str],
    gate_name: str,
) -> tuple[bool, str]:
    """Parse and evaluate a single condition."""

    # Handle 'contains' operator
    contains_match = re.match(r'^(\S+)\s+contains\s+"([^"]*)"$', condition)
    if contains_match:
        ref, substring = contains_match.groups()
        value = _resolve_ref(ref, outputs)
        if value is None:
            return False, f"Gate {gate_name!r}: output {ref!r} not found"
        passed = substring in str(value)
        if not passed:
            return False, (f"Gate {gate_name!r}: {ref!r} does not contain {substring!r}")
        return True, ""

    # Handle numeric comparisons: ref OP number
    numeric_match = re.match(r"^(\S+)\s*(>=|<=|!=|==|>|<)\s*(-?[\d.]+)$", condition)
    if numeric_match:
        ref, op, num_str = numeric_match.groups()
        value = _resolve_ref(ref, outputs)
        if value is None:
            return False, f"Gate {gate_name!r}: output {ref!r} not found"

        try:
            actual = float(value)
        except (ValueError, TypeError):
            return False, (f"Gate {gate_name!r}: {ref!r} is not numeric ({value!r})")

        expected = float(num_str)
        op_fn = _NUMERIC_OPS[op]
        passed = op_fn(actual, expected)
        if not passed:
            return False, (f"Gate {gate_name!r}: {actual} {op} {expected} is false")
        return True, ""

    raise ForgeError(f"Gate {gate_name!r}: unsupported condition syntax: {condition!r}")


def _resolve_ref(ref: str, outputs: dict[str, str]) -> str | None:
    """Resolve an output reference like 'review.score' or 'review.length'.

    Handles:
      - "name" → direct lookup in outputs
      - "name.field" → JSON parse outputs[name] and extract field
      - "name.length" → len(outputs[name])
    """
    parts = ref.split(".", 1)

    if len(parts) == 1:
        return outputs.get(ref)

    base, field_name = parts

    # Special: .length
    if field_name == "length":
        raw = outputs.get(base)
        if raw is None:
            return None
        return str(len(raw))

    # Direct key match first (e.g., "reviewer.score" as a flat key)
    if ref in outputs:
        return outputs[ref]

    # JSON field access
    raw = outputs.get(base)
    if raw is None:
        return None

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and field_name in parsed:
            return str(parsed[field_name])
    except (json.JSONDecodeError, TypeError):
        pass

    return None
