"""Rust-backed intent graph backend.

Wraps the PyO3 ``IntentGraph`` from ``convergent._core`` and converts
between Python ``Intent`` dataclasses and the dict format used by
the Rust layer.

Key format differences:
- InterfaceKind: Rust returns Debug format (``"Function"``) vs Python ``"function"``
- Constraints: Rust dicts omit ``severity`` — defaults to ``"required"``
- Timestamps: Rust returns RFC3339 strings — parsed to datetime
"""

from __future__ import annotations

from datetime import datetime, timezone

from convergent._serialization import (
    dict_to_constraint,
    dict_to_evidence,
    spec_to_dict,
)
from convergent.intent import (
    Evidence,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)

try:
    from convergent._core import IntentGraph as _RustIntentGraph

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

# Rust Debug format → Python enum value
_RUST_KIND_MAP: dict[str, str] = {
    "Function": "function",
    "Class": "class",
    "Model": "model",
    "Endpoint": "endpoint",
    "Migration": "migration",
    "Config": "config",
}


def _rust_dict_to_spec(d: dict) -> InterfaceSpec:
    """Convert a dict returned by Rust to InterfaceSpec.

    Handles the capitalized Debug format for InterfaceKind.
    """
    raw_kind = d["kind"]
    kind_value = _RUST_KIND_MAP.get(raw_kind, raw_kind.lower())
    return InterfaceSpec(
        name=d["name"],
        kind=InterfaceKind(kind_value),
        signature=d["signature"],
        module_path=d.get("module_path", ""),
        tags=d.get("tags", []),
    )


def _parse_timestamp(ts: str | None) -> datetime:
    """Parse RFC3339 timestamp from Rust, falling back to UTC now."""
    if not ts:
        return datetime.now(timezone.utc)
    # Python 3.11+ handles RFC3339 with fromisoformat; 3.10 needs +00:00 not Z
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _rust_dict_to_intent(d: dict) -> Intent:
    """Convert a dict returned by Rust PyIntentGraph to an Intent."""
    provides = [_rust_dict_to_spec(s) for s in d.get("provides", [])]
    requires = [_rust_dict_to_spec(s) for s in d.get("requires", [])]
    constraints = [dict_to_constraint(c) for c in d.get("constraints", [])]

    # Rust doesn't serialize evidence in query results — build from kind/description
    evidence: list[Evidence] = []
    for e in d.get("evidence", []):
        evidence.append(dict_to_evidence(e))

    return Intent(
        id=d["id"],
        agent_id=d["agent_id"],
        timestamp=_parse_timestamp(d.get("timestamp")),
        intent=d["intent"],
        provides=provides,
        requires=requires,
        constraints=constraints,
        evidence=evidence,
        stability=d.get("stability", 0.3),
        parent_id=d.get("parent_id"),
    )


def _intent_to_rust_dict(intent: Intent) -> dict:
    """Convert an Intent to the dict format expected by Rust PyIntentGraph."""
    d = {
        "id": intent.id,
        "agent_id": intent.agent_id,
        "intent": intent.intent,
        "provides": [spec_to_dict(s) for s in intent.provides],
        "requires": [spec_to_dict(s) for s in intent.requires],
        "constraints": [
            {"target": c.target, "requirement": c.requirement, "affects_tags": c.affects_tags}
            for c in intent.constraints
        ],
        "stability": intent.stability,
        "evidence": [{"kind": e.kind.value, "description": e.description} for e in intent.evidence],
    }
    if intent.parent_id is not None:
        d["parent_id"] = intent.parent_id
    return d


class RustGraphBackend:
    """Intent graph backend powered by the Rust ``IntentGraph`` via PyO3.

    Drop-in replacement for ``SQLiteBackend`` or ``PythonGraphBackend``.
    Requires ``convergent._core`` (built with ``maturin develop --release``).

    Args:
        db_path: Path to SQLite database file, or ``None`` for in-memory.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if not HAS_RUST:
            raise RuntimeError("Rust backend not available. Build with: maturin develop --release")
        self._graph = _RustIntentGraph(db_path)

    def publish(self, intent: Intent) -> float:
        """Publish an intent and return its computed stability."""
        return self._graph.publish(_intent_to_rust_dict(intent))

    def query_all(self, min_stability: float | None = None) -> list[Intent]:
        """Query all intents, optionally filtered by minimum stability."""
        raw = self._graph.query_all(min_stability)
        return [_rust_dict_to_intent(d) for d in raw]

    def query_by_agent(self, agent_id: str) -> list[Intent]:
        """Query intents published by a specific agent."""
        raw = self._graph.query_by_agent(agent_id)
        return [_rust_dict_to_intent(d) for d in raw]

    def find_overlapping(
        self,
        specs: list[InterfaceSpec],
        exclude_agent: str,
        min_stability: float,
    ) -> list[Intent]:
        """Find intents with overlapping interfaces."""
        specs_dicts = [spec_to_dict(s) for s in specs]
        raw = self._graph.find_overlapping(specs_dicts, exclude_agent, min_stability)
        return [_rust_dict_to_intent(d) for d in raw]

    def count(self) -> int:
        """Return the total number of intents in the graph."""
        return self._graph.count()

    def summary(self) -> dict:
        """Return graph summary statistics from the Rust core."""
        return dict(self._graph.summary())

    def close(self) -> None:
        """No-op — Rust manages its own connection lifetime."""
