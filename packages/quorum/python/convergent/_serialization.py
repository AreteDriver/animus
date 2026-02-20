"""Serialization helpers for converting between Intent dataclasses and dicts.

Used by SQLiteBackend, RustGraphBackend, and any future backends that
need to serialize/deserialize intents through dict intermediaries.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from convergent.intent import (
    Constraint,
    ConstraintSeverity,
    Evidence,
    EvidenceKind,
    Intent,
    InterfaceKind,
    InterfaceSpec,
)


def spec_to_dict(spec: InterfaceSpec) -> dict:
    return {
        "name": spec.name,
        "kind": spec.kind.value,
        "signature": spec.signature,
        "module_path": spec.module_path,
        "tags": spec.tags,
    }


def dict_to_spec(d: dict) -> InterfaceSpec:
    return InterfaceSpec(
        name=d["name"],
        kind=InterfaceKind(d["kind"]),
        signature=d["signature"],
        module_path=d.get("module_path", ""),
        tags=d.get("tags", []),
    )


def constraint_to_dict(c: Constraint) -> dict:
    return {
        "target": c.target,
        "requirement": c.requirement,
        "severity": c.severity.value,
        "affects_tags": c.affects_tags,
    }


def dict_to_constraint(d: dict) -> Constraint:
    return Constraint(
        target=d["target"],
        requirement=d["requirement"],
        severity=ConstraintSeverity(d.get("severity", "required")),
        affects_tags=d.get("affects_tags", []),
    )


def evidence_to_dict(e: Evidence) -> dict:
    return {
        "kind": e.kind.value,
        "description": e.description,
        "timestamp": e.timestamp.isoformat(),
    }


def dict_to_evidence(d: dict) -> Evidence:
    ts = d.get("timestamp")
    timestamp = datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc)
    return Evidence(
        kind=EvidenceKind(d["kind"]),
        description=d["description"],
        timestamp=timestamp,
    )


def row_to_intent(row: sqlite3.Row) -> Intent:
    """Reconstruct an Intent from a database row."""
    provides = [dict_to_spec(d) for d in json.loads(row["provides"])]
    requires = [dict_to_spec(d) for d in json.loads(row["requires"])]
    constraints = [dict_to_constraint(d) for d in json.loads(row["constraints"])]
    evidence = [dict_to_evidence(d) for d in json.loads(row["evidence"])]

    return Intent(
        id=row["id"],
        agent_id=row["agent_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        intent=row["intent"],
        provides=provides,
        requires=requires,
        constraints=constraints,
        evidence=evidence,
        stability=row["stability"],
        parent_id=row["parent_id"],
    )
