"""Tests for MissionOrder and MissionResult data models."""

from __future__ import annotations

import pytest
from datetime import timedelta

from animus.mission.order import (
    MissionConstraint,
    MissionOrder,
    MissionResult,
    MissionStatus,
)


class TestMissionConstraint:
    def test_creation(self):
        c = MissionConstraint(name="max_tokens", value=1000, kind="budget")
        assert c.name == "max_tokens"
        assert c.value == 1000
        assert c.kind == "budget"

    def test_roundtrip_dict(self):
        c = MissionConstraint(name="readonly", value=True, kind="authority", description="No writes")
        data = c.to_dict()
        c2 = MissionConstraint.from_dict(data)
        assert c2.name == c.name
        assert c2.value == c.value
        assert c2.kind == c.kind
        assert c2.description == c.description


class TestMissionResult:
    def test_success(self):
        r = MissionResult(success=True, outputs={"files": ["a.py"]}, metrics={"coverage": 0.85})
        assert r.success is True
        assert r.outputs["files"] == ["a.py"]

    def test_failure(self):
        r = MissionResult(success=False, errors=["timeout"])
        assert r.success is False
        assert r.errors == ["timeout"]

    def test_roundtrip_dict(self):
        r = MissionResult(
            success=True,
            outputs={"key": "val"},
            artifacts=["/tmp/out.json"],
            metrics={"score": 0.9},
        )
        data = r.to_dict()
        r2 = MissionResult.from_dict(data)
        assert r2.success == r.success
        assert r2.outputs == r.outputs
        assert r2.artifacts == r.artifacts
        assert r2.metrics == r.metrics


class TestMissionOrder:
    def test_default_authority_limits(self):
        o = MissionOrder(citizen_id="c1", mission_type="scan")
        assert "read" in o.authority_limits
        assert "write" not in o.authority_limits

    def test_default_return_conditions(self):
        o = MissionOrder(citizen_id="c1", mission_type="scan")
        assert "on_complete" in o.return_conditions
        assert "on_timeout" in o.return_conditions

    def test_add_constraint_fluent(self):
        o = MissionOrder(citizen_id="c1", mission_type="scan")
        o.add_constraint("max_tokens", 5000, "budget").add_constraint("readonly", True, "authority")
        assert len(o.constraints) == 2
        assert o.constraints[0].name == "max_tokens"

    def test_is_active_states(self):
        o = MissionOrder(citizen_id="c1", mission_type="scan")
        o.status = MissionStatus.ISSUED
        assert o.is_active is True
        o.status = MissionStatus.RUNNING
        assert o.is_active is True
        o.status = MissionStatus.COMPLETED
        assert o.is_active is False

    def test_has_debriefed(self):
        o = MissionOrder(citizen_id="c1", mission_type="scan")
        assert o.has_debriefed is False
        o.status = MissionStatus.DEBRIEFED
        assert o.has_debriefed is True

    def test_roundtrip_dict(self):
        o = MissionOrder(
            citizen_id="c1",
            mission_type="scan",
            objectives={"target": "repo"},
            constraints=[MissionConstraint("max_tokens", 1000, "budget")],
            authority_limits=["read"],
            return_conditions=["on_complete"],
            timeout=timedelta(minutes=15),
            priority=3,
            tags=["security"],
        )
        o.status = MissionStatus.RUNNING
        o.issued_at = __import__("datetime").datetime.now()
        o.result = MissionResult(success=True, outputs={"found": 2})

        data = o.to_dict()
        o2 = MissionOrder.from_dict(data)
        assert o2.citizen_id == o.citizen_id
        assert o2.mission_type == o.mission_type
        assert o2.objectives == o.objectives
        assert len(o2.constraints) == 1
        assert o2.status == o.status
        assert o2.result is not None
        assert o2.result.success == o.result.success
