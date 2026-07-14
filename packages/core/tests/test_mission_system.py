"""Tests for MissionSystem orchestrator.

Covers: issue, lifecycle (start/pause/resume), result reporting, timeout,
cancel, debrief, queries, persistence, stats, and adversarial cases.
"""

from __future__ import annotations

import json
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from animus.mission.order import MissionConstraint, MissionOrder, MissionResult, MissionStatus
from animus.mission.runtime import LocalRuntime
from animus.mission.system import MissionConfig, MissionSystem


class TestIssue:
    def test_issue_success(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            issued = system.issue(order)
            assert issued.status == MissionStatus.ISSUED
            assert issued.issued_at is not None
            assert issued.id in system._missions

    def test_issue_with_custom_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            rt = LocalRuntime()
            issued = system.issue(order, runtime=rt)
            assert issued.status == MissionStatus.ISSUED

    def test_issue_max_concurrent(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(max_concurrent_missions=2, persistence_dir=Path(td)))
            system.issue(MissionOrder(citizen_id="c1", mission_type="scan"))
            system.issue(MissionOrder(citizen_id="c2", mission_type="scan"))
            with pytest.raises(RuntimeError, match="Max concurrent"):
                system.issue(MissionOrder(citizen_id="c3", mission_type="scan"))

    def test_issue_runtime_capability_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(enforce_runtime_caps=True, persistence_dir=Path(td)))
            order = MissionOrder(
                citizen_id="c1",
                mission_type="scan",
                constraints=[MissionConstraint("max_concurrent", 9999, "general")],
            )
            rt = LocalRuntime()
            # LocalRuntime max_concurrent is 5, order requires 9999
            with pytest.raises(RuntimeError, match="max concurrent"):
                system.issue(order, runtime=rt)


class TestLifecycle:
    def test_start(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            started = system.start(order.id)
            assert started.status == MissionStatus.RUNNING
            assert started.started_at is not None

    def test_pause_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            system.start(order.id)
            system.pause(order.id)
            assert system.status(order.id) == MissionStatus.PAUSED
            system.resume(order.id)
            assert system.status(order.id) == MissionStatus.RUNNING

    def test_report_result_success(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            result = MissionResult(success=True, outputs={"found": 3})
            updated = system.report_result(order.id, result)
            assert updated.status == MissionStatus.COMPLETED
            assert updated.result.success is True

    def test_report_result_failure(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            result = MissionResult(success=False, errors=["crash"])
            updated = system.report_result(order.id, result)
            assert updated.status == MissionStatus.FAILED
            assert updated.result.success is False

    def test_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            timed = system.timeout(order.id)
            assert timed.status == MissionStatus.TIMED_OUT
            assert timed.result.success is False
            assert "timed out" in timed.result.errors[0]

    def test_cancel(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            cancelled = system.cancel(order.id, reason="user_request")
            assert cancelled.status == MissionStatus.FAILED
            assert "user_request" in cancelled.result.errors[0]


class TestDebrief:
    def test_debrief_completed(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            result = MissionResult(success=True, outputs={"found": 3})
            system.report_result(order.id, result)
            debriefed = system.debrief(order.id)
            assert debriefed is not None
            assert debriefed.success is True
            assert system.get(order.id).status == MissionStatus.DEBRIEFED

    def test_debrief_failed(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            result = MissionResult(success=False, errors=["crash"])
            system.report_result(order.id, result)
            debriefed = system.debrief(order.id)
            assert debriefed is not None
            assert debriefed.success is False

    def test_debrief_not_ready(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            # Still ISSUED — cannot debrief
            assert system.debrief(order.id) is None

    def test_auto_debrief(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=True, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            result = MissionResult(success=True)
            system.report_result(order.id, result)
            # auto_debrief should have run
            assert system.get(order.id).status == MissionStatus.DEBRIEFED


class TestQueries:
    def test_list_active(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            o1 = MissionOrder(citizen_id="c1", mission_type="scan")
            o2 = MissionOrder(citizen_id="c2", mission_type="scan")
            system.issue(o1)
            system.issue(o2)
            system.start(o1.id)
            active = system.list_active()
            assert len(active) == 2  # ISSUED counts as active

    def test_list_completed(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            system.report_result(order.id, MissionResult(success=True))
            completed = system.list_completed()
            assert len(completed) == 1

    def test_list_failed(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            system.report_result(order.id, MissionResult(success=False, errors=["e"]))
            failed = system.list_failed()
            assert len(failed) == 1

    def test_history(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            system.report_result(order.id, MissionResult(success=True))
            system.debrief(order.id)
            h = system.history(citizen_id="c1")
            assert len(h) == 1
            assert h[0]["citizen_id"] == "c1"

    def test_stats(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=False, persistence_dir=Path(td)))
            o1 = MissionOrder(citizen_id="c1", mission_type="scan")
            o2 = MissionOrder(citizen_id="c2", mission_type="scan")
            system.issue(o1)
            system.issue(o2)
            system.report_result(o1.id, MissionResult(success=True))
            system.report_result(o2.id, MissionResult(success=False, errors=["e"]))
            s = system.stats()
            assert s["total"] == 2
            assert s["completed"] == 1
            assert s["failed"] == 1
            assert s["success_rate"] == 0.5


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            config = MissionConfig(persistence_dir=Path(td), auto_debrief=False)
            system = MissionSystem(config)
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            system.report_result(order.id, MissionResult(success=True))
            system.debrief(order.id)

            # Create new system pointing at same dir
            system2 = MissionSystem(config)
            loaded = system2.get(order.id)
            assert loaded is not None
            assert loaded.status == MissionStatus.DEBRIEFED
            assert loaded.result.success is True

    def test_persist_creates_dir(self):
        with tempfile.TemporaryDirectory() as td:
            config = MissionConfig(persistence_dir=Path(td) / "nested" / "missions")
            system = MissionSystem(config)
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            assert (Path(td) / "nested" / "missions" / "mission_state.json").exists()


class TestAdversarial:
    def test_issue_with_none_runtime_uses_local(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            issued = system.issue(order, runtime=None)
            assert issued.status == MissionStatus.ISSUED

    def test_get_missing_order(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            assert system.get("nonexistent") is None

    def test_status_missing_order(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            assert system.status("nonexistent") is None

    def test_debrief_missing_order(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            assert system.debrief("nonexistent") is None

    def test_timeout_missing_order(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            assert system.timeout("nonexistent") is None

    def test_cancel_missing_order(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            assert system.cancel("nonexistent") is None

    def test_start_already_running(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            system.start(order.id)
            system.start(order.id)  # idempotent
            assert system.status(order.id) == MissionStatus.RUNNING

    def test_report_result_on_debriefed(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(auto_debrief=True, persistence_dir=Path(td)))
            order = MissionOrder(citizen_id="c1", mission_type="scan")
            system.issue(order)
            system.report_result(order.id, MissionResult(success=True))
            # Already debriefed by auto_debrief
            assert system.get(order.id).status == MissionStatus.DEBRIEFED
            # Reporting again should work but stay debriefed
            system.report_result(order.id, MissionResult(success=False, errors=["late"]))
            assert system.get(order.id).status == MissionStatus.DEBRIEFED
            assert system.get(order.id).result.success is False

    def test_stats_empty(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            s = system.stats()
            assert s["total"] == 0
            assert s["success_rate"] == 0.0

    def test_repr(self):
        with tempfile.TemporaryDirectory() as td:
            system = MissionSystem(config=MissionConfig(persistence_dir=Path(td)))
            r = repr(system)
            assert "MissionSystem" in r
            assert "total=0" in r


class TestMissionConstraintKinds:
    def test_constraint_kinds_preserved(self):
        c = MissionConstraint(name="no_network", value=True, kind="safety")
        assert c.to_dict()["kind"] == "safety"
