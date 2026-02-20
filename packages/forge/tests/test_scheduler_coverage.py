"""Tests for scheduler/schedule_manager.py coverage."""

import sys
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from animus_forge.scheduler.schedule_manager import (
    CronConfig,
    IntervalConfig,
    ScheduleManager,
    ScheduleStatus,
    ScheduleType,
    WorkflowSchedule,
    _parse_datetime,
)


def _mock_backend():
    """Create a mock database backend with all required methods."""
    backend = MagicMock()
    backend.fetchone.return_value = None
    backend.fetchall.return_value = []
    backend.execute.return_value = None
    backend.executescript.return_value = None

    @contextmanager
    def _txn():
        yield

    backend.transaction = _txn
    return backend


def _make_manager(backend=None):
    """Construct a ScheduleManager with mocked dependencies."""
    backend = backend or _mock_backend()
    mock_scheduler = MagicMock()
    mock_scheduler.running = False
    mock_scheduler.get_job.return_value = None
    with (
        patch(
            "animus_forge.scheduler.schedule_manager.get_settings",
            return_value=MagicMock(),
        ),
        patch(
            "animus_forge.scheduler.schedule_manager.get_database",
            return_value=backend,
        ),
        patch(
            "animus_forge.scheduler.schedule_manager.WorkflowEngineAdapter",
            return_value=MagicMock(),
        ),
        patch(
            "animus_forge.scheduler.schedule_manager.BackgroundScheduler",
            return_value=mock_scheduler,
        ),
    ):
        mgr = ScheduleManager(backend=backend)
    return mgr


def _cron_schedule(sid="sched-1", workflow_id="wf-1", status=ScheduleStatus.ACTIVE):
    """Create a WorkflowSchedule with cron config for testing."""
    return WorkflowSchedule(
        id=sid,
        workflow_id=workflow_id,
        name="Test Cron Schedule",
        schedule_type=ScheduleType.CRON,
        cron_config=CronConfig(minute="0", hour="*/2"),
        status=status,
    )


def _interval_schedule(sid="sched-2", workflow_id="wf-2"):
    """Create a WorkflowSchedule with interval config for testing."""
    return WorkflowSchedule(
        id=sid,
        workflow_id=workflow_id,
        name="Test Interval Schedule",
        schedule_type=ScheduleType.INTERVAL,
        interval_config=IntervalConfig(minutes=30),
        status=ScheduleStatus.ACTIVE,
    )


# ---------- Tests ----------


class TestParseDateTime:
    def test_none_returns_none(self):
        assert _parse_datetime(None) is None

    def test_datetime_passthrough(self):
        now = datetime.now()
        assert _parse_datetime(now) is now

    def test_iso_string(self):
        dt = _parse_datetime("2025-06-15T10:30:00")
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.hour == 10


class TestCreateSchedule:
    def test_create_cron_schedule_success(self):
        mgr = _make_manager()
        mgr.workflow_engine.load_workflow.return_value = MagicMock()
        # _save_schedule calls fetchone then _insert; fetchone returns None (no existing)
        mgr.backend.fetchone.return_value = None
        schedule = _cron_schedule()

        result = mgr.create_schedule(schedule)

        assert result is True
        assert "sched-1" in mgr._schedules

    def test_create_schedule_workflow_not_found(self):
        mgr = _make_manager()
        mgr.workflow_engine.load_workflow.return_value = None
        schedule = _cron_schedule()

        try:
            mgr.create_schedule(schedule)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "not found" in str(exc)

    def test_create_schedule_save_failure_returns_false(self):
        mgr = _make_manager()
        mgr.workflow_engine.load_workflow.return_value = MagicMock()
        # Make the insert raise to trigger save failure
        mgr.backend.execute.side_effect = Exception("DB error")
        schedule = _cron_schedule()

        result = mgr.create_schedule(schedule)

        assert result is False


class TestUpdateSchedule:
    def test_update_preserves_created_at_and_run_count(self):
        mgr = _make_manager()
        original = _cron_schedule()
        original.created_at = datetime(2024, 1, 1)
        original.run_count = 42
        original.last_run = datetime(2025, 6, 1)
        mgr._schedules["sched-1"] = original

        # fetchone returns existing row so _save_schedule takes update path
        mgr.backend.fetchone.return_value = {"id": "sched-1"}

        updated = _cron_schedule()
        updated.created_at = datetime(2099, 12, 31)  # Should be overwritten
        updated.run_count = 0  # Should be overwritten

        result = mgr.update_schedule(updated)

        assert result is True
        saved = mgr._schedules["sched-1"]
        assert saved.created_at == datetime(2024, 1, 1)
        assert saved.run_count == 42
        assert saved.last_run == datetime(2025, 6, 1)

    def test_update_schedule_not_found(self):
        mgr = _make_manager()
        schedule = _cron_schedule(sid="nonexistent")

        try:
            mgr.update_schedule(schedule)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "not found" in str(exc)


class TestDeleteSchedule:
    def test_delete_existing_schedule(self):
        mgr = _make_manager()
        mgr._schedules["sched-1"] = _cron_schedule()
        mgr.scheduler.get_job.return_value = MagicMock()

        result = mgr.delete_schedule("sched-1")

        assert result is True
        assert "sched-1" not in mgr._schedules
        mgr.scheduler.remove_job.assert_called_once_with("schedule_sched-1")

    def test_delete_nonexistent_returns_false(self):
        mgr = _make_manager()

        result = mgr.delete_schedule("no-such-id")

        assert result is False


class TestGetAndListSchedules:
    def test_get_schedule_returns_cached(self):
        mgr = _make_manager()
        sched = _cron_schedule()
        mgr._schedules["sched-1"] = sched

        assert mgr.get_schedule("sched-1") is sched

    def test_get_schedule_returns_none_for_missing(self):
        mgr = _make_manager()

        assert mgr.get_schedule("missing") is None

    def test_list_schedules_returns_dicts(self):
        mgr = _make_manager()
        sched = _cron_schedule()
        sched.last_run = datetime(2025, 6, 1)
        mgr._schedules["sched-1"] = sched
        mgr.scheduler.get_job.return_value = None

        result = mgr.list_schedules()

        assert len(result) == 1
        entry = result[0]
        assert entry["id"] == "sched-1"
        assert entry["schedule_type"] == "cron"
        assert entry["status"] == "active"
        assert entry["run_count"] == 0
        assert entry["last_run"] is not None


class TestPauseResumeSchedule:
    def test_pause_schedule(self):
        mgr = _make_manager()
        mgr._schedules["sched-1"] = _cron_schedule()
        mgr.scheduler.get_job.return_value = MagicMock()
        mgr.backend.fetchone.return_value = {"id": "sched-1"}

        result = mgr.pause_schedule("sched-1")

        assert result is True
        assert mgr._schedules["sched-1"].status == ScheduleStatus.PAUSED
        mgr.scheduler.pause_job.assert_called_once_with("schedule_sched-1")

    def test_pause_nonexistent_returns_false(self):
        mgr = _make_manager()

        assert mgr.pause_schedule("nope") is False

    def test_resume_schedule_with_existing_job(self):
        mgr = _make_manager()
        sched = _cron_schedule()
        sched.status = ScheduleStatus.PAUSED
        mgr._schedules["sched-1"] = sched

        mock_job = MagicMock()
        mock_job.next_run_time = datetime(2025, 7, 1)
        mgr.scheduler.get_job.return_value = mock_job
        mgr.backend.fetchone.return_value = {"id": "sched-1"}

        result = mgr.resume_schedule("sched-1")

        assert result is True
        assert mgr._schedules["sched-1"].status == ScheduleStatus.ACTIVE
        mgr.scheduler.resume_job.assert_called_once_with("schedule_sched-1")

    def test_resume_nonexistent_returns_false(self):
        mgr = _make_manager()

        assert mgr.resume_schedule("nope") is False


class TestTriggerNow:
    def test_trigger_now_executes_workflow(self):
        mgr = _make_manager()
        sched = _cron_schedule()
        mgr._schedules["sched-1"] = sched

        mock_workflow = MagicMock()
        mock_workflow.variables = {}
        mock_result = MagicMock()
        mock_result.status = "completed"
        mgr.workflow_engine.load_workflow.return_value = mock_workflow
        mgr.workflow_engine.execute_workflow.return_value = mock_result
        mgr.scheduler.get_job.return_value = None
        mgr.backend.fetchone.return_value = {"id": "sched-1"}

        result = mgr.trigger_now("sched-1")

        assert result is True
        mgr.workflow_engine.execute_workflow.assert_called_once()
        # run_count should have incremented
        assert mgr._schedules["sched-1"].run_count == 1

    def test_trigger_now_nonexistent_returns_false(self):
        mgr = _make_manager()

        assert mgr.trigger_now("missing") is False


class TestExecutionHistory:
    def test_get_execution_history(self):
        mgr = _make_manager()
        mgr.backend.fetchall.return_value = [
            {
                "schedule_id": "sched-1",
                "workflow_id": "wf-1",
                "executed_at": "2025-06-15T12:00:00",
                "status": "completed",
                "duration_seconds": 3.5,
                "error": None,
            },
        ]

        logs = mgr.get_execution_history("sched-1", limit=5)

        assert len(logs) == 1
        assert logs[0].schedule_id == "sched-1"
        assert logs[0].status == "completed"
        assert logs[0].duration_seconds == 3.5

    def test_get_execution_history_empty(self):
        mgr = _make_manager()
        mgr.backend.fetchall.return_value = []

        logs = mgr.get_execution_history("sched-1")

        assert logs == []


class TestLoadAllSchedules:
    def test_load_schedules_from_db(self):
        backend = _mock_backend()
        backend.fetchall.return_value = [
            {
                "id": "sched-db",
                "workflow_id": "wf-db",
                "name": "DB Schedule",
                "description": "",
                "schedule_type": "interval",
                "cron_config": None,
                "interval_config": '{"minutes": 10, "seconds": 0, "hours": 0, "days": 0}',
                "variables": None,
                "status": "active",
                "created_at": "2025-01-01T00:00:00",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
            },
        ]

        mgr = _make_manager(backend=backend)
        # start() calls _load_all_schedules and registers active jobs
        mgr.scheduler.running = False
        mgr.scheduler.get_job.return_value = None
        mgr.start()

        assert "sched-db" in mgr._schedules
        loaded = mgr._schedules["sched-db"]
        assert loaded.schedule_type == ScheduleType.INTERVAL
        assert loaded.interval_config.minutes == 10


class TestStartShutdown:
    def test_start_starts_scheduler(self):
        mgr = _make_manager()
        mgr.scheduler.running = False

        mgr.start()

        mgr.scheduler.start.assert_called_once()

    def test_shutdown_stops_running_scheduler(self):
        mgr = _make_manager()
        mgr.scheduler.running = True

        mgr.shutdown()

        mgr.scheduler.shutdown.assert_called_once_with(wait=False)

    def test_shutdown_noop_when_not_running(self):
        mgr = _make_manager()
        mgr.scheduler.running = False

        mgr.shutdown()

        mgr.scheduler.shutdown.assert_not_called()
