"""Additional coverage tests for schedule manager."""

import json
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")


def _make_manager():
    """Create a ScheduleManager with mocked dependencies."""
    with (
        patch("animus_forge.scheduler.schedule_manager.get_settings") as mock_settings,
        patch("animus_forge.scheduler.schedule_manager.get_database") as mock_db,
        patch("animus_forge.scheduler.schedule_manager.WorkflowEngineAdapter") as mock_engine,
        patch("animus_forge.scheduler.schedule_manager.BackgroundScheduler") as mock_scheduler,
    ):
        settings = MagicMock()
        mock_settings.return_value = settings
        backend = MagicMock()
        backend.fetchall.return_value = []
        mock_db.return_value = backend

        from animus_forge.scheduler.schedule_manager import ScheduleManager

        mgr = ScheduleManager(backend=backend)
        return mgr, backend, mock_engine.return_value, mock_scheduler.return_value


def _make_schedule(**kwargs):
    from animus_forge.scheduler.schedule_manager import (
        CronConfig,
        ScheduleStatus,
        ScheduleType,
        WorkflowSchedule,
    )

    defaults = {
        "id": "sch-1",
        "workflow_id": "wf-1",
        "name": "Test Schedule",
        "schedule_type": ScheduleType.CRON,
        "cron_config": CronConfig(minute="0", hour="*"),
        "status": ScheduleStatus.ACTIVE,
    }
    defaults.update(kwargs)
    return WorkflowSchedule(**defaults)


class TestScheduleManagerInit:
    def test_init(self):
        mgr, backend, _, _ = _make_manager()
        assert mgr is not None
        backend.executescript.assert_called_once()


class TestScheduleManagerStartShutdown:
    def test_start(self):
        mgr, backend, _, scheduler = _make_manager()
        backend.fetchall.return_value = []
        scheduler.running = False
        mgr.start()
        scheduler.start.assert_called_once()

    def test_shutdown(self):
        mgr, _, _, scheduler = _make_manager()
        scheduler.running = True
        mgr.shutdown()
        scheduler.shutdown.assert_called_once()

    def test_shutdown_not_running(self):
        mgr, _, _, scheduler = _make_manager()
        scheduler.running = False
        mgr.shutdown()
        scheduler.shutdown.assert_not_called()


class TestRowToSchedule:
    def test_valid_cron_row(self):
        mgr, _, _, _ = _make_manager()
        row = {
            "id": "sch-1",
            "workflow_id": "wf-1",
            "name": "Test",
            "description": "",
            "schedule_type": "cron",
            "cron_config": json.dumps({"minute": "0", "hour": "8"}),
            "interval_config": None,
            "variables": json.dumps({"key": "val"}),
            "status": "active",
            "created_at": "2024-01-01T00:00:00",
            "last_run": None,
            "next_run": None,
            "run_count": 0,
        }
        schedule = mgr._row_to_schedule(row)
        assert schedule is not None
        assert schedule.id == "sch-1"
        assert schedule.cron_config is not None

    def test_valid_interval_row(self):
        mgr, _, _, _ = _make_manager()
        row = {
            "id": "sch-2",
            "workflow_id": "wf-1",
            "name": "Interval Test",
            "description": "",
            "schedule_type": "interval",
            "cron_config": None,
            "interval_config": json.dumps({"minutes": 30}),
            "variables": None,
            "status": "paused",
            "created_at": "2024-01-01T00:00:00",
            "last_run": "2024-01-01T12:00:00",
            "next_run": None,
            "run_count": 5,
        }
        schedule = mgr._row_to_schedule(row)
        assert schedule is not None
        assert schedule.interval_config is not None

    def test_invalid_row(self):
        mgr, _, _, _ = _make_manager()
        schedule = mgr._row_to_schedule({"bad": "data"})
        assert schedule is None


class TestLoadAllSchedules:
    def test_load_active(self):
        mgr, backend, _, _ = _make_manager()
        backend.fetchall.return_value = [
            {
                "id": "sch-1",
                "workflow_id": "wf-1",
                "name": "Test",
                "description": "",
                "schedule_type": "cron",
                "cron_config": json.dumps({"minute": "0"}),
                "interval_config": None,
                "variables": None,
                "status": "active",
                "created_at": "2024-01-01T00:00:00",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
            }
        ]
        mgr._load_all_schedules()
        assert "sch-1" in mgr._schedules

    def test_load_with_error(self):
        mgr, backend, _, _ = _make_manager()
        backend.fetchall.return_value = [{"bad": "data"}]
        mgr._load_all_schedules()
        assert len(mgr._schedules) == 0


class TestBuildTrigger:
    def test_build_cron_trigger(self):
        from animus_forge.scheduler.schedule_manager import CronConfig

        mgr, _, _, _ = _make_manager()
        schedule = _make_schedule(
            cron_config=CronConfig(minute="30", hour="8", day_of_week="mon-fri")
        )
        trigger = mgr._create_trigger(schedule)
        assert trigger is not None

    def test_build_interval_trigger(self):
        from animus_forge.scheduler.schedule_manager import (
            IntervalConfig,
            ScheduleType,
        )

        mgr, _, _, _ = _make_manager()
        schedule = _make_schedule(
            schedule_type=ScheduleType.INTERVAL,
            cron_config=None,
            interval_config=IntervalConfig(minutes=15),
        )
        trigger = mgr._create_trigger(schedule)
        assert trigger is not None

    def test_build_interval_trigger_zero_defaults(self):
        from animus_forge.scheduler.schedule_manager import (
            IntervalConfig,
            ScheduleType,
        )

        mgr, _, _, _ = _make_manager()
        schedule = _make_schedule(
            schedule_type=ScheduleType.INTERVAL,
            cron_config=None,
            interval_config=IntervalConfig(),  # All zeros
        )
        trigger = mgr._create_trigger(schedule)
        assert trigger is not None


class TestCreateSchedule:
    def test_create(self):
        mgr, backend, engine, scheduler = _make_manager()
        engine.load_workflow.return_value = MagicMock()
        backend.fetchone.return_value = None
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)

        schedule = _make_schedule()
        result = mgr.create_schedule(schedule)
        assert result is True
        assert "sch-1" in mgr._schedules

    def test_create_save_failure(self):
        mgr, backend, engine, _ = _make_manager()
        engine.load_workflow.return_value = MagicMock()
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)
        backend.execute.side_effect = RuntimeError("db error")

        schedule = _make_schedule()
        result = mgr.create_schedule(schedule)
        assert result is False

    def test_create_workflow_not_found(self):
        mgr, _, engine, _ = _make_manager()
        engine.load_workflow.return_value = None
        with pytest.raises(ValueError, match="not found"):
            mgr.create_schedule(_make_schedule())


class TestScheduleOperations:
    def _setup(self):
        mgr, backend, engine, scheduler = _make_manager()
        sch = _make_schedule()
        mgr._schedules["sch-1"] = sch
        backend.fetchone.return_value = {"id": "sch-1"}
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)
        return mgr, backend, engine, scheduler

    def test_get_schedule(self):
        mgr, _, _, _ = self._setup()
        assert mgr.get_schedule("sch-1") is not None
        assert mgr.get_schedule("nonexistent") is None

    def test_list_schedules(self):
        mgr, _, _, _ = self._setup()
        schedules = mgr.list_schedules()
        assert len(schedules) == 1

    def test_delete_schedule(self):
        mgr, backend, _, scheduler = self._setup()
        result = mgr.delete_schedule("sch-1")
        assert result is True
        assert "sch-1" not in mgr._schedules
        scheduler.remove_job.assert_called()

    def test_delete_nonexistent(self):
        mgr, _, _, _ = self._setup()
        assert mgr.delete_schedule("nonexistent") is False

    def test_pause_schedule(self):
        mgr, backend, _, scheduler = self._setup()
        from animus_forge.scheduler.schedule_manager import ScheduleStatus

        result = mgr.pause_schedule("sch-1")
        assert result is True
        assert mgr._schedules["sch-1"].status == ScheduleStatus.PAUSED

    def test_resume_schedule(self):
        from animus_forge.scheduler.schedule_manager import ScheduleStatus

        mgr, backend, _, scheduler = self._setup()
        mgr._schedules["sch-1"].status = ScheduleStatus.PAUSED
        result = mgr.resume_schedule("sch-1")
        assert result is True
        assert mgr._schedules["sch-1"].status == ScheduleStatus.ACTIVE


class TestExecuteScheduledWorkflow:
    def test_execute_success(self):
        mgr, backend, engine, scheduler = _make_manager()
        sch = _make_schedule()
        mgr._schedules["sch-1"] = sch
        backend.fetchone.return_value = {"id": "sch-1"}
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)

        mock_workflow = MagicMock()
        mock_workflow.variables = {}
        engine.load_workflow.return_value = mock_workflow
        mock_result = MagicMock()
        mock_result.status = "success"
        engine.execute_workflow.return_value = mock_result
        scheduler.get_job.return_value = None

        mgr._execute_scheduled_workflow("sch-1")
        assert mgr._schedules["sch-1"].run_count == 1

    def test_execute_nonexistent(self):
        mgr, _, _, _ = _make_manager()
        mgr._execute_scheduled_workflow("nonexistent")  # Should not raise

    def test_execute_failure(self):
        mgr, backend, engine, scheduler = _make_manager()
        sch = _make_schedule()
        mgr._schedules["sch-1"] = sch
        backend.fetchone.return_value = {"id": "sch-1"}
        backend.transaction.return_value.__enter__ = MagicMock()
        backend.transaction.return_value.__exit__ = MagicMock(return_value=False)
        engine.load_workflow.side_effect = RuntimeError("load failed")
        scheduler.get_job.return_value = None

        mgr._execute_scheduled_workflow("sch-1")
        assert mgr._schedules["sch-1"].run_count == 1


class TestGetExecutionHistory:
    def test_get_history(self):
        mgr, backend, _, _ = _make_manager()
        backend.fetchall.return_value = [
            {
                "schedule_id": "sch-1",
                "workflow_id": "wf-1",
                "executed_at": "2024-01-01T00:00:00",
                "status": "success",
                "duration_seconds": 1.5,
                "error": None,
            }
        ]
        logs = mgr.get_execution_history("sch-1")
        assert len(logs) == 1

    def test_empty_history(self):
        mgr, backend, _, _ = _make_manager()
        backend.fetchall.return_value = []
        assert len(mgr.get_execution_history("sch-1")) == 0


class TestParseDatetime:
    def test_none(self):
        from animus_forge.scheduler.schedule_manager import _parse_datetime

        assert _parse_datetime(None) is None

    def test_datetime_object(self):
        from animus_forge.scheduler.schedule_manager import _parse_datetime

        dt = datetime(2024, 1, 1)
        assert _parse_datetime(dt) is dt

    def test_string(self):
        from animus_forge.scheduler.schedule_manager import _parse_datetime

        result = _parse_datetime("2024-01-01T00:00:00")
        assert isinstance(result, datetime)
