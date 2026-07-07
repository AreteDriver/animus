"""Tests for the daemon package: ResourceGuard, SessionManager, TaskScheduler, events, and core."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from animus.daemon.core import AnimusDaemon, DaemonConfig, DaemonState
from animus.daemon.events import (
    DaemonEvent,
    EventPriority,
    EventType,
    FileWatchEvent,
    FileWatchHandler,
    MCPEvent,
    MCPHandler,
    ScheduledEvent,
    SignalEvent,
    TimerEvent,
    TimerHandler,
    WebhookEvent,
    WebhookHandler,
)
from animus.daemon.resource_guard import ResourceGuard, ResourceLimits
from animus.daemon.scheduler import ScheduledTask, ScheduleType, TaskScheduler
from animus.daemon.session_manager import SessionManager, WarmSession


# ── ResourceGuard Tests ───────────────────────────────────────────


class TestResourceGuard:
    def test_init(self):
        limits = ResourceLimits(max_concurrent_tasks=5, max_tokens_per_minute=1000)
        rg = ResourceGuard(limits=limits)
        assert rg.limits.max_concurrent_tasks == 5
        assert rg.limits.max_tokens_per_minute == 1000
        assert rg.can_execute is True

    def test_acquire_release_slot(self):
        limits = ResourceLimits(max_concurrent_tasks=2, task_cooldown_seconds=0)
        rg = ResourceGuard(limits=limits)
        ok1, _ = rg.acquire_task_slot("task-1")
        assert ok1 is True
        ok2, _ = rg.acquire_task_slot("task-2")
        assert ok2 is True
        ok3, reason = rg.acquire_task_slot("task-3")
        assert ok3 is False
        assert "concurrency" in reason.lower() or "Max" in reason

        rg.release_task_slot("task-1")
        assert len(rg._active_tasks) == 1

        rg.release_task_slot("task-2")
        assert len(rg._active_tasks) == 0

    def test_token_window(self):
        limits = ResourceLimits(max_tokens_per_minute=100)
        rg = ResourceGuard(limits=limits)
        ok = rg.report_tokens(50)
        assert ok is True
        ok = rg.report_tokens(60)
        assert ok is False  # Over limit

    def test_emergency_stop(self):
        rg = ResourceGuard()
        assert rg.can_execute is True
        rg.emergency_stop("test")
        assert rg.can_execute is False
        rg.emergency_clear()
        assert rg.can_execute is True

    def test_can_execute_limits(self):
        limits = ResourceLimits(max_concurrent_tasks=1)
        rg = ResourceGuard(limits=limits)
        rg.acquire_task_slot("only-slot")
        assert rg.can_execute is False


# ── SessionManager Tests ──────────────────────────────────────────


class TestSessionManager:
    def test_create_session(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SessionManager(persistence_dir=td, max_sessions=5)
            session = sm.create(user_id="user-1")
            assert session.user_id == "user-1"
            assert session.is_complete is False
            assert len(sm.list_sessions()) == 1

    def test_session_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SessionManager(persistence_dir=td, max_sessions=5)
            session = sm.create(user_id="user-1")
            session.original_prompt = "hello"
            sm.update(session)

            # New manager loads existing sessions
            sm2 = SessionManager(persistence_dir=td, max_sessions=5)
            loaded = sm2.get(session.session_id)
            assert loaded is not None
            assert loaded.original_prompt == "hello"

    def test_prune_max_sessions(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SessionManager(persistence_dir=td, max_sessions=2)
            s1 = sm.create(user_id="u1")
            time.sleep(0.05)
            s2 = sm.create(user_id="u2")
            time.sleep(0.05)
            s3 = sm.create(user_id="u3")
            sm.prune_old_sessions(max_age_hours=168)
            # Oldest should be pruned due to cap
            assert len(sm.list_sessions()) == 2
            assert sm.get(s1.session_id) is None

    def test_prune_stale(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SessionManager(persistence_dir=td, max_sessions=10)
            session = sm.create(user_id="stale")
            # Persist, then manually edit the file to set last_active to 2 hours ago
            stale_time = (datetime.now() - timedelta(hours=2)).isoformat()
            path = sm._session_path(session.session_id)
            data = json.loads(path.read_text())
            data["last_active"] = stale_time
            path.write_text(json.dumps(data))

            sm2 = SessionManager(persistence_dir=td, max_sessions=10)
            sm2.prune_old_sessions(max_age_hours=1)
            assert sm2.get(session.session_id) is None

    def test_complete_close(self):
        with tempfile.TemporaryDirectory() as td:
            sm = SessionManager(persistence_dir=td, max_sessions=5)
            session = sm.create(user_id="u1")
            sm.complete(session.session_id)
            loaded = sm.get(session.session_id)
            assert loaded.is_complete is True


# ── TaskScheduler Tests ─────────────────────────────────────────────


class TestTaskScheduler:
    def test_schedule_interval(self):
        with tempfile.TemporaryDirectory() as td:
            ts = TaskScheduler(persistence_dir=td)
            task = ts.schedule_interval("check fleet", seconds=60)
            assert task.schedule_type == ScheduleType.INTERVAL
            assert task.schedule_config["seconds"] == 60
            assert task.is_due  # First run is immediate

    def test_schedule_one_shot(self):
        with tempfile.TemporaryDirectory() as td:
            ts = TaskScheduler(persistence_dir=td)
            future = datetime.now() + timedelta(hours=1)
            task = ts.schedule_one_shot("future task", run_at=future)
            assert task.schedule_type == ScheduleType.ONE_SHOT
            assert not task.is_due  # Not yet

    def test_due_one_shot(self):
        with tempfile.TemporaryDirectory() as td:
            ts = TaskScheduler(persistence_dir=td)
            past = datetime.now() - timedelta(minutes=1)
            task = ts.schedule_one_shot("past task", run_at=past)
            assert task.is_due

    def test_mark_run(self):
        with tempfile.TemporaryDirectory() as td:
            ts = TaskScheduler(persistence_dir=td)
            task = ts.schedule_interval("repeating", seconds=60)
            ts.mark_run(task.task_id)
            loaded = ts.get_task(task.task_id)
            assert loaded.run_count == 1
            assert loaded.next_run is not None

    def test_cancel(self):
        with tempfile.TemporaryDirectory() as td:
            ts = TaskScheduler(persistence_dir=td)
            task = ts.schedule_interval("to cancel", seconds=60)
            assert ts.cancel(task.task_id) is True
            assert ts.get_task(task.task_id) is None
            assert ts.cancel(task.task_id) is False

    def test_cron_check(self):
        with tempfile.TemporaryDirectory() as td:
            ts = TaskScheduler(persistence_dir=td)
            now = datetime.now()
            cron = f"{now.minute} {now.hour} * * *"
            task = ts.schedule_cron("daily", cron)
            assert task.is_due

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            ts1 = TaskScheduler(persistence_dir=td)
            task = ts1.schedule_interval("persisted", seconds=300)

            ts2 = TaskScheduler(persistence_dir=td)
            loaded = ts2.get_task(task.task_id)
            assert loaded is not None
            assert loaded.description == "persisted"


# ── Event Tests ─────────────────────────────────────────────────────


class TestEvents:
    def test_daemon_event_creation(self):
        event = DaemonEvent(
            event_type=EventType.TIMER,
            payload={"data": "test"},
            priority=EventPriority.HIGH,
        )
        assert event.event_type == EventType.TIMER
        assert event.priority == EventPriority.HIGH

    def test_file_watch_event(self):
        event = FileWatchEvent(
            event_type=EventType.FILE_WATCH,
            path="/tmp/test.md",
            change_type="modified",
        )
        assert event.path == "/tmp/test.md"
        assert event.change_type == "modified"

    def test_signal_event(self):
        event = SignalEvent(event_type=EventType.SIGNAL, signal_number=15, signal_name="SIGTERM")
        assert event.signal_name == "SIGTERM"

    def test_timer_handler(self):
        handler = TimerHandler(interval_seconds=1.0)
        tick = handler.create_tick()
        assert tick.tick_number == 1
        assert tick.interval_seconds == 1.0

    def test_file_watch_handler_scan(self):
        with tempfile.TemporaryDirectory() as td:
            handler = FileWatchHandler(watch_path=td, patterns=["*.txt"])
            # Create a file
            (Path(td) / "test.txt").write_text("hello")
            events = handler.scan()
            assert len(events) == 1
            assert events[0].change_type == "created"

            # Modify
            time.sleep(0.1)
            (Path(td) / "test.txt").write_text("hello again")
            events = handler.scan()
            assert len(events) == 1
            assert events[0].change_type == "modified"

    def test_webhook_handler(self):
        handler = WebhookHandler(allowed_endpoints=["/hook1"])
        event = WebhookEvent(event_type=EventType.WEBHOOK, endpoint="/hook1", body="test")
        assert handler.can_handle(event) is True

        event2 = WebhookEvent(event_type=EventType.WEBHOOK, endpoint="/hook2", body="test")
        assert handler.can_handle(event2) is True
        result = asyncio.run(handler.handle(event2))
        assert result["handled"] is False
        assert result["reason"] == "endpoint_not_allowed"

    def test_mcp_handler(self):
        handler = MCPHandler()
        event = MCPEvent(event_type=EventType.MCP, tool_name="test_tool")
        assert handler.can_handle(event) is True
        result = asyncio.run(handler.handle(event))
        assert result["handled"] is True
        assert result["tool"] == "test_tool"


# ── Daemon Core Tests ───────────────────────────────────────────────


class TestDaemonCore:
    @pytest.fixture
    def temp_daemon(self):
        with tempfile.TemporaryDirectory() as td:
            config = DaemonConfig(
                persistence_dir=td,
                sessions_dir=str(Path(td) / "sessions"),
                scheduler_dir=str(Path(td) / "scheduler"),
                tick_interval=0.05,
                scheduler_check_interval=0.1,
                file_scan_interval=0.2,
                session_save_interval=0.5,
                meta_thinker_check_interval=0.5,
                max_concurrent_tasks=2,
                max_tokens_per_minute=10000,
                max_sessions=5,
                enable_file_watch=True,
                enable_scheduler=True,
            )
            daemon = AnimusDaemon(config=config)
            yield daemon

    def test_daemon_init(self, temp_daemon):
        assert temp_daemon.state == DaemonState.INIT
        assert temp_daemon.config.max_concurrent_tasks == 2

    def test_pid_file(self, temp_daemon):
        temp_daemon._write_pid()
        pid = temp_daemon._read_pid()
        assert pid == os.getpid()
        assert temp_daemon.is_running() is True
        temp_daemon._remove_pid()
        assert temp_daemon.is_running() is False

    def test_save_load_state(self, temp_daemon):
        temp_daemon.state = DaemonState.RUNNING
        temp_daemon._tick_count = 42
        temp_daemon._save_state()
        state = temp_daemon._load_state()
        assert state["tick_count"] == 42
        assert state["state"] == "running"

    def test_status(self, temp_daemon):
        status = temp_daemon.get_status()
        assert "state" in status
        assert "running" in status
        assert "uptime_seconds" in status

    def test_schedule_background_task(self, temp_daemon):
        task = temp_daemon.schedule_background_task("test bg", seconds=300)
        assert task.description == "test bg"
        assert temp_daemon.scheduler.get_task(task.task_id) is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, temp_daemon):
        started = await temp_daemon.start()
        assert started is True
        assert temp_daemon.state == DaemonState.RUNNING
        await temp_daemon.stop()
        assert temp_daemon.state == DaemonState.STOPPED

    @pytest.mark.asyncio
    async def test_double_start(self, temp_daemon):
        started = await temp_daemon.start()
        assert started is True
        started2 = await temp_daemon.start()
        assert started2 is False
        await temp_daemon.stop()

    @pytest.mark.asyncio
    async def test_tick(self, temp_daemon):
        await temp_daemon.start()
        initial_count = temp_daemon._tick_count
        await temp_daemon._tick()
        assert temp_daemon._tick_count == initial_count + 1
        await temp_daemon.stop()

    @pytest.mark.asyncio
    async def test_process_timer_event(self, temp_daemon):
        await temp_daemon.start()
        timer_event = TimerEvent(event_type=EventType.TIMER, tick_number=1)
        result = await temp_daemon._dispatch_event(timer_event)
        assert result["handled"] is True
        await temp_daemon.stop()

    @pytest.mark.asyncio
    async def test_process_file_event(self, temp_daemon):
        await temp_daemon.start()
        file_event = FileWatchEvent(event_type=EventType.FILE_WATCH, path="/tmp/test.txt", change_type="created")
        result = await temp_daemon._dispatch_event(file_event)
        # Handler won't handle it if file doesn't match watch path
        assert isinstance(result, dict)
        await temp_daemon.stop()

    @pytest.mark.asyncio
    async def test_process_scheduled_event(self, temp_daemon):
        await temp_daemon.start()
        # Schedule a task
        task = temp_daemon.schedule_background_task("quick task", seconds=1)
        temp_daemon.scheduler.mark_run(task.task_id)

        scheduled_event = ScheduledEvent(
            event_type=EventType.SCHEDULED,
            task_id=task.task_id,
            task_description="quick task",
        )
        # Should handle without error
        await temp_daemon._handle_scheduled_task(scheduled_event)
        assert temp_daemon.stats["tasks_executed"] == 1
        await temp_daemon.stop()

    @pytest.mark.asyncio
    async def test_signal_shutdown(self, temp_daemon):
        await temp_daemon.start()
        sig_event = SignalEvent(event_type=EventType.SIGNAL, signal_number=signal.SIGTERM, signal_name="SIGTERM")
        await temp_daemon._event_queue.put(sig_event)
        await temp_daemon._process_events()
        assert temp_daemon.state == DaemonState.STOPPED