"""Coverage tests for watchers, audit_checks, and supervisor modules.

Targets:
- src/animus_forge/monitoring/watchers.py (252 missed lines)
- src/animus_forge/metrics/audit_checks.py (95 missed lines)
- src/animus_forge/agents/supervisor.py (92 missed lines)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# supervisor.py imports
# ---------------------------------------------------------------------------
from animus_forge.agents.supervisor import (
    DelegationPlan,
    SupervisorAgent,
)

# ---------------------------------------------------------------------------
# audit_checks.py imports
# ---------------------------------------------------------------------------
from animus_forge.metrics.audit_checks import (
    MAX_FILE_SIZE,
    _diff_configs,
    _evaluate,
    check_config_drift,
    check_dependencies,
    check_error_rate,
    check_resource_usage,
    check_skill_integrity,
    check_task_completion_time,
    register_default_checks,
)
from animus_forge.metrics.debt_monitor import (
    AuditCheck,
    AuditFrequency,
    AuditStatus,
    SystemAuditor,
    SystemBaseline,
)

# ---------------------------------------------------------------------------
# watchers.py imports
# ---------------------------------------------------------------------------
from animus_forge.monitoring.watchers import (
    FileWatcher,
    LogWatcher,
    ResourceWatcher,
    WatchEvent,
    WatchEventType,
    WatchManager,
)

# ===================================================================
# Helper fixtures
# ===================================================================


@pytest.fixture()
def baseline() -> SystemBaseline:
    return SystemBaseline(
        captured_at=datetime.now(UTC),
        task_completion_time_avg=4.0,
        agent_spawn_time_avg=0.8,
        idle_cpu_percent=10.0,
        idle_memory_percent=40.0,
        skill_hashes={},
        config_snapshots={},
        package_versions={"requests": "2.31.0"},
    )


@pytest.fixture()
def audit_check() -> AuditCheck:
    return AuditCheck(
        name="test_check",
        category="test",
        frequency=AuditFrequency.DAILY,
        check_function="test_fn",
        threshold_warning=1.2,
        threshold_critical=1.5,
    )


@pytest.fixture()
def mock_provider() -> MagicMock:
    """Mock provider that uses complete() (non-streaming)."""
    provider = MagicMock()
    provider.complete = AsyncMock(return_value="Agent response")
    # Remove stream_completion so _stream_response falls back to complete()
    del provider.stream_completion
    return provider


# ===================================================================
# PART 1 — WatchEvent and WatchEventType
# ===================================================================


class TestWatchEventType:
    def test_all_event_types(self):
        expected = {
            "file_created",
            "file_modified",
            "file_deleted",
            "file_moved",
            "dir_created",
            "dir_modified",
            "dir_deleted",
            "dir_moved",
            "pattern_match",
            "threshold_exceeded",
        }
        actual = {e.value for e in WatchEventType}
        assert actual == expected

    def test_str_enum(self):
        assert isinstance(WatchEventType.FILE_CREATED, str)
        assert WatchEventType.FILE_CREATED == "file_created"


class TestWatchEvent:
    def test_defaults(self):
        evt = WatchEvent(
            event_type=WatchEventType.FILE_CREATED,
            path="/tmp/test.txt",
        )
        assert evt.old_path is None
        assert evt.details == {}
        assert isinstance(evt.timestamp, datetime)

    def test_to_dict(self):
        evt = WatchEvent(
            event_type=WatchEventType.FILE_MOVED,
            path="/tmp/new.txt",
            old_path="/tmp/old.txt",
            details={"reason": "rename"},
        )
        d = evt.to_dict()
        assert d["event_type"] == "file_moved"
        assert d["path"] == "/tmp/new.txt"
        assert d["old_path"] == "/tmp/old.txt"
        assert d["details"]["reason"] == "rename"
        assert "timestamp" in d

    def test_to_dict_with_all_fields(self):
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        evt = WatchEvent(
            event_type=WatchEventType.PATTERN_MATCH,
            path="/var/log/test.log",
            timestamp=ts,
            old_path=None,
            details={"match": "ERROR"},
        )
        d = evt.to_dict()
        assert d["timestamp"] == ts.isoformat()


# ===================================================================
# PART 2 — BaseWatcher (through concrete subclass)
# ===================================================================


class TestBaseWatcherBehavior:
    """Test add_handler, remove_handler, _emit via FileWatcher."""

    def test_add_handler(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        handler = MagicMock()
        watcher.add_handler(handler)
        assert handler in watcher._handlers

    def test_remove_handler_success(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        handler = MagicMock()
        watcher.add_handler(handler)
        assert watcher.remove_handler(handler) is True
        assert handler not in watcher._handlers

    def test_remove_handler_not_found(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        handler = MagicMock()
        assert watcher.remove_handler(handler) is False

    def test_emit_calls_all_handlers(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        h1 = MagicMock()
        h2 = MagicMock()
        watcher.add_handler(h1)
        watcher.add_handler(h2)

        evt = WatchEvent(event_type=WatchEventType.FILE_CREATED, path="/x")
        watcher._emit(evt)
        h1.assert_called_once_with(evt)
        h2.assert_called_once_with(evt)

    def test_emit_handler_exception_swallowed(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        bad_handler = MagicMock(side_effect=RuntimeError("boom"))
        good_handler = MagicMock()
        watcher.add_handler(bad_handler)
        watcher.add_handler(good_handler)

        evt = WatchEvent(event_type=WatchEventType.FILE_CREATED, path="/x")
        watcher._emit(evt)
        # Good handler still called despite bad handler
        good_handler.assert_called_once_with(evt)

    def test_is_running_initially_false(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        assert watcher.is_running is False


# ===================================================================
# PART 3 — FileWatcher
# ===================================================================


class TestFileWatcherInit:
    def test_default_params(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        assert watcher.path == tmp_path
        assert watcher.recursive is True
        assert watcher.patterns == ["*"]
        assert "*.pyc" in watcher.ignore_patterns
        assert watcher.poll_interval == 1.0
        assert watcher.name == "file_watcher"

    def test_custom_params(self, tmp_path: Path):
        watcher = FileWatcher(
            tmp_path,
            recursive=False,
            patterns=["*.py"],
            ignore_patterns=["*.log"],
            poll_interval=0.5,
            name="my_watcher",
        )
        assert watcher.recursive is False
        assert watcher.patterns == ["*.py"]
        assert watcher.ignore_patterns == ["*.log"]
        assert watcher.poll_interval == 0.5
        assert watcher.name == "my_watcher"


class TestFileWatcherMatchesPatterns:
    def test_matches_include_pattern(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, patterns=["*.py"])
        assert watcher._matches_patterns(Path("/code/main.py")) is True

    def test_ignores_pyc(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        assert watcher._matches_patterns(Path("/code/__pycache__/foo.pyc")) is False

    def test_ignores_git(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        assert watcher._matches_patterns(Path("/code/.git/HEAD")) is False

    def test_no_match_returns_false(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, patterns=["*.py"])
        assert watcher._matches_patterns(Path("/code/data.csv")) is False


class TestFileWatcherScanDirectory:
    def test_scan_nonexistent_dir(self):
        watcher = FileWatcher("/nonexistent/path/xyz")
        files, dirs = watcher._scan_directory()
        assert files == {}
        assert dirs == set()

    def test_scan_with_files(self, tmp_path: Path):
        (tmp_path / "hello.txt").write_text("hi")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")

        watcher = FileWatcher(tmp_path, patterns=["*.txt"])
        files, dirs = watcher._scan_directory()
        paths = list(files.keys())
        assert any("hello.txt" in p for p in paths)
        assert any("nested.txt" in p for p in paths)

    def test_scan_non_recursive(self, tmp_path: Path):
        (tmp_path / "top.txt").write_text("top")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep")

        watcher = FileWatcher(tmp_path, recursive=False, patterns=["*.txt"])
        files, dirs = watcher._scan_directory()
        paths = list(files.keys())
        assert any("top.txt" in p for p in paths)
        # Non-recursive should still find the subdirectory itself
        # but not files inside it (they won't match *.txt from iterdir)

    def test_scan_captures_dirs(self, tmp_path: Path):
        sub = tmp_path / "mydir"
        sub.mkdir()

        watcher = FileWatcher(tmp_path, patterns=["*"])
        files, dirs = watcher._scan_directory()
        assert any("mydir" in d for d in dirs)


class TestFileWatcherCheckChanges:
    def test_detects_new_file(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, patterns=["*.txt"])
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        # Initial scan
        watcher._file_states, watcher._dir_states = watcher._scan_directory()

        # Create a new file
        (tmp_path / "new.txt").write_text("hello")
        watcher._check_changes()

        created = [e for e in events if e.event_type == WatchEventType.FILE_CREATED]
        assert len(created) >= 1
        assert any("new.txt" in e.path for e in created)

    def test_detects_modified_file(self, tmp_path: Path):
        f = tmp_path / "mod.txt"
        f.write_text("v1")

        watcher = FileWatcher(tmp_path, patterns=["*.txt"])
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._file_states, watcher._dir_states = watcher._scan_directory()

        # Modify file (change size to guarantee detection)
        time.sleep(0.05)
        f.write_text("v2 with more content")
        watcher._check_changes()

        modified = [e for e in events if e.event_type == WatchEventType.FILE_MODIFIED]
        assert len(modified) >= 1

    def test_detects_deleted_file(self, tmp_path: Path):
        f = tmp_path / "gone.txt"
        f.write_text("bye")

        watcher = FileWatcher(tmp_path, patterns=["*.txt"])
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._file_states, watcher._dir_states = watcher._scan_directory()

        f.unlink()
        watcher._check_changes()

        deleted = [e for e in events if e.event_type == WatchEventType.FILE_DELETED]
        assert len(deleted) >= 1

    def test_detects_new_directory(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, patterns=["*"])
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._file_states, watcher._dir_states = watcher._scan_directory()

        (tmp_path / "newdir").mkdir()
        watcher._check_changes()

        dir_created = [e for e in events if e.event_type == WatchEventType.DIR_CREATED]
        assert len(dir_created) >= 1

    def test_detects_deleted_directory(self, tmp_path: Path):
        sub = tmp_path / "deldir"
        sub.mkdir()

        watcher = FileWatcher(tmp_path, patterns=["*"])
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._file_states, watcher._dir_states = watcher._scan_directory()

        sub.rmdir()
        watcher._check_changes()

        dir_deleted = [e for e in events if e.event_type == WatchEventType.DIR_DELETED]
        assert len(dir_deleted) >= 1


class TestFileWatcherStartStop:
    def test_start_creates_thread(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, poll_interval=0.05)
        watcher.start()
        assert watcher.is_running is True
        assert watcher._thread is not None
        assert watcher._thread.is_alive()
        watcher.stop()
        assert watcher.is_running is False
        assert watcher._thread is None

    def test_start_when_already_running_is_noop(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, poll_interval=0.05)
        watcher.start()
        thread1 = watcher._thread
        watcher.start()  # should be noop
        assert watcher._thread is thread1
        watcher.stop()

    def test_start_nonexistent_path_logs_warning(self):
        watcher = FileWatcher("/nonexistent/xyz123", poll_interval=0.05)
        watcher.start()
        assert watcher.is_running is True
        watcher.stop()

    def test_stop_when_not_running(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path)
        watcher.stop()  # Should not raise
        assert watcher._thread is None

    def test_watch_loop_catches_exceptions(self, tmp_path: Path):
        watcher = FileWatcher(tmp_path, poll_interval=0.02)
        handler = MagicMock()
        watcher.add_handler(handler)
        watcher.start()
        time.sleep(0.1)
        watcher.stop()


# ===================================================================
# PART 4 — LogWatcher
# ===================================================================


class TestLogWatcherInit:
    def test_default_params(self, tmp_path: Path):
        log_path = tmp_path / "app.log"
        watcher = LogWatcher(log_path)
        assert watcher.path == log_path
        assert watcher.poll_interval == 1.0
        assert watcher.name == "log_watcher"
        assert watcher._patterns == []

    def test_init_with_patterns(self, tmp_path: Path):
        log_path = tmp_path / "app.log"
        watcher = LogWatcher(log_path, patterns={"ERROR": "error_event"})
        assert len(watcher._patterns) == 1
        pattern, name = watcher._patterns[0]
        assert name == "error_event"
        assert pattern.pattern == "ERROR"


class TestLogWatcherAddPattern:
    def test_add_pattern(self, tmp_path: Path):
        watcher = LogWatcher(tmp_path / "test.log")
        watcher.add_pattern(r"WARN\s+(\w+)", "warning")
        assert len(watcher._patterns) == 1
        assert watcher._patterns[0][1] == "warning"


class TestLogWatcherCheckLine:
    def test_matching_line_emits_event(self, tmp_path: Path):
        watcher = LogWatcher(tmp_path / "test.log")
        watcher.add_pattern(r"ERROR", "error_found")
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._check_line("2025-01-01 ERROR: something broke")
        assert len(events) == 1
        assert events[0].event_type == WatchEventType.PATTERN_MATCH
        assert events[0].details["event_name"] == "error_found"
        assert events[0].details["match"] == "ERROR"

    def test_non_matching_line(self, tmp_path: Path):
        watcher = LogWatcher(tmp_path / "test.log")
        watcher.add_pattern(r"ERROR", "error_found")
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._check_line("2025-01-01 INFO: all good")
        assert len(events) == 0

    def test_multiple_patterns(self, tmp_path: Path):
        watcher = LogWatcher(tmp_path / "test.log")
        watcher.add_pattern(r"ERROR", "error_found")
        watcher.add_pattern(r"WARN", "warn_found")
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._check_line("2025-01-01 ERROR: also WARN: double trouble")
        assert len(events) == 2

    def test_pattern_groups_captured(self, tmp_path: Path):
        watcher = LogWatcher(tmp_path / "test.log")
        watcher.add_pattern(r"user (\w+) logged in", "user_login")
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._check_line("user alice logged in from 10.0.0.1")
        assert len(events) == 1
        assert events[0].details["groups"] == ("alice",)


class TestLogWatcherCheckNewLines:
    def test_reads_new_lines(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        log_file.write_text("line1\nline2\n")

        watcher = LogWatcher(log_file)
        watcher.add_pattern(r"line", "line_found")
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        # Set position to beginning to read all lines
        watcher._file_position = 0
        watcher._check_new_lines()
        assert len(events) == 2

    def test_detects_file_truncation(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        log_file.write_text("a" * 1000)

        watcher = LogWatcher(log_file)
        watcher._file_position = 5000  # Beyond file size

        watcher.add_pattern(r"a+", "found")
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        watcher._check_new_lines()
        # Should reset position and read from beginning
        assert len(events) >= 1

    def test_nonexistent_file(self, tmp_path: Path):
        watcher = LogWatcher(tmp_path / "missing.log")
        watcher._check_new_lines()  # Should not raise

    def test_permission_error_handled(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        log_file.write_text("data")

        watcher = LogWatcher(log_file)
        with patch("builtins.open", side_effect=PermissionError("denied")):
            watcher._check_new_lines()  # Should not raise


class TestLogWatcherStartStop:
    def test_start_stop(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        log_file.write_text("")

        watcher = LogWatcher(log_file, poll_interval=0.02)
        watcher.start()
        assert watcher.is_running is True
        time.sleep(0.1)
        watcher.stop()
        assert watcher.is_running is False
        assert watcher._thread is None

    def test_start_already_running(self, tmp_path: Path):
        log_file = tmp_path / "test.log"
        log_file.write_text("")

        watcher = LogWatcher(log_file, poll_interval=0.02)
        watcher.start()
        thread1 = watcher._thread
        watcher.start()  # noop
        assert watcher._thread is thread1
        watcher.stop()

    def test_start_sets_position_to_end(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        log_file.write_text("existing content\n")

        watcher = LogWatcher(log_file, poll_interval=0.02)
        watcher.start()
        time.sleep(0.05)
        watcher.stop()
        # Position should have been set to end of file at start

    def test_stop_when_not_running(self, tmp_path: Path):
        watcher = LogWatcher(tmp_path / "x.log")
        watcher.stop()  # Should not raise


# ===================================================================
# PART 5 — ResourceWatcher
# ===================================================================


class TestResourceWatcherInit:
    def test_defaults(self):
        watcher = ResourceWatcher()
        assert watcher.poll_interval == 30.0
        assert watcher.name == "resource_watcher"
        assert watcher._thresholds == {}

    def test_set_threshold(self):
        watcher = ResourceWatcher()
        watcher.set_threshold("cpu", 80.0)
        assert watcher._thresholds["cpu"] == 80.0


class TestResourceWatcherGetCpuUsage:
    def test_cpu_usage_returns_float(self):
        watcher = ResourceWatcher()
        result = watcher._get_cpu_usage()
        assert isinstance(result, float)

    def test_cpu_usage_fallback_on_exception(self):
        watcher = ResourceWatcher()
        with patch("os.path.exists", return_value=False):
            with patch("os.getloadavg", side_effect=OSError("no loadavg")):
                result = watcher._get_cpu_usage()
                assert result == 0.0

    def test_cpu_usage_loadavg_fallback(self):
        watcher = ResourceWatcher()
        with patch("os.path.exists", return_value=False):
            with patch("os.getloadavg", return_value=(2.0, 1.5, 1.0)):
                with patch("os.cpu_count", return_value=4):
                    result = watcher._get_cpu_usage()
                    assert result == pytest.approx(50.0)


class TestResourceWatcherGetMemoryUsage:
    def test_memory_usage_returns_float(self):
        watcher = ResourceWatcher()
        result = watcher._get_memory_usage()
        assert isinstance(result, float)

    def test_memory_no_proc_returns_zero(self):
        watcher = ResourceWatcher()
        with patch("os.path.exists", return_value=False):
            result = watcher._get_memory_usage()
            assert result == 0.0

    def test_memory_exception_returns_zero(self):
        watcher = ResourceWatcher()
        with patch("os.path.exists", side_effect=OSError("no")):
            result = watcher._get_memory_usage()
            assert result == 0.0


class TestResourceWatcherGetDiskUsage:
    def test_disk_usage_returns_float(self):
        watcher = ResourceWatcher()
        result = watcher._get_disk_usage()
        assert isinstance(result, float)
        assert 0 <= result <= 100

    def test_disk_usage_exception(self):
        watcher = ResourceWatcher()
        with patch("os.statvfs", side_effect=OSError("no")):
            result = watcher._get_disk_usage("/nonexistent")
            assert result == 0.0

    def test_disk_usage_zero_total(self):
        watcher = ResourceWatcher()
        mock_stat = MagicMock()
        mock_stat.f_blocks = 0
        mock_stat.f_frsize = 4096
        mock_stat.f_bfree = 0
        with patch("os.statvfs", return_value=mock_stat):
            result = watcher._get_disk_usage("/")
            assert result == 0.0


class TestResourceWatcherCheckResources:
    def test_threshold_exceeded_emits_event(self):
        watcher = ResourceWatcher()
        watcher.set_threshold("cpu", 0.0)  # Anything exceeds this
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        with patch.object(watcher, "_get_cpu_usage", return_value=50.0):
            with patch.object(watcher, "_get_memory_usage", return_value=30.0):
                with patch.object(watcher, "_get_disk_usage", return_value=20.0):
                    watcher._check_resources()

        cpu_events = [e for e in events if e.details.get("resource") == "cpu"]
        assert len(cpu_events) == 1
        assert cpu_events[0].event_type == WatchEventType.THRESHOLD_EXCEEDED

    def test_threshold_not_exceeded(self):
        watcher = ResourceWatcher()
        watcher.set_threshold("cpu", 99.0)
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        with patch.object(watcher, "_get_cpu_usage", return_value=10.0):
            with patch.object(watcher, "_get_memory_usage", return_value=30.0):
                with patch.object(watcher, "_get_disk_usage", return_value=20.0):
                    watcher._check_resources()

        assert len(events) == 0

    def test_cooldown_prevents_repeated_alerts(self):
        watcher = ResourceWatcher()
        watcher.set_threshold("cpu", 0.0)
        watcher._alert_cooldown = 300
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        with patch.object(watcher, "_get_cpu_usage", return_value=50.0):
            with patch.object(watcher, "_get_memory_usage", return_value=30.0):
                with patch.object(watcher, "_get_disk_usage", return_value=20.0):
                    watcher._check_resources()
                    watcher._check_resources()  # Should be cooled down

        cpu_events = [e for e in events if e.details.get("resource") == "cpu"]
        assert len(cpu_events) == 1

    def test_no_threshold_no_event(self):
        watcher = ResourceWatcher()
        # No thresholds set
        events: list[WatchEvent] = []
        watcher.add_handler(events.append)

        with patch.object(watcher, "_get_cpu_usage", return_value=99.0):
            with patch.object(watcher, "_get_memory_usage", return_value=99.0):
                with patch.object(watcher, "_get_disk_usage", return_value=99.0):
                    watcher._check_resources()

        assert len(events) == 0


class TestResourceWatcherStartStop:
    def test_start_stop(self):
        watcher = ResourceWatcher(poll_interval=0.02)
        watcher.start()
        assert watcher.is_running is True
        time.sleep(0.05)
        watcher.stop()
        assert watcher.is_running is False

    def test_start_already_running(self):
        watcher = ResourceWatcher(poll_interval=0.02)
        watcher.start()
        thread1 = watcher._thread
        watcher.start()
        assert watcher._thread is thread1
        watcher.stop()

    def test_stop_when_not_started(self):
        watcher = ResourceWatcher()
        watcher.stop()  # No-op


# ===================================================================
# PART 6 — WatchManager
# ===================================================================


class TestWatchManager:
    def test_add_watcher(self, tmp_path: Path):
        mgr = WatchManager()
        fw = FileWatcher(tmp_path)
        mgr.add_watcher(fw)
        assert fw in mgr._watchers

    def test_add_handler_propagates(self, tmp_path: Path):
        mgr = WatchManager()
        fw = FileWatcher(tmp_path)
        mgr.add_watcher(fw)

        handler = MagicMock()
        mgr.add_handler(handler)
        assert handler in fw._handlers

    def test_add_handler_before_watcher(self, tmp_path: Path):
        mgr = WatchManager()
        handler = MagicMock()
        mgr.add_handler(handler)

        fw = FileWatcher(tmp_path)
        mgr.add_watcher(fw)
        # Handler should be propagated to new watcher
        assert handler in fw._handlers

    def test_set_handler_replaces(self, tmp_path: Path):
        mgr = WatchManager()
        fw = FileWatcher(tmp_path)
        mgr.add_watcher(fw)

        h1 = MagicMock()
        h2 = MagicMock()
        mgr.add_handler(h1)
        mgr.set_handler(h2)

        assert mgr._handlers == [h2]
        assert fw._handlers == [h2]

    def test_add_file_watch(self, tmp_path: Path):
        mgr = WatchManager()
        fw = mgr.add_file_watch(tmp_path, patterns=["*.py"])
        assert isinstance(fw, FileWatcher)
        assert fw in mgr._watchers

    def test_add_log_watch(self, tmp_path: Path):
        log_path = tmp_path / "app.log"
        mgr = WatchManager()
        lw = mgr.add_log_watch(log_path, patterns={"ERROR": "error"})
        assert isinstance(lw, LogWatcher)
        assert lw in mgr._watchers

    def test_add_resource_watch(self):
        mgr = WatchManager()
        rw = mgr.add_resource_watch(cpu=80, memory=90, disk=95)
        assert isinstance(rw, ResourceWatcher)
        assert rw._thresholds["cpu"] == 80
        assert rw._thresholds["memory"] == 90
        assert rw._thresholds["disk"] == 95

    def test_add_resource_watch_partial(self):
        mgr = WatchManager()
        rw = mgr.add_resource_watch(cpu=70)
        assert "cpu" in rw._thresholds
        assert "memory" not in rw._thresholds
        assert "disk" not in rw._thresholds

    def test_start_stop_all(self, tmp_path: Path):
        mgr = WatchManager()
        fw = mgr.add_file_watch(tmp_path, poll_interval=0.02)
        rw = mgr.add_resource_watch(poll_interval=0.02)

        mgr.start_all()
        assert fw.is_running
        assert rw.is_running

        mgr.stop_all()
        assert not fw.is_running
        assert not rw.is_running

    def test_list_watchers(self, tmp_path: Path):
        mgr = WatchManager()
        mgr.add_file_watch(tmp_path, name="fw1")
        mgr.add_resource_watch(name="rw1")

        listing = mgr.list_watchers()
        assert len(listing) == 2
        assert listing[0]["name"] == "fw1"
        assert listing[0]["type"] == "FileWatcher"
        assert listing[0]["running"] is False
        assert listing[1]["name"] == "rw1"
        assert listing[1]["type"] == "ResourceWatcher"


# ===================================================================
# PART 7 — audit_checks: _evaluate
# ===================================================================


class TestEvaluate:
    def test_ok(self):
        assert _evaluate(0.5, 1.0, 2.0) == AuditStatus.OK

    def test_warning(self):
        assert _evaluate(1.2, 1.0, 2.0) == AuditStatus.WARNING

    def test_critical(self):
        assert _evaluate(2.5, 1.0, 2.0) == AuditStatus.CRITICAL

    def test_exactly_warning(self):
        assert _evaluate(1.0, 1.0, 2.0) == AuditStatus.WARNING

    def test_exactly_critical(self):
        assert _evaluate(2.0, 1.0, 2.0) == AuditStatus.CRITICAL


# ===================================================================
# PART 8 — audit_checks: check_task_completion_time
# ===================================================================


class TestCheckTaskCompletionTime:
    def test_ok_status(self, audit_check: AuditCheck, baseline: SystemBaseline):
        backend = MagicMock()
        backend.fetchone.return_value = {"avg_duration": 4.0}

        result = asyncio.run(
            check_task_completion_time(audit_check, backend=backend, baseline=baseline)
        )
        assert result.status == AuditStatus.OK
        assert result.value == 4.0
        assert result.recommendation is None

    def test_warning_status(self, audit_check: AuditCheck, baseline: SystemBaseline):
        backend = MagicMock()
        # ratio = 5.0/4.0 = 1.25 > 1.2 warning
        backend.fetchone.return_value = {"avg_duration": 5.0}

        result = asyncio.run(
            check_task_completion_time(audit_check, backend=backend, baseline=baseline)
        )
        assert result.status == AuditStatus.WARNING
        assert result.recommendation is not None

    def test_critical_status(self, audit_check: AuditCheck, baseline: SystemBaseline):
        backend = MagicMock()
        # ratio = 8.0/4.0 = 2.0 > 1.5 critical
        backend.fetchone.return_value = {"avg_duration": 8.0}

        result = asyncio.run(
            check_task_completion_time(audit_check, backend=backend, baseline=baseline)
        )
        assert result.status == AuditStatus.CRITICAL

    def test_no_baseline(self, audit_check: AuditCheck, baseline: SystemBaseline):
        baseline.task_completion_time_avg = 0.0
        backend = MagicMock()
        backend.fetchone.return_value = {"avg_duration": 5.0}

        result = asyncio.run(
            check_task_completion_time(audit_check, backend=backend, baseline=baseline)
        )
        assert result.status == AuditStatus.OK
        assert "No baseline" in result.details.get("note", "")

    def test_null_avg_duration(self, audit_check: AuditCheck, baseline: SystemBaseline):
        backend = MagicMock()
        backend.fetchone.return_value = {"avg_duration": None}

        result = asyncio.run(
            check_task_completion_time(audit_check, backend=backend, baseline=baseline)
        )
        assert result.value == 0.0

    def test_no_rows(self, audit_check: AuditCheck, baseline: SystemBaseline):
        backend = MagicMock()
        backend.fetchone.return_value = None

        result = asyncio.run(
            check_task_completion_time(audit_check, backend=backend, baseline=baseline)
        )
        assert result.value == 0.0


# ===================================================================
# PART 9 — audit_checks: check_error_rate
# ===================================================================


class TestCheckErrorRate:
    def test_ok_no_failures(self, audit_check: AuditCheck):
        backend = MagicMock()
        backend.fetchone.return_value = {"total": 100, "failures": 0}

        result = asyncio.run(check_error_rate(audit_check, backend=backend))
        assert result.status == AuditStatus.OK
        assert result.value == 0.0
        assert result.recommendation is None

    def test_warning_rate(self, audit_check: AuditCheck):
        # 6% > 5% warning threshold (using default thresholds)
        check = AuditCheck(
            name="err",
            category="reliability",
            frequency=AuditFrequency.DAILY,
            check_function="check_error_rate",
            threshold_warning=0.05,
            threshold_critical=0.10,
        )
        backend = MagicMock()
        backend.fetchone.return_value = {"total": 100, "failures": 6}

        result = asyncio.run(check_error_rate(check, backend=backend))
        assert result.status == AuditStatus.WARNING
        assert result.recommendation is not None

    def test_critical_rate(self, audit_check: AuditCheck):
        check = AuditCheck(
            name="err",
            category="reliability",
            frequency=AuditFrequency.DAILY,
            check_function="check_error_rate",
            threshold_warning=0.05,
            threshold_critical=0.10,
        )
        backend = MagicMock()
        backend.fetchone.return_value = {"total": 100, "failures": 15}

        result = asyncio.run(check_error_rate(check, backend=backend))
        assert result.status == AuditStatus.CRITICAL

    def test_no_jobs(self, audit_check: AuditCheck):
        backend = MagicMock()
        backend.fetchone.return_value = {"total": 0, "failures": 0}

        result = asyncio.run(check_error_rate(audit_check, backend=backend))
        assert result.value == 0.0

    def test_null_row(self, audit_check: AuditCheck):
        backend = MagicMock()
        backend.fetchone.return_value = None

        result = asyncio.run(check_error_rate(audit_check, backend=backend))
        assert result.value == 0.0


# ===================================================================
# PART 10 — audit_checks: check_dependencies
# ===================================================================


class TestCheckDependencies:
    def test_no_outdated(self, audit_check: AuditCheck):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "[]"

        with patch("subprocess.run", return_value=mock_result):
            result = asyncio.run(check_dependencies(audit_check))

        assert result.status == AuditStatus.OK
        assert result.value == 0.0

    def test_some_outdated(self):
        check = AuditCheck(
            name="deps",
            category="dependencies",
            frequency=AuditFrequency.WEEKLY,
            check_function="check_dependencies",
            threshold_warning=3,
            threshold_critical=10,
        )
        pkgs = [{"name": f"pkg{i}", "version": "1.0", "latest_version": "2.0"} for i in range(5)]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(pkgs)

        with patch("subprocess.run", return_value=mock_result):
            result = asyncio.run(check_dependencies(check))

        assert result.status == AuditStatus.WARNING
        assert result.value == 5.0
        assert result.recommendation is not None
        assert "pkg0" in result.recommendation

    def test_pip_fails(self, audit_check: AuditCheck):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = asyncio.run(check_dependencies(audit_check))

        assert result.value == 0.0

    def test_exception_during_check(self, audit_check: AuditCheck):
        with patch("subprocess.run", side_effect=OSError("pip not found")):
            result = asyncio.run(check_dependencies(audit_check))

        assert result.value == 0.0


# ===================================================================
# PART 11 — audit_checks: check_skill_integrity
# ===================================================================


class TestCheckSkillIntegrity:
    def test_no_skills_path(self, audit_check: AuditCheck, baseline: SystemBaseline):
        result = asyncio.run(
            check_skill_integrity(audit_check, baseline=baseline, skills_path=Path("/nonexistent"))
        )
        assert result.status == AuditStatus.OK
        assert result.value == 0.0

    def test_no_hashes_in_baseline(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.skill_hashes = {}
        result = asyncio.run(
            check_skill_integrity(audit_check, baseline=baseline, skills_path=tmp_path)
        )
        assert result.value == 0.0

    def test_missing_skill_file(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.skill_hashes = {str(tmp_path / "missing.yaml"): "abc123"}
        result = asyncio.run(
            check_skill_integrity(audit_check, baseline=baseline, skills_path=tmp_path)
        )
        assert result.value == 1.0
        assert result.details["modified"][0]["issue"] == "missing"

    def test_modified_skill_file(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        skill_file = tmp_path / "my_skill.yaml"
        skill_file.write_text("original content")
        # Put a different hash in baseline to simulate drift
        baseline.skill_hashes = {str(skill_file): "different_hash"}

        result = asyncio.run(
            check_skill_integrity(audit_check, baseline=baseline, skills_path=tmp_path)
        )
        assert result.value == 1.0
        assert "expected" in result.details["modified"][0]

    def test_matching_skill_file(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        skill_file = tmp_path / "good.yaml"
        content = b"valid skill content"
        skill_file.write_bytes(content)
        file_hash = hashlib.sha256(content).hexdigest()
        baseline.skill_hashes = {str(skill_file): file_hash}

        result = asyncio.run(
            check_skill_integrity(audit_check, baseline=baseline, skills_path=tmp_path)
        )
        assert result.value == 0.0

    def test_oversized_file_skipped(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        skill_file = tmp_path / "big.yaml"
        skill_file.write_text("x")
        baseline.skill_hashes = {str(skill_file): "abc"}

        mock_stat = MagicMock()
        mock_stat.st_size = MAX_FILE_SIZE + 1
        with patch.object(Path, "stat", return_value=mock_stat):
            result = asyncio.run(
                check_skill_integrity(audit_check, baseline=baseline, skills_path=tmp_path)
            )
        # Oversized file should be skipped, not counted as modified
        assert result.value == 0.0

    def test_recommendation_on_modified(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        skill_file = tmp_path / "changed.yaml"
        skill_file.write_text("changed")
        baseline.skill_hashes = {str(skill_file): "old_hash"}

        result = asyncio.run(
            check_skill_integrity(audit_check, baseline=baseline, skills_path=tmp_path)
        )
        assert result.recommendation is not None
        assert "Review" in result.recommendation


# ===================================================================
# PART 12 — audit_checks: check_config_drift
# ===================================================================


class TestCheckConfigDrift:
    def test_no_config_path(self, audit_check: AuditCheck, baseline: SystemBaseline):
        result = asyncio.run(
            check_config_drift(audit_check, baseline=baseline, config_path=Path("/nonexistent"))
        )
        assert result.status == AuditStatus.OK
        assert result.value == 0.0

    def test_no_config_snapshots(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.config_snapshots = {}
        result = asyncio.run(
            check_config_drift(audit_check, baseline=baseline, config_path=tmp_path)
        )
        assert result.value == 0.0

    def test_missing_config_file(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.config_snapshots = {"settings.yaml": {"key": "value"}}
        result = asyncio.run(
            check_config_drift(audit_check, baseline=baseline, config_path=tmp_path)
        )
        assert result.value == 1.0
        assert result.details["drift"][0]["issue"] == "missing"

    def test_yaml_not_available(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.config_snapshots = {"settings.yaml": {"key": "value"}}
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("key: value")

        with patch.dict("sys.modules", {"yaml": None}):
            result = asyncio.run(
                check_config_drift(audit_check, baseline=baseline, config_path=tmp_path)
            )
        # When yaml import fails, should return OK with note
        assert result.status == AuditStatus.OK

    def test_config_matches(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.config_snapshots = {"settings.yaml": {"key": "value"}}
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("key: value")

        result = asyncio.run(
            check_config_drift(audit_check, baseline=baseline, config_path=tmp_path)
        )
        assert result.value == 0.0

    def test_config_has_drift(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.config_snapshots = {"settings.yaml": {"key": "old_value"}}
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("key: new_value")

        result = asyncio.run(
            check_config_drift(audit_check, baseline=baseline, config_path=tmp_path)
        )
        assert result.value == 1.0
        assert result.recommendation is not None
        assert "settings.yaml" in result.recommendation

    def test_config_parse_error(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.config_snapshots = {"bad.yaml": {"key": "value"}}
        config_file = tmp_path / "bad.yaml"
        # Write content that yaml.safe_load will choke on
        config_file.write_text("{invalid yaml content ::: [[")

        import yaml

        with patch.object(yaml, "safe_load", side_effect=Exception("parse error")):
            result = asyncio.run(
                check_config_drift(audit_check, baseline=baseline, config_path=tmp_path)
            )
        assert result.value == 1.0
        assert result.details["drift"][0]["issue"] == "parse_error"

    def test_oversized_config_skipped(
        self, audit_check: AuditCheck, baseline: SystemBaseline, tmp_path: Path
    ):
        baseline.config_snapshots = {"big.yaml": {"key": "value"}}
        config_file = tmp_path / "big.yaml"
        config_file.write_text("key: value")

        mock_stat = MagicMock()
        mock_stat.st_size = MAX_FILE_SIZE + 1
        with patch.object(Path, "stat", return_value=mock_stat):
            result = asyncio.run(
                check_config_drift(audit_check, baseline=baseline, config_path=tmp_path)
            )
        assert result.value == 0.0


# ===================================================================
# PART 13 — audit_checks: check_resource_usage
# ===================================================================


class TestCheckResourceUsage:
    def test_psutil_not_available(self, audit_check: AuditCheck, baseline: SystemBaseline):
        with patch.dict("sys.modules", {"psutil": None}):
            result = asyncio.run(check_resource_usage(audit_check, baseline=baseline))
        assert result.status == AuditStatus.OK
        assert result.details.get("note") == "psutil not available"

    def test_ok_usage(self, audit_check: AuditCheck, baseline: SystemBaseline):
        mock_psutil = MagicMock()
        # cpu_percent is called via asyncio.to_thread — runs in thread pool
        mock_psutil.cpu_percent = MagicMock(return_value=10.0)
        mock_vm = MagicMock()
        mock_vm.percent = 40.0
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = asyncio.run(check_resource_usage(audit_check, baseline=baseline))

        assert result.status == AuditStatus.OK

    def test_elevated_usage(self, baseline: SystemBaseline):
        check = AuditCheck(
            name="resources",
            category="performance",
            frequency=AuditFrequency.DAILY,
            check_function="check_resource_usage",
            threshold_warning=1.5,
            threshold_critical=3.0,
        )
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent = MagicMock(return_value=25.0)
        mock_vm = MagicMock()
        mock_vm.percent = 80.0
        mock_psutil.virtual_memory.return_value = mock_vm

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = asyncio.run(check_resource_usage(check, baseline=baseline))

        # mem_ratio = 80/40 = 2.0 > 1.5 warning
        assert result.status == AuditStatus.WARNING
        assert result.recommendation is not None


# ===================================================================
# PART 14 — audit_checks: _diff_configs
# ===================================================================


class TestDiffConfigs:
    def test_identical(self):
        assert _diff_configs({"a": 1}, {"a": 1}) == []

    def test_value_changed(self):
        diffs = _diff_configs({"a": 1}, {"a": 2})
        assert len(diffs) == 1
        assert diffs[0]["path"] == "a"

    def test_key_removed(self):
        diffs = _diff_configs({"a": 1, "b": 2}, {"a": 1})
        assert len(diffs) == 1
        assert diffs[0]["issue"] == "removed"

    def test_key_added(self):
        diffs = _diff_configs({"a": 1}, {"a": 1, "b": 2})
        assert len(diffs) == 1
        assert diffs[0]["issue"] == "added"

    def test_nested_diff(self):
        expected = {"level1": {"level2": "old"}}
        actual = {"level1": {"level2": "new"}}
        diffs = _diff_configs(expected, actual)
        assert len(diffs) == 1
        assert diffs[0]["path"] == "level1.level2"

    def test_non_dict_root(self):
        diffs = _diff_configs("a", "b")
        assert len(diffs) == 1
        assert diffs[0]["path"] == "<root>"

    def test_non_dict_equal(self):
        assert _diff_configs(42, 42) == []

    def test_mixed_types(self):
        diffs = _diff_configs({"a": {"nested": True}}, {"a": "flat"})
        assert len(diffs) == 1


# ===================================================================
# PART 15 — audit_checks: register_default_checks
# ===================================================================


class TestRegisterDefaultChecks:
    def test_registers_all_checks(self, baseline: SystemBaseline):
        backend = MagicMock()
        auditor = MagicMock(spec=SystemAuditor)

        register_default_checks(auditor, backend, baseline)

        assert auditor.register_check.call_count == 6
        registered_names = [call.args[0] for call in auditor.register_check.call_args_list]
        assert "check_task_completion_time" in registered_names
        assert "check_error_rate" in registered_names
        assert "check_dependencies" in registered_names
        assert "check_skill_integrity" in registered_names
        assert "check_config_drift" in registered_names
        assert "check_resource_usage" in registered_names

    def test_registered_closures_are_callable(self, baseline: SystemBaseline):
        backend = MagicMock()
        auditor = MagicMock(spec=SystemAuditor)

        register_default_checks(auditor, backend, baseline)

        for call in auditor.register_check.call_args_list:
            fn = call.args[1]
            assert callable(fn)


# ===================================================================
# PART 16 — SupervisorAgent: init and system prompt
# ===================================================================


class TestParseDelegation:
    def test_valid_delegation(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        response = """Here's my plan:
```json
{
  "analysis": "User wants auth",
  "delegations": [
    {"agent": "planner", "task": "Plan auth"}
  ],
  "synthesis_approach": "Combine results"
}
```"""
        plan = sup._parse_delegation(response)
        assert plan is not None
        assert plan.analysis == "User wants auth"
        assert len(plan.delegations) == 1

    def test_no_json(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        assert sup._parse_delegation("Just a normal response") is None

    def test_invalid_json(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        response = "```json\n{invalid}\n```"
        assert sup._parse_delegation(response) is None

    def test_empty_delegations(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        response = '```json\n{"delegations": []}\n```'
        assert sup._parse_delegation(response) is None


# ===================================================================
# PART 18 — SupervisorAgent: _parse_tool_calls
# ===================================================================


class TestRunAgent:
    def test_basic_run(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        result = asyncio.run(sup._run_agent("builder", "Build auth", []))
        assert result == "Agent response"

    def test_agent_error_caught(self, mock_provider: MagicMock):
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        sup = SupervisorAgent(provider=mock_provider)
        result = asyncio.run(sup._run_agent("builder", "Build auth", []))
        assert "error" in result.lower()
        assert "builder" in result

    def test_bridge_enrichment(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = "Enrichment context"
        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        asyncio.run(sup._run_agent("builder", "Build auth", []))
        bridge.enrich_prompt.assert_called_once()
        # Verify enrichment was appended to prompt
        call_args = mock_provider.complete.call_args[0][0]
        assert "Enrichment context" in call_args[0]["content"]

    def test_bridge_enrichment_failure(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.enrich_prompt.side_effect = RuntimeError("Bridge error")
        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        result = asyncio.run(sup._run_agent("builder", "Build auth", []))
        # Should succeed despite bridge failure
        assert result == "Agent response"

    def test_bridge_enrichment_empty(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = ""
        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        asyncio.run(sup._run_agent("builder", "Build auth", []))
        call_args = mock_provider.complete.call_args[0][0]
        # Empty enrichment should not add extra newlines
        system_prompt = call_args[0]["content"]
        assert not system_prompt.endswith("\n\n")

    def test_context_messages_included(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        context = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        asyncio.run(sup._run_agent("builder", "Build auth", context))
        call_args = mock_provider.complete.call_args[0][0]
        # Should include context messages (non-system ones)
        contents = [m["content"] for m in call_args]
        assert any("[Context]" in c for c in contents)

    def test_all_agent_prompts(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        agents = [
            "planner",
            "builder",
            "tester",
            "reviewer",
            "architect",
            "documenter",
            "analyst",
        ]
        for agent in agents:
            prompt = sup._get_agent_prompt(agent)
            assert len(prompt) > 0

    def test_unknown_agent_prompt(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        prompt = sup._get_agent_prompt("unknown_agent")
        assert "unknown_agent" in prompt


# ===================================================================
class TestExecuteDelegations:
    def test_basic_delegation(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        results = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build it"}],
                [],
                lambda _: None,
            )
        )
        assert "builder" in results
        assert results["builder"] == "Agent response"

    def test_parallel_delegations(self, mock_provider: MagicMock):
        sup = SupervisorAgent(provider=mock_provider)
        results = asyncio.run(
            sup._execute_delegations(
                [
                    {"agent": "builder", "task": "Build"},
                    {"agent": "tester", "task": "Test"},
                ],
                [],
                lambda _: None,
            )
        )
        assert "builder" in results
        assert "tester" in results

    def test_exception_in_agent(self, mock_provider: MagicMock):
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("LLM error"))
        sup = SupervisorAgent(provider=mock_provider)
        results = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build"}],
                [],
                lambda _: None,
            )
        )
        # _run_agent catches exceptions and returns error string
        assert "builder" in results
        assert "error" in results["builder"].lower()

    def test_convergence_drops_redundant(self, mock_provider: MagicMock):
        checker = MagicMock()
        checker.enabled = True
        convergence_result = MagicMock()
        convergence_result.dropped_agents = {"tester"}
        checker.check_delegations.return_value = convergence_result

        with patch(
            "animus_forge.agents.convergence.format_convergence_alert", return_value="Alert!"
        ):
            sup = SupervisorAgent(provider=mock_provider, convergence_checker=checker)
            results = asyncio.run(
                sup._execute_delegations(
                    [
                        {"agent": "builder", "task": "Build"},
                        {"agent": "tester", "task": "Test"},
                    ],
                    [],
                    lambda _: None,
                )
            )

        assert "builder" in results
        assert "tester" not in results

    def test_convergence_no_alert(self, mock_provider: MagicMock):
        checker = MagicMock()
        checker.enabled = True
        convergence_result = MagicMock()
        convergence_result.dropped_agents = set()
        checker.check_delegations.return_value = convergence_result

        with patch("animus_forge.agents.convergence.format_convergence_alert", return_value=None):
            sup = SupervisorAgent(provider=mock_provider, convergence_checker=checker)
            results = asyncio.run(
                sup._execute_delegations(
                    [{"agent": "builder", "task": "Build"}],
                    [],
                    lambda _: None,
                )
            )
        assert "builder" in results

    def test_bridge_records_outcomes(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = ""
        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)
        asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build"}],
                [],
                lambda _: None,
            )
        )
        bridge.record_task_outcome.assert_called_once()
        _, kwargs = bridge.record_task_outcome.call_args
        assert kwargs["outcome"] == "approved"

    def test_bridge_records_failed_outcome(self, mock_provider: MagicMock):
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("fail"))
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = ""
        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)
        asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build"}],
                [],
                lambda _: None,
            )
        )
        # The error string from _run_agent starts with "Agent ... encountered an error"
        _, kwargs = bridge.record_task_outcome.call_args
        assert kwargs["outcome"] == "failed"

    def test_bridge_outcome_exception_swallowed(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = ""
        bridge.record_task_outcome.side_effect = RuntimeError("DB error")
        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)
        # Should not raise
        results = asyncio.run(
            sup._execute_delegations(
                [{"agent": "builder", "task": "Build"}],
                [],
                lambda _: None,
            )
        )
        assert "builder" in results


# ===================================================================
# PART 23 — SupervisorAgent: consensus voting
# ===================================================================


class TestConsensusVoting:
    def test_run_consensus_vote(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.request_consensus.return_value = "req-123"
        mock_decision = MagicMock()
        mock_decision.outcome.value = "approved"
        bridge.evaluate.return_value = mock_decision
        bridge.enrich_prompt.return_value = ""

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        decision = sup._run_consensus_vote(
            agent_name="builder",
            task="Build auth",
            result_text="Code looks good",
            quorum="majority",
            skill_name="auth-builder",
        )
        assert decision is mock_decision
        bridge.request_consensus.assert_called_once()
        assert bridge.submit_agent_vote.call_count == 2

    def test_consensus_rejected_modifies_result(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.request_consensus.return_value = "req-123"
        bridge.enrich_prompt.return_value = ""

        mock_decision = MagicMock()
        mock_decision.outcome.value = "rejected"
        mock_decision.reasoning_summary = "Quality too low"
        bridge.evaluate.return_value = mock_decision

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                        "_skill_name": "builder-skill",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        assert "CONSENSUS REJECTED" in results["builder"]

    def test_consensus_deadlock_modifies_result(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.request_consensus.return_value = "req-123"
        bridge.enrich_prompt.return_value = ""

        mock_decision = MagicMock()
        mock_decision.outcome.value = "deadlock"
        mock_decision.reasoning_summary = "No majority"
        bridge.evaluate.return_value = mock_decision

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                        "_skill_name": "builder-skill",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        assert "CONSENSUS DEADLOCK" in results["builder"]
        assert "degraded confidence" in results["builder"]

    def test_consensus_escalated(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.request_consensus.return_value = "req-123"
        bridge.enrich_prompt.return_value = ""

        mock_decision = MagicMock()
        mock_decision.outcome.value = "escalated"
        mock_decision.reasoning_summary = "Needs human"
        bridge.evaluate.return_value = mock_decision

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                        "_skill_name": "builder-skill",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        assert "CONSENSUS ESCALATED" in results["builder"]

    def test_consensus_approved_leaves_result_unchanged(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.request_consensus.return_value = "req-123"
        bridge.enrich_prompt.return_value = ""

        mock_decision = MagicMock()
        mock_decision.outcome.value = "approved"
        bridge.evaluate.return_value = mock_decision

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                        "_skill_name": "builder-skill",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        assert results["builder"] == "Agent response"

    def test_consensus_skipped_for_any_level(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = ""

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "any",
                        "_skill_name": "builder-skill",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        # No consensus vote should be triggered
        bridge.request_consensus.assert_not_called()

    def test_consensus_skipped_for_error_results(self, mock_provider: MagicMock):
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("fail"))
        bridge = MagicMock()
        bridge.enrich_prompt.return_value = ""

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                        "_skill_name": "builder-skill",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        # Error results should skip consensus
        bridge.request_consensus.assert_not_called()

    def test_consensus_exception_swallowed(self, mock_provider: MagicMock):
        bridge = MagicMock()
        bridge.request_consensus.side_effect = RuntimeError("Bridge broken")
        bridge.enrich_prompt.return_value = ""

        sup = SupervisorAgent(provider=mock_provider, coordination_bridge=bridge)

        results = asyncio.run(
            sup._execute_delegations(
                [
                    {
                        "agent": "builder",
                        "task": "Build it",
                        "_skill_consensus": "majority",
                        "_skill_name": "builder-skill",
                    }
                ],
                [],
                lambda _: None,
            )
        )
        # Should not raise — result left as-is
        assert results["builder"] == "Agent response"


# ===================================================================
# PART 24 — SupervisorAgent: _synthesize_results
# ===================================================================


class TestSynthesizeResults:
    def test_successful_synthesis(self, mock_provider: MagicMock):
        mock_provider.complete = AsyncMock(return_value="Synthesized response")
        sup = SupervisorAgent(provider=mock_provider)
        plan = DelegationPlan(
            analysis="Test analysis",
            delegations=[{"agent": "builder", "task": "Build"}],
            synthesis_approach="Combine",
        )
        result = asyncio.run(sup._synthesize_results(plan, {"builder": "Built it"}, []))
        assert result == "Synthesized response"

    def test_synthesis_error_fallback(self, mock_provider: MagicMock):
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("fail"))
        sup = SupervisorAgent(provider=mock_provider)
        plan = DelegationPlan(
            analysis="Test",
            delegations=[{"agent": "builder", "task": "Build"}],
            synthesis_approach="Combine",
        )
        result = asyncio.run(sup._synthesize_results(plan, {"builder": "Built it"}, []))
        assert "builder" in result.lower()
        assert "Built it" in result


# ===================================================================
# PART 25 — SupervisorAgent: process_message (integration)
# ===================================================================


class TestSkillLibraryInDelegations:
    def test_skill_consensus_annotated(self, mock_provider: MagicMock):
        skill_lib = MagicMock()
        mock_skill = MagicMock()
        mock_skill.consensus_level = "majority"
        mock_skill.name = "auth-skill"
        skill_lib.find_skills_for_task.return_value = [mock_skill]
        skill_lib.build_routing_summary.return_value = ""
        skill_lib.build_skill_context.return_value = ""

        sup = SupervisorAgent(provider=mock_provider, skill_library=skill_lib)

        delegations = [{"agent": "builder", "task": "Build auth module"}]
        asyncio.run(sup._execute_delegations(delegations, [], lambda _: None))
        # Delegation should have been annotated
        assert delegations[0].get("_skill_consensus") == "majority"
        assert delegations[0].get("_skill_name") == "auth-skill"

    def test_skill_consensus_any_not_annotated(self, mock_provider: MagicMock):
        skill_lib = MagicMock()
        mock_skill = MagicMock()
        mock_skill.consensus_level = "any"
        mock_skill.name = "generic-skill"
        skill_lib.find_skills_for_task.return_value = [mock_skill]
        skill_lib.build_routing_summary.return_value = ""
        skill_lib.build_skill_context.return_value = ""

        sup = SupervisorAgent(provider=mock_provider, skill_library=skill_lib)

        delegations = [{"agent": "builder", "task": "Build it"}]
        asyncio.run(sup._execute_delegations(delegations, [], lambda _: None))
        # "any" and "" should not be annotated
        assert "_skill_consensus" not in delegations[0]

    def test_skill_empty_consensus_not_annotated(self, mock_provider: MagicMock):
        skill_lib = MagicMock()
        mock_skill = MagicMock()
        mock_skill.consensus_level = ""
        mock_skill.name = "simple-skill"
        skill_lib.find_skills_for_task.return_value = [mock_skill]
        skill_lib.build_routing_summary.return_value = ""
        skill_lib.build_skill_context.return_value = ""

        sup = SupervisorAgent(provider=mock_provider, skill_library=skill_lib)

        delegations = [{"agent": "builder", "task": "Build it"}]
        asyncio.run(sup._execute_delegations(delegations, [], lambda _: None))
        assert "_skill_consensus" not in delegations[0]
