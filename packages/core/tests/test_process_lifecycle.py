"""Tests for process lifecycle utilities: LockedPidFile, SystemProcessRegistry, ProcessGuard."""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from animus.infrastructure import (
    AlreadyRunningError,
    LockedPidFile,
    ProcessGuard,
    ProcessState,
    RegisteredProcess,
    SystemProcessRegistry,
)
from animus.infrastructure.process_lifecycle import (
    print_status_table,
    run_cleanup,
    _human_duration,
)


# ---------------------------------------------------------------------------
# LockedPidFile
# ---------------------------------------------------------------------------


class TestLockedPidFile:
    def test_acquire_writes_pid(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "test.pid"
        lock = LockedPidFile(pid_file, "test")
        lock.acquire()
        assert pid_file.exists()
        assert int(pid_file.read_text().strip()) == os.getpid()
        lock.release()

    def test_release_removes_file(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "test.pid"
        lock = LockedPidFile(pid_file, "test")
        lock.acquire()
        lock.release()
        assert not pid_file.exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "test.pid"
        with LockedPidFile(pid_file, "test"):
            assert pid_file.exists()
        assert not pid_file.exists()

    def test_singleton_second_acquire_fails(self, tmp_path: Path) -> None:
        """A separate process holding the lock must block a new acquire."""
        pid_file = tmp_path / "test.pid"
        # Spawn a child process that acquires and holds the lock
        script = f"""
import sys
sys.path.insert(0, '/home/arete/projects/animus/packages/core')
from animus.infrastructure import LockedPidFile
lock = LockedPidFile('{pid_file}', 'test')
lock.acquire()
print('LOCK_HELD')
sys.stdout.flush()
import time
time.sleep(10)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for child to report lock held
        try:
            line = proc.stdout.readline()
            assert "LOCK_HELD" in line
            lock2 = LockedPidFile(pid_file, "test")
            with pytest.raises(AlreadyRunningError):
                lock2.acquire()
        finally:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)

    def test_stale_pid_recovery(self, tmp_path: Path) -> None:
        """A PID file pointing to a dead process should be reclaimed."""
        pid_file = tmp_path / "test.pid"
        # Write a fake PID that is definitely not alive
        fake_pid = 999999
        pid_file.write_text(str(fake_pid))

        lock = LockedPidFile(pid_file, "test")
        lock.acquire()
        assert int(pid_file.read_text().strip()) == os.getpid()
        lock.release()

    def test_pid_reuse_safety(self, tmp_path: Path) -> None:
        """If another live process owns the PID but it's not animus, treat as stale."""
        pid_file = tmp_path / "test.pid"
        # Spawn a non-animus sleep process
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            pid_file.write_text(str(proc.pid))
            lock = LockedPidFile(pid_file, "test")
            lock.acquire()
            assert int(pid_file.read_text().strip()) == os.getpid()
            lock.release()
        finally:
            proc.kill()
            proc.wait()

    def test_peek_returns_running_state(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "test.pid"
        assert LockedPidFile.peek(pid_file, "test") == (False, None)

        # Spawn a child whose cmdline contains "animus" so peek recognises it
        script = "import time; time.sleep(10)  # animus-test"
        proc = subprocess.Popen([sys.executable, "-c", script])
        try:
            pid_file.write_text(str(proc.pid))
            running, pid = LockedPidFile.peek(pid_file, "test")
            assert running is True
            assert pid == proc.pid
        finally:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# SystemProcessRegistry
# ---------------------------------------------------------------------------


class TestSystemProcessRegistry:
    def test_register_and_list(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        reg = SystemProcessRegistry(db)
        proc = reg.register(component="daemon", pid=os.getpid(), command_line="python -m animus.daemon")
        assert proc.component == "daemon"
        assert proc.pid == os.getpid()
        active = reg.list_active()
        assert len(active) == 1
        assert active[0].process_id == proc.process_id

    def test_heartbeat_updates(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        reg = SystemProcessRegistry(db)
        proc = reg.register(component="mcp_server", pid=os.getpid(), command_line="python -m animus.mcp_server")
        assert reg.heartbeat(proc.process_id) is True
        # Unregister
        assert reg.unregister(proc.process_id) is True
        assert reg.list_active() == []

    def test_sweep_removes_dead(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        reg = SystemProcessRegistry(db)
        fake_pid = 999999
        proc = reg.register(component="daemon", pid=fake_pid, command_line="python -m animus.daemon")
        result = reg.sweep()
        assert len(result.removed) == 1
        assert result.removed[0].process_id == proc.process_id
        assert reg.list_active() == []

    def test_sweep_marks_orphan(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        reg = SystemProcessRegistry(db)
        # Register ourselves with a fake dead parent
        fake_ppid = 999998
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO processes (process_id, component, pid, ppid, command_line, start_time, last_heartbeat, state)"
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("test-1", "tray", os.getpid(), fake_ppid, "animus-tray", "2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00", "running"),
            )
        result = reg.sweep()
        assert len(result.marked_orphan) == 1
        # Re-query to verify DB was updated
        active = reg.list_active()
        assert active[0].state == ProcessState.ORPHAN

    def test_sweep_marks_suspect(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        reg = SystemProcessRegistry(db)
        old_time = "2020-01-01T00:00:00+00:00"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO processes (process_id, component, pid, ppid, command_line, start_time, last_heartbeat, state)"
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("test-1", "daemon", os.getpid(), os.getppid(), "animus-daemon", old_time, old_time, "running"),
            )
        result = reg.sweep(suspect_threshold_seconds=1.0)
        assert len(result.marked_suspect) == 1
        # Re-query to verify DB was updated
        active = reg.list_active()
        assert active[0].state == ProcessState.SUSPECT

    def test_summary_counts(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        reg = SystemProcessRegistry(db)
        reg.register(component="daemon", pid=1, command_line="c1")
        reg.register(component="mcp_server", pid=2, command_line="c2")
        summary = reg.summary()
        assert summary["total"] == 2
        assert summary["by_component"]["daemon"]["running"] == 1


# ---------------------------------------------------------------------------
# ProcessGuard
# ---------------------------------------------------------------------------


class TestProcessGuard:
    def test_guard_enforces_singleton(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "guard.pid"

        class MyDaemon(ProcessGuard):
            def __init__(self) -> None:
                super().__init__(component="my_daemon", pid_file=pid_file)

        # Spawn a child process that holds the guard
        script = f"""
import sys, time
sys.path.insert(0, '/home/arete/projects/animus/packages/core')
from animus.infrastructure import ProcessGuard
class D(ProcessGuard):
    def __init__(self): super().__init__(component='my_daemon', pid_file='{pid_file}')
d = D()
d.guard().__enter__()
print('GUARD_HELD')
sys.stdout.flush()
time.sleep(10)
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            line = proc.stdout.readline()
            assert "GUARD_HELD" in line

            d2 = MyDaemon()
            with pytest.raises(AlreadyRunningError):
                d2.guard().__enter__()
        finally:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=5)

    def test_guard_registers_and_heartbeats(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        pid_file = tmp_path / "guard.pid"
        reg = SystemProcessRegistry(db)

        class MyDaemon(ProcessGuard):
            def __init__(self) -> None:
                super().__init__(component="my_daemon", pid_file=pid_file, registry=reg, heartbeat_interval=0.5)

        d = MyDaemon()
        with d.guard():
            time.sleep(1.0)
            procs = reg.list_active(component="my_daemon")
            assert len(procs) == 1
            assert procs[0].state == ProcessState.RUNNING

        # After exit, should be unregistered
        assert reg.list_active(component="my_daemon") == []

    def test_guard_releases_on_exit(self, tmp_path: Path) -> None:
        pid_file = tmp_path / "guard.pid"

        class MyDaemon(ProcessGuard):
            def __init__(self) -> None:
                super().__init__(component="my_daemon", pid_file=pid_file)

        d = MyDaemon()
        with d.guard():
            pass
        assert not pid_file.exists()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


class TestCliHelpers:
    def test_human_duration(self) -> None:
        assert _human_duration(45) == "45s"
        assert _human_duration(120) == "2m"
        assert _human_duration(3600) == "1.0h"
        assert _human_duration(90000) == "1.0d"

    def test_print_status_table_empty(self, capsys: pytest.CaptureFixture) -> None:
        db = Path(tempfile.mkdtemp()) / "registry.db"
        reg = SystemProcessRegistry(db)
        print_status_table(reg, json=False)
        captured = capsys.readouterr()
        assert "No Animus processes registered" in captured.out

    def test_run_cleanup_dry_run(self, tmp_path: Path) -> None:
        db = tmp_path / "registry.db"
        reg = SystemProcessRegistry(db)
        reg.register(component="daemon", pid=999999, command_line="c")
        affected = run_cleanup(reg, dry_run=True, kill_orphans=False)
        # dry_run returns what *would* be removed (dead PID) without mutating registry
        assert len(affected) == 1
        assert affected[0].pid == 999999
