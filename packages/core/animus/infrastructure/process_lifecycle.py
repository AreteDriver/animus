"""Process lifecycle utilities: strong singletons, registry, and visibility.

Provides:
  - LockedPidFile: advisory-but-locked PID file using fcntl (POSIX).
  - SystemProcessRegistry: SQLite-backed registry of all Animus OS processes.
  - ProcessGuard: mixin that wires both into any long-lived component.
  - CLI helpers for ``animus status`` and ``animus cleanup``.

Windows fallback: fcntl is unavailable; falls back to soft PID file with a warning.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sqlite3
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("animus.infrastructure.process_lifecycle")

# ---------------------------------------------------------------------------
# Platform: fcntl availability
# ---------------------------------------------------------------------------
try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlreadyRunningError(RuntimeError):
    """Raised when a singleton component is already alive."""

    def __init__(self, component: str, pid: int | None = None) -> None:
        self.component = component
        self.pid = pid
        msg = f"{component} is already running"
        if pid is not None:
            msg += f" (pid {pid})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# LockedPidFile
# ---------------------------------------------------------------------------


class LockedPidFile:
    """Strong-singleton PID file using POSIX file locking.

    Usage::

        with LockedPidFile("~/.animus/daemon.pid", "daemon") as pid_file:
            # We hold the exclusive lock. No other process can enter.
            ...

    If another process holds the lock, raises :class:`AlreadyRunningError`.
    If a stale PID file exists (owner dead), it is removed and re-created.
    """

    def __init__(
        self,
        path: str | Path,
        component: str,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.path = Path(path).expanduser()
        self.component = component
        self.timeout_seconds = timeout_seconds
        self._fd: int | None = None
        self._acquired = False

    def _is_process_alive(self, pid: int) -> bool:
        """Check PID liveness and executable identity."""
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False

        # Defensive: verify /proc/pid/exe to avoid PID-reuse false positives.
        # For venv-based execution the executable path contains the project name.
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
            if "animus" not in exe.lower() and "python" not in exe.lower():
                logger.warning(
                    "PID %d exe %s does not look like animus/python; treating as stale",
                    pid,
                    exe,
                )
                return False
        except (OSError, FileNotFoundError):
            pass
        return True

    def _try_lock(self) -> bool:
        """Attempt to acquire exclusive lock without blocking forever."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open RDWR so we can read stale PID and then write ours
        self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)

        if _HAS_FCNTL:
            try:
                fcntl.lockf(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                # Lock held by another process
                return False
        else:
            # No fcntl — best-effort: read PID, check alive, warn
            logger.warning(
                "fcntl unavailable on this platform; using soft PID file for %s",
                self.component,
            )
            existing = self._read_pid_from_fd()
            if existing is not None and self._is_process_alive(existing):
                return False

        return True

    def _read_pid_from_fd(self) -> int | None:
        """Read PID from the open fd."""
        if self._fd is None:
            return None
        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            data = os.read(self._fd, 64).decode().strip()
            if data:
                return int(data)
        except (ValueError, OSError):
            pass
        return None

    @classmethod
    def peek(cls, path: str | Path, component: str) -> tuple[bool, int | None]:
        """Check if a PID file is held by a live process without acquiring the lock.

        Returns:
            (is_running, owner_pid)
        """
        path = Path(path).expanduser()
        if not path.exists():
            return False, None
        try:
            pid = int(path.read_text().strip())
        except (ValueError, OSError):
            return False, None

        # Quick liveness check
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False, pid

        # Executable identity check (same logic as _is_process_alive)
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
            if "animus" not in exe.lower() and "python" not in exe.lower():
                return False, pid
        except (OSError, FileNotFoundError):
            pass

        return True, pid

    def _write_pid(self) -> None:
        """Write current PID to fd."""
        if self._fd is None:
            return
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, str(os.getpid()).encode())
        os.fsync(self._fd)

    def acquire(self) -> LockedPidFile:
        """Acquire the lock. Raises AlreadyRunningError if another instance lives."""
        if self._acquired:
            return self

        if not self._try_lock():
            # Lock held — read the owner PID from fd (which we opened)
            existing_pid = self._read_pid_from_fd()
            # If fd is None (shouldn't happen), try from path
            if existing_pid is None and self.path.exists():
                try:
                    existing_pid = int(self.path.read_text().strip())
                except (ValueError, OSError):
                    pass

            # Cleanup stale PID file before failing
            if existing_pid is not None and not self._is_process_alive(existing_pid):
                logger.info(
                    "Removing stale PID file for %s (pid %d dead)",
                    self.component,
                    existing_pid,
                )
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass
                # Retry once
                if self._try_lock():
                    self._write_pid()
                    self._acquired = True
                    atexit.register(self.release)
                    return self
            else:
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None

            raise AlreadyRunningError(self.component, pid=existing_pid)

        self._write_pid()
        self._acquired = True
        atexit.register(self.release)
        return self

    def release(self) -> None:
        """Release lock and remove PID file."""
        if not self._acquired:
            return
        self._acquired = False
        try:
            if self._fd is not None:
                if _HAS_FCNTL:
                    try:
                        fcntl.lockf(self._fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(self._fd)
                self._fd = None
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> LockedPidFile:
        return self.acquire()

    def __exit__(self, *args: Any) -> None:
        self.release()

    def __del__(self) -> None:
        # Best-effort cleanup if object GC'd without explicit exit
        try:
            self.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registry data model
# ---------------------------------------------------------------------------


class ProcessState(Enum):
    """Lifecycle states for a registered OS process."""

    RUNNING = "running"
    SUSPECT = "suspect"  # heartbeat missing
    ORPHAN = "orphan"  # reparented to init (ppid == 1 or parent dead)
    STOPPED = "stopped"  # known dead, pending sweep


@dataclass
class RegisteredProcess:
    """Row in the system process registry."""

    process_id: str
    component: str  # e.g. "daemon", "mcp_server", "tray"
    pid: int
    ppid: int
    command_line: str
    start_time: datetime
    last_heartbeat: datetime
    state: ProcessState
    pid_file: str | None
    port: int | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["start_time"] = self.start_time.isoformat()
        d["last_heartbeat"] = self.last_heartbeat.isoformat()
        return d


# ---------------------------------------------------------------------------
# SystemProcessRegistry
# ---------------------------------------------------------------------------


class SystemProcessRegistry:
    """SQLite-backed registry of all Animus OS-level processes.

    Provides discovery, health checking, and orphan detection.
    Thread-safe via SQLite's built-in locking.
    """

    _TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS processes (
        process_id TEXT PRIMARY KEY,
        component TEXT NOT NULL,
        pid INTEGER NOT NULL,
        ppid INTEGER NOT NULL,
        command_line TEXT NOT NULL,
        start_time TEXT NOT NULL,
        last_heartbeat TEXT NOT NULL,
        state TEXT NOT NULL DEFAULT 'running',
        pid_file TEXT,
        port INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_component ON processes(component);
    CREATE INDEX IF NOT EXISTS idx_pid ON processes(pid);
    CREATE INDEX IF NOT EXISTS idx_state ON processes(state);
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".animus" / "process_registry.db"
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.executescript(self._TABLE_SQL)

    def register(
        self,
        *,
        component: str,
        pid: int,
        command_line: str,
        pid_file: str | Path | None = None,
        port: int | None = None,
        process_id: str | None = None,
    ) -> RegisteredProcess:
        """Register a new running process."""
        import uuid

        now = datetime.now(timezone.utc)
        proc = RegisteredProcess(
            process_id=process_id or str(uuid.uuid4()),
            component=component,
            pid=pid,
            ppid=os.getppid(),
            command_line=command_line,
            start_time=now,
            last_heartbeat=now,
            state=ProcessState.RUNNING,
            pid_file=str(pid_file) if pid_file else None,
            port=port,
        )
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO processes
                (process_id, component, pid, ppid, command_line, start_time,
                 last_heartbeat, state, pid_file, port)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(process_id) DO UPDATE SET
                  pid=excluded.pid,
                  ppid=excluded.ppid,
                  command_line=excluded.command_line,
                  start_time=excluded.start_time,
                  last_heartbeat=excluded.last_heartbeat,
                  state=excluded.state,
                  pid_file=excluded.pid_file,
                  port=excluded.port
                """,
                (
                    proc.process_id,
                    proc.component,
                    proc.pid,
                    proc.ppid,
                    proc.command_line,
                    proc.start_time.isoformat(),
                    proc.last_heartbeat.isoformat(),
                    proc.state.value,
                    proc.pid_file,
                    proc.port,
                ),
            )
            conn.execute("COMMIT")
        logger.info("Registered %s (pid %d) as %s", component, pid, proc.process_id)
        return proc

    def heartbeat(self, process_id: str) -> bool:
        """Update last_heartbeat for a process."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            cur = conn.execute(
                "UPDATE processes SET last_heartbeat = ?, state = ? WHERE process_id = ?",
                (now, ProcessState.RUNNING.value, process_id),
            )
            return cur.rowcount > 0

    def unregister(self, process_id: str) -> bool:
        """Remove a process from the registry."""
        with self._connection() as conn:
            cur = conn.execute("DELETE FROM processes WHERE process_id = ?", (process_id,))
            return cur.rowcount > 0

    def list_active(
        self,
        component: str | None = None,
        state: ProcessState | None = None,
    ) -> list[RegisteredProcess]:
        """List registered processes, optionally filtered."""
        sql = "SELECT * FROM processes WHERE 1=1"
        params: list[Any] = []
        if component is not None:
            sql += " AND component = ?"
            params.append(component)
        if state is not None:
            sql += " AND state = ?"
            params.append(state.value)
        sql += " ORDER BY start_time DESC"

        rows: list[RegisteredProcess] = []
        with self._connection() as conn:
            for row in conn.execute(sql, params):
                rows.append(self._row_to_process(row))
        return rows

    def get_by_pid(self, pid: int) -> RegisteredProcess | None:
        """Look up a process by PID."""
        with self._connection() as conn:
            cur = conn.execute("SELECT * FROM processes WHERE pid = ?", (pid,))
            row = cur.fetchone()
            if row:
                return self._row_to_process(row)
        return None

    def _row_to_process(self, row: sqlite3.Row) -> RegisteredProcess:
        def _dt(val: str | None) -> datetime:
            if val is None:
                return datetime.min.replace(tzinfo=timezone.utc)
            # Handle both ISO with +00:00 and Z
            val = val.replace("Z", "+00:00")
            return datetime.fromisoformat(val)

        return RegisteredProcess(
            process_id=row[0],
            component=row[1],
            pid=row[2],
            ppid=row[3],
            command_line=row[4],
            start_time=_dt(row[5]),
            last_heartbeat=_dt(row[6]),
            state=ProcessState(row[7]),
            pid_file=row[8],
            port=row[9],
        )

    def _is_process_alive(self, pid: int, command_line: str) -> bool:
        """Check whether a registered process is still alive and its exe still matches."""
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False

        # Verify executable path to avoid PID-reuse false positives
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
            if "animus" not in exe.lower() and "python" not in exe.lower():
                return False
        except (OSError, FileNotFoundError):
            pass
        return True

    def _is_parent_alive(self, ppid: int) -> bool:
        """Check whether the parent process is still alive."""
        if ppid <= 1:
            return True  # init/systemd never dies
        try:
            os.kill(ppid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def sweep(self, suspect_threshold_seconds: float = 120.0) -> SweepResult:
        """Scan registry, mark suspects/orphans, remove dead entries, return summary."""
        now = datetime.now(timezone.utc)
        removed: list[RegisteredProcess] = []
        marked_orphan: list[RegisteredProcess] = []
        marked_suspect: list[RegisteredProcess] = []

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for row in conn.execute("SELECT * FROM processes"):
                proc = self._row_to_process(row)
                alive = self._is_process_alive(proc.pid, proc.command_line)
                if not alive:
                    conn.execute("DELETE FROM processes WHERE process_id = ?", (proc.process_id,))
                    removed.append(proc)
                    continue

                # Orphan check
                if not self._is_parent_alive(proc.ppid):
                    if proc.state != ProcessState.ORPHAN:
                        conn.execute(
                            "UPDATE processes SET state = ? WHERE process_id = ?",
                            (ProcessState.ORPHAN.value, proc.process_id),
                        )
                        marked_orphan.append(proc)
                    continue

                # Suspect check (stale heartbeat)
                stale = now - proc.last_heartbeat > timedelta(seconds=suspect_threshold_seconds)
                if stale and proc.state == ProcessState.RUNNING:
                    conn.execute(
                        "UPDATE processes SET state = ? WHERE process_id = ?",
                        (ProcessState.SUSPECT.value, proc.process_id),
                    )
                    marked_suspect.append(proc)

            conn.execute("COMMIT")

        logger.info(
            "Sweep complete: removed=%d orphan=%d suspect=%d",
            len(removed),
            len(marked_orphan),
            len(marked_suspect),
        )
        return SweepResult(
            removed=removed, marked_orphan=marked_orphan, marked_suspect=marked_suspect
        )

    def summary(self) -> dict[str, Any]:
        """Return aggregate counts by component and state."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT component, state, COUNT(*) FROM processes
                GROUP BY component, state
                """
            ).fetchall()
        by_component: dict[str, dict[str, int]] = {}
        for component, state, count in rows:
            by_component.setdefault(component, {})[state] = count
        total = sum(sum(v.values()) for v in by_component.values())
        return {"total": total, "by_component": by_component}


@dataclass
class SweepResult:
    """Result of a registry sweep."""

    removed: list[RegisteredProcess]
    marked_orphan: list[RegisteredProcess]
    marked_suspect: list[RegisteredProcess]


# ---------------------------------------------------------------------------
# ProcessGuard
# ---------------------------------------------------------------------------


class ProcessGuard:
    """Mixin for long-lived Animus components.

    Enforces singleton via LockedPidFile, registers in SystemProcessRegistry,
    and runs a heartbeat thread.

    Usage::

        class MyDaemon(ProcessGuard):
            def __init__(self):
                super().__init__(
                    component="daemon",
                    pid_file="~/.animus/daemon.pid",
                )

            def run(self):
                with self.guard():
                    # singleton enforced, registered, heartbeat running
                    ...
    """

    def __init__(
        self,
        *,
        component: str,
        pid_file: str | Path,
        registry: SystemProcessRegistry | None = None,
        heartbeat_interval: float = 30.0,
        port: int | None = None,
    ) -> None:
        self._component = component
        self._pid_file = Path(pid_file).expanduser()
        self._registry = registry or SystemProcessRegistry()
        self._heartbeat_interval = heartbeat_interval
        self._port = port
        self._process_id: str | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()
        self._pid_lock: LockedPidFile | None = None

    def guard(self) -> ProcessGuardContext:
        """Return a context manager that enforces singleton + registry + heartbeat."""
        return ProcessGuardContext(self)

    def _start_heartbeat(self) -> None:
        """Begin background heartbeat thread."""
        self._stop_heartbeat.clear()

        def _beat() -> None:
            while not self._stop_heartbeat.wait(self._heartbeat_interval):
                if self._process_id:
                    ok = self._registry.heartbeat(self._process_id)
                    if not ok:
                        logger.warning("Heartbeat failed for %s; re-registering", self._component)
                        # Re-register if row was lost (e.g., manual DB wipe)
                        self._do_register()

        self._heartbeat_thread = threading.Thread(
            target=_beat, daemon=True, name=f"{self._component}-heartbeat"
        )
        self._heartbeat_thread.start()

    def _do_register(self) -> None:
        """Register this process in the registry."""
        cmd = " ".join(sys.argv)
        proc = self._registry.register(
            component=self._component,
            pid=os.getpid(),
            command_line=cmd,
            pid_file=self._pid_file,
            port=self._port,
            process_id=self._process_id,
        )
        self._process_id = proc.process_id

    def _enter(self) -> None:
        """Acquire singleton, register, start heartbeat."""
        self._pid_lock = LockedPidFile(self._pid_file, self._component)
        self._pid_lock.acquire()
        self._do_register()
        self._start_heartbeat()
        logger.info("%s guard active (pid %d)", self._component, os.getpid())

    def _exit(self) -> None:
        """Stop heartbeat, unregister, release singleton."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
        if self._process_id:
            self._registry.unregister(self._process_id)
        if self._pid_lock is not None:
            self._pid_lock.release()
            self._pid_lock = None
        logger.info("%s guard released", self._component)


class ProcessGuardContext:
    """Context manager returned by ProcessGuard.guard()."""

    def __init__(self, guard: ProcessGuard) -> None:
        self._guard = guard

    def __enter__(self) -> ProcessGuard:
        self._guard._enter()
        return self._guard

    def __exit__(self, *args: Any) -> None:
        self._guard._exit()


# ---------------------------------------------------------------------------
# CLI helpers (used by animus CLI, not standalone)
# ---------------------------------------------------------------------------


def print_status_table(registry: SystemProcessRegistry | None = None, json: bool = False) -> None:
    """Print human-readable or JSON status of all Animus processes."""
    registry = registry or SystemProcessRegistry()
    registry.sweep(suspect_threshold_seconds=120.0)
    procs = registry.list_active()

    # Gather extra runtime info
    rows: list[dict[str, Any]] = []
    for p in procs:
        uptime = "unknown"
        if p.state != ProcessState.STOPPED:
            uptime_seconds = (datetime.now(timezone.utc) - p.start_time).total_seconds()
            uptime = _human_duration(uptime_seconds)
        rows.append(
            {
                "component": p.component,
                "pid": p.pid,
                "state": p.state.value,
                "uptime": uptime,
                "port": p.port,
                "command": p.command_line[:60],
            }
        )

    if json:
        import json as _json

        print(_json.dumps(rows, indent=2))
        return

    if not rows:
        print("No Animus processes registered.")
        return

    # Simple aligned table
    print(f"{'Component':<14} {'PID':>7} {'State':<10} {'Uptime':<12} {'Port':>6} {'Command'}")
    print("-" * 90)
    for r in rows:
        port_str = str(r["port"]) if r["port"] else "-"
        print(
            f"{r['component']:<14} {r['pid']:>7} {r['state']:<10} "
            f"{r['uptime']:<12} {port_str:>6} {r['command']}"
        )


def run_cleanup(
    registry: SystemProcessRegistry | None = None,
    dry_run: bool = True,
    kill_orphans: bool = False,
) -> list[RegisteredProcess]:
    """Sweep registry, optionally kill orphans, return affected processes."""
    registry = registry or SystemProcessRegistry()

    if dry_run:
        # Simulate sweep without mutating registry
        all_procs = registry.list_active()
        affected: list[RegisteredProcess] = []
        for p in all_procs:
            alive = registry._is_process_alive(p.pid, p.command_line)
            if not alive:
                affected.append(p)
                continue
            if not registry._is_parent_alive(p.ppid):
                affected.append(p)
        return affected

    result = registry.sweep(suspect_threshold_seconds=120.0)
    affected = list(result.removed)

    if kill_orphans:
        for p in result.marked_orphan:
            try:
                os.kill(p.pid, signal.SIGTERM)
                affected.append(p)
            except (ProcessLookupError, PermissionError):
                pass

    # Also clean up any PID files for removed processes
    for p in result.removed:
        if p.pid_file:
            try:
                Path(p.pid_file).unlink(missing_ok=True)
            except OSError:
                pass

    return affected


def _human_duration(seconds: float) -> str:
    """Convert seconds to human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"
