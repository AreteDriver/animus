"""Machine-readable systemd state reader.

Implements the ``SystemdStateReader`` over ``systemctl --user show``,
which is the load-bearing interface mandated by ADR-007. ``systemctl
status`` is human-oriented and is explicitly *not* used.

The reader is *pure*: it accepts a :class:`SystemdInvoker` protocol.
Production invokes ``systemctl --user show``; the test harness
provides a recorded set of responses.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger("animus_bootstrap.lifecycle.systemd")


class SystemdStateError(RuntimeError):
    """Raised when systemd state cannot be determined."""


@dataclass
class UnitState:
    """A subset of the systemd ``show`` output for one unit."""

    name: str
    active_state: str | None
    sub_state: str | None
    load_state: str | None
    main_pid: int | None
    result: str | None
    exec_main_start_timestamp: str | None
    memory_current: int | None
    cpu_usage_nsec: int | None
    tasks_current: int | None

    @property
    def is_active(self) -> bool:
        return self.active_state == "active"

    @property
    def is_failed(self) -> bool:
        return self.active_state == "failed"

    @property
    def is_activating(self) -> bool:
        return self.active_state == "activating"

    @property
    def is_deactivating(self) -> bool:
        return self.active_state == "deactivating"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "active_state": self.active_state,
            "sub_state": self.sub_state,
            "load_state": self.load_state,
            "main_pid": self.main_pid,
            "result": self.result,
            "exec_main_start_timestamp": self.exec_main_start_timestamp,
            "memory_current": self.memory_current,
            "cpu_usage_nsec": self.cpu_usage_nsec,
            "tasks_current": self.tasks_current,
        }


def parse_show_output(name: str, output: str) -> UnitState:
    """Parse the ``KEY=VALUE`` output of ``systemctl --user show <name>``.

    The output is stable since systemd v219. Lines are stripped of
    whitespace; ``=`` in values is preserved. Whitespace within values
    is preserved (``MemoryCurrent=`` may include spaces from the
    ``systemctl`` formatting).
    """
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        parsed[key.strip()] = value.strip()

    def _int(key: str) -> int | None:
        v = parsed.get(key)
        if v is None or v == "":
            return None
        try:
            return int(v)
        except ValueError:
            return None

    return UnitState(
        name=name,
        active_state=parsed.get("ActiveState"),
        sub_state=parsed.get("SubState"),
        load_state=parsed.get("LoadState"),
        main_pid=_int("MainPID"),
        result=parsed.get("Result"),
        exec_main_start_timestamp=parsed.get("ExecMainStartTimestamp"),
        memory_current=_int("MemoryCurrent"),
        cpu_usage_nsec=_int("CPUUsageNSec"),
        tasks_current=_int("TasksCurrent"),
    )


class SystemdInvoker(Protocol):
    """The backend the :class:`SystemdStateReader` uses to talk to systemd.

    Production invokes ``systemctl --user show``. The test harness
    provides a fake invoker that returns canned responses.
    """

    def show(self, unit: str) -> str:
        """Return the raw output of ``systemctl --user show <unit>``."""
        ...

    def list_drop_ins(self, unit: str) -> list[str]:
        """Return the list of drop-in filenames under ``<unit>.d/``."""
        ...


# Properties of interest. The full list is in `man systemctl`; this
# subset is what ``UnitState`` consumes.
SHOW_PROPERTIES = (
    "ActiveState",
    "SubState",
    "LoadState",
    "MainPID",
    "Result",
    "ExecMainStartTimestamp",
    "MemoryCurrent",
    "CPUUsageNSec",
    "TasksCurrent",
)


@dataclass
class SystemdStateReader:
    """Reads systemd state via ``systemctl --user show``."""

    invoker: SystemdInvoker

    def read(self, unit: str) -> UnitState:
        """Read a single unit's state."""
        try:
            output = self.invoker.show(unit)
        except Exception as exc:
            logger.warning("systemd show %s failed: %s", unit, exc)
            raise SystemdStateError(str(exc)) from exc
        return parse_show_output(unit, output)

    def read_many(self, units: Iterable[str]) -> dict[str, UnitState]:
        """Read multiple units. Failed reads return ``None``."""
        out: dict[str, UnitState] = {}
        for unit in units:
            try:
                out[unit] = self.read(unit)
            except SystemdStateError:
                out[unit] = UnitState(
                    name=unit,
                    active_state=None,
                    sub_state=None,
                    load_state=None,
                    main_pid=None,
                    result=None,
                    exec_main_start_timestamp=None,
                    memory_current=None,
                    cpu_usage_nsec=None,
                    tasks_current=None,
                )
        return out

    def target_is_active(self, target: str) -> bool | None:
        """Read a target's active state. Returns ``None`` on failure."""
        try:
            state = self.read(target)
        except SystemdStateError:
            return None
        return state.is_active
