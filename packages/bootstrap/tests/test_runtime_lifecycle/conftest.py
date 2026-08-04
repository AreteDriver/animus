"""Pytest fixtures for the runtime lifecycle test harness.

The harness enforces isolation from the live Animus runtime. Every
fixture in this file:

- Uses a temporary ``XDG_CONFIG_HOME`` and ``XDG_RUNTIME_DIR`` under
  ``tmp_path``.
- Uses a unique test prefix (``animus-test-<uuid8>-``) on unit names.
- Uses a temporary registry database under ``tmp_path``.
- Uses a temporary ``profile.json`` under ``tmp_path``.
- Allocates a free port from the OS.
- Cleans up in both success and failure paths.

No fixture in this file resolves a unit name against the live systemd
user manager. Tests that need ``systemctl show`` semantics receive a
:class:`FakeSystemd` that records calls and returns canned responses.

ADR-007 §Test matrix + Build spec §16.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import pytest


# ---------------------------------------------------------------------------
# Test prefix helpers
# ---------------------------------------------------------------------------


def make_test_prefix() -> str:
    """Return a unique test prefix like ``animus-test-3f2a91b7-``."""
    return f"animus-test-{uuid.uuid4().hex[:8]}-"


# ---------------------------------------------------------------------------
# XDG isolation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set ``XDG_CONFIG_HOME`` and ``XDG_RUNTIME_DIR`` to a temp dir.

    Returns the temp directory root. The env vars are restored on
    teardown via the ``monkeypatch`` fixture.
    """
    xdg_root = tmp_path / "xdg"
    config_home = xdg_root / "config"
    runtime_dir = xdg_root / "runtime"
    config_home.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    # Some libraries cache env-var reads at import time; restore them
    # explicitly at teardown.
    return xdg_root


@pytest.fixture
def temp_unit_dir(tmp_path: Path, clean_xdg: Path) -> Path:
    """Return the temp ``systemd user`` directory under XDG_CONFIG_HOME.

    The directory is created but empty; tests write unit files and
    drop-ins here.
    """
    unit_dir = clean_xdg / "config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    return unit_dir


@pytest.fixture
def temp_profile_path(tmp_path: Path, clean_xdg: Path) -> Path:
    """Return the temp ``profile.json`` path under XDG_CONFIG_HOME."""
    profile_dir = clean_xdg / "config" / "animus"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir / "profile.json"


@pytest.fixture
def temp_registry_path(tmp_path: Path, clean_xdg: Path) -> Path:
    """Return a temp path for the SystemProcessRegistry database."""
    data_dir = clean_xdg / "config" / "animus" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "process_registry.db"


@pytest.fixture
def temp_port() -> int:
    """Allocate a free TCP port from the OS and return it.

    The socket is closed immediately; the port is unlikely to be
    re-attached before the test uses it, but in the rare case of a
    race the test will fail loudly rather than silently collide.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class FakeSystemd:
    """In-memory replacement for ``systemctl --user`` for tests.

    Records every method call so tests can assert against the recorded
    sequence. State is held in dictionaries keyed by unit name. The
    harness can preset state via ``set_unit_state`` and ``set_wants``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._states: dict[str, dict[str, str]] = {}
        self._wants: dict[str, set[str]] = {}
        self._drop_ins: dict[str, dict[str, str]] = {}
        self._drop_in_files: dict[str, set[str]] = {}

    # -- test setup helpers -----------------------------------------------

    def set_unit_state(self, unit: str, **properties: str) -> None:
        """Preset ``systemctl show`` properties for a unit."""
        self._states.setdefault(unit, {}).update(properties)

    def set_wants(self, host_target: str, runtime_target: str) -> None:
        """Preset a host target's ``.wants/`` symlink to the runtime target."""
        self._wants.setdefault(host_target, set()).add(runtime_target)

    def set_drop_in(self, unit: str, filename: str, content: str) -> None:
        """Preset a drop-in file for a unit."""
        self._drop_ins.setdefault(unit, {})[filename] = content
        self._drop_in_files.setdefault(unit, set()).add(filename)

    # -- SwitchBackend surface --------------------------------------------

    def is_target_active(self, target: str) -> bool:
        self.calls.append(("is_target_active", (target,), {}))
        state = self._states.get(target, {})
        return state.get("ActiveState") == "active"

    def daemon_reload(self) -> None:
        self.calls.append(("daemon_reload", (), {}))

    def add_wants(self, host_target: str, runtime_target: str) -> None:
        self.calls.append(("add_wants", (host_target, runtime_target), {}))
        self._wants.setdefault(host_target, set()).add(runtime_target)

    def remove_wants(self, host_target: str, runtime_target: str) -> None:
        self.calls.append(("remove_wants", (host_target, runtime_target), {}))
        self._wants.setdefault(host_target, set()).discard(runtime_target)

    def show(self, unit: str, properties: tuple = ()) -> dict[str, str]:
        self.calls.append(("show", (unit, properties), {}))
        state = dict(self._states.get(unit, {}))
        # Synthesize Wants / Requires / After ONLY when the unit
        # being shown has an explicit ``add_wants`` against it. The
        # real ``systemctl show <target>`` reports the target's
        # outgoing Wants/Requires list, which is what the runtime
        # target's host target will report after ``add-wants``.
        wants_set = self._wants.get(unit, set())
        if wants_set:
            state["Wants"] = " ".join(sorted(wants_set))
        # Inject drop-in-effective MemoryMax / KillMode values.  Real
        # systemd merges drop-ins on top of the base unit file; the
        # fake mirrors that ordering so callers can rely on the
        # same precedence they'd see against ``systemctl show``.
        for filename, content in self._drop_ins.get(unit, {}).items():
            for line in content.splitlines():
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and not key.startswith("["):
                    state[key] = value
        if properties:
            return {k: v for k, v in state.items() if k in properties}
        return state

    def write_drop_in(self, unit: str, filename: str, content: str) -> None:
        self.calls.append(("write_drop_in", (unit, filename), {}))
        self._drop_ins.setdefault(unit, {})[filename] = content
        self._drop_in_files.setdefault(unit, set()).add(filename)

    def remove_drop_in(self, unit: str, filename: str) -> None:
        self.calls.append(("remove_drop_in", (unit, filename), {}))
        self._drop_ins.get(unit, {}).pop(filename, None)
        self._drop_in_files.get(unit, set()).discard(filename)

    def list_drop_ins(self, unit: str) -> list[str]:
        self.calls.append(("list_drop_ins", (unit,), {}))
        return sorted(self._drop_in_files.get(unit, set()))

    # -- SystemdInvoker surface -------------------------------------------

    def show_raw(self, unit: str) -> str:
        """Return a ``KEY=VALUE`` string suitable for ``parse_show_output``."""
        state = self._states.get(unit, {})
        return "\n".join(f"{k}={v}" for k, v in state.items())

    def list_drop_ins_for_invoker(self, unit: str) -> list[str]:
        return self.list_drop_ins(unit)

    # -- assertions -------------------------------------------------------

    def has_wants(self, host_target: str, runtime_target: str) -> bool:
        return runtime_target in self._wants.get(host_target, set())

    def drop_in_files(self, unit: str) -> list[str]:
        return sorted(self._drop_in_files.get(unit, set()))


@pytest.fixture
def fake_systemd() -> FakeSystemd:
    """Return a fresh :class:`FakeSystemd` for the test."""
    return FakeSystemd()
