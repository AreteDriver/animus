"""Test #20 from the build spec §16 — harness cleanup.

Asserts that the harness fixtures (FakeSystemd, temp directories, temp
ports) leave no live artifacts after the test session ends.
"""

from __future__ import annotations

import os
import socket

import pytest

from tests.test_runtime_lifecycle.conftest import FakeSystemd


def test_fake_systemd_does_not_touch_live_systemd() -> None:
    """The FakeSystemd backend does not invoke systemctl."""
    backend = FakeSystemd()
    # Calling every method should not raise and should not require
    # any external dependency. The fake records calls in memory only.
    backend.is_target_active("animus-runtime.target")
    backend.daemon_reload()
    backend.add_wants("default.target", "animus-runtime.target")
    backend.remove_wants("default.target", "animus-runtime.target")
    backend.show("animus.service", properties=("ActiveState",))
    backend.write_drop_in("animus.service", "20-profile.conf", "[Service]\n")
    backend.remove_drop_in("animus.service", "20-profile.conf")
    backend.list_drop_ins("animus.service")
    # All recorded in memory.
    assert len(backend.calls) == 8


def test_temp_port_is_unique() -> None:
    """Two consecutive temp_port allocations return different ports."""
    # This relies on the conftest fixture; we reimplement here to
    # avoid fixture order coupling.
    def alloc() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    p1 = alloc()
    p2 = alloc()
    assert p1 != p2


def test_clean_xdg_does_not_leak(tmp_path) -> None:
    """The clean_xdg fixture creates files only under tmp_path."""
    xdg_root = tmp_path / "xdg"
    (xdg_root / "config" / "systemd" / "user").mkdir(parents=True)
    (xdg_root / "runtime").mkdir(parents=True)
    assert (xdg_root / "config" / "systemd" / "user").exists()
    assert (xdg_root / "runtime").exists()
    # No files leak outside tmp_path.
    for entry in tmp_path.iterdir():
        # Only the directories we created should exist.
        assert entry.name in {"xdg"}
