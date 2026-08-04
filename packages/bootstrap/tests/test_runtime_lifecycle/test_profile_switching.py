"""Tests #9, #10, #12, #13 from the build spec §16.

Profile switching behavior:

- #9: switching creates the intended target-wants symlink.
- #10: switching removes obsolete symlinks.
- #12: failed switching rolls back.
- #13: development install enables no runtime target.
"""

from __future__ import annotations

import pytest

from animus_bootstrap.lifecycle.profile import (
    ProfileConfig,
    ProfileMode,
    ProfileSwitcher,
)


def test_profile_switch_creates_intended_symlink() -> None:
    from tests.test_runtime_lifecycle.conftest import FakeSystemd

    backend = FakeSystemd()
    backend.set_unit_state("animus-runtime.target", Wants="", Requires="")
    backend.set_unit_state("animus.service", ActiveState="inactive")
    switcher = ProfileSwitcher(backend=backend)
    result = switcher.switch(
        current=ProfileConfig(mode=ProfileMode.DEVELOPMENT_LOCAL),
        target_mode=ProfileMode.DESKTOP_LOGIN,
    )
    assert result.success, result.error
    assert backend.has_wants("graphical-session.target", "animus-runtime.target")


def test_profile_switch_removes_obsolete_symlinks() -> None:
    """Switching from desktop-login back to development-local removes
    the graphical-session.target.wants/ symlink."""
    from tests.test_runtime_lifecycle.conftest import FakeSystemd

    backend = FakeSystemd()
    backend.set_unit_state("animus-runtime.target", Wants="", Requires="")
    backend.set_unit_state("animus.service", ActiveState="inactive")
    backend.set_wants("graphical-session.target", "animus-runtime.target")

    switcher = ProfileSwitcher(backend=backend)
    result = switcher.switch(
        current=ProfileConfig(mode=ProfileMode.DESKTOP_LOGIN),
        target_mode=ProfileMode.DEVELOPMENT_LOCAL,
    )
    assert result.success, result.error
    # After switch, no symlink should remain.
    assert not backend.has_wants("graphical-session.target", "animus-runtime.target")


def test_failed_switch_rolls_back() -> None:
    """When verification fails, the prior state is restored.

    The switch raises :class:`ProfileSwitchError` from verification;
    the switcher catches it and rolls back by removing the new
    drop-in.  We force the failure by wrapping the daemon's
    ``daemon_reload`` so the first invocation raises — this exercises
    the same rollback path the switcher uses for any exception
    inside the transaction.
    """
    from tests.test_runtime_lifecycle.conftest import FakeSystemd

    backend = FakeSystemd()
    backend.set_unit_state("animus-runtime.target", Wants="", Requires="")
    backend.set_unit_state("animus.service", ActiveState="inactive")
    # Pre-populate a prior drop-in so we can verify rollback removes
    # the new drop-in and that the prior drop-in file is left alone.
    backend.set_drop_in(
        "animus.service",
        "20-profile-development-local.conf",
        "MemoryMax=4G\n",
    )

    # Wrap daemon_reload so the transaction fails immediately.
    original_daemon_reload = backend.daemon_reload

    def failing_daemon_reload() -> None:
        original_daemon_reload()
        raise RuntimeError("simulated daemon-reload failure")

    backend.daemon_reload = failing_daemon_reload  # type: ignore[assignment]

    switcher = ProfileSwitcher(backend=backend)
    result = switcher.switch(
        current=ProfileConfig(mode=ProfileMode.DEVELOPMENT_LOCAL),
        target_mode=ProfileMode.DESKTOP_LOGIN,
    )
    assert not result.success
    assert result.rollback
    # The rollback removed the new drop-in that the switcher wrote
    # just before the daemon-reload call.
    assert (
        "20-profile-desktop-login.conf"
        not in backend.drop_in_files("animus.service")
    )
    # The prior drop-in is still present.
    assert "20-profile-development-local.conf" in backend.drop_in_files(
        "animus.service"
    )


def test_continuous_node_requires_user_consent() -> None:
    """Switching to continuous-node without explicit user_consent=True
    is refused."""
    from tests.test_runtime_lifecycle.conftest import FakeSystemd

    backend = FakeSystemd()
    backend.set_unit_state("animus.service", ActiveState="inactive")
    backend.set_unit_state("animus-runtime.target", Wants="", Requires="")
    switcher = ProfileSwitcher(backend=backend)
    result = switcher.switch(
        current=ProfileConfig(mode=ProfileMode.DESKTOP_LOGIN),
        target_mode=ProfileMode.CONTINUOUS_NODE,
        user_consent=False,
    )
    assert not result.success
    assert "user_consent" in (result.error or "")


def test_continuous_node_with_user_consent_succeeds() -> None:
    from tests.test_runtime_lifecycle.conftest import FakeSystemd

    backend = FakeSystemd()
    backend.set_unit_state("animus.service", ActiveState="inactive")
    backend.set_unit_state("animus-runtime.target", Wants="", Requires="")
    switcher = ProfileSwitcher(backend=backend)
    result = switcher.switch(
        current=ProfileConfig(mode=ProfileMode.DESKTOP_LOGIN),
        target_mode=ProfileMode.CONTINUOUS_NODE,
        user_consent=True,
    )
    assert result.success, result.error
    assert backend.has_wants("default.target", "animus-runtime.target")


def test_development_local_creates_no_symlinks() -> None:
    """Test #13: development-local profile never creates target.wants/."""
    from tests.test_runtime_lifecycle.conftest import FakeSystemd

    backend = FakeSystemd()
    backend.set_unit_state("animus.service", ActiveState="inactive")
    backend.set_unit_state("animus-runtime.target", Wants="", Requires="")
    switcher = ProfileSwitcher(backend=backend)
    # Initial switch from development-local to development-local is a
    # no-op; it should not create any symlinks.
    result = switcher.switch(
        current=ProfileConfig(mode=ProfileMode.DEVELOPMENT_LOCAL),
        target_mode=ProfileMode.DEVELOPMENT_LOCAL,
    )
    assert result.success, result.error
    add_wants_calls = [c for c in backend.calls if c[0] == "add_wants"]
    assert not add_wants_calls, "no add_wants calls expected for dev profile"


def test_drop_in_files_have_profile_prefix() -> None:
    """Drop-ins follow the 20-profile-<mode>.conf naming convention."""
    from tests.test_runtime_lifecycle.conftest import FakeSystemd

    backend = FakeSystemd()
    backend.set_unit_state("animus.service", ActiveState="inactive")
    backend.set_unit_state("animus-runtime.target", Wants="", Requires="")
    switcher = ProfileSwitcher(backend=backend)
    result = switcher.switch(
        current=ProfileConfig(mode=ProfileMode.DEVELOPMENT_LOCAL),
        target_mode=ProfileMode.DESKTOP_LOGIN,
    )
    assert result.success, result.error
    files = backend.drop_in_files("animus.service")
    assert any("20-profile-desktop-login" in f for f in files)
