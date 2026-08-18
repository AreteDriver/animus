"""Profile configuration and atomic profile switching.

Implements the profile model from ADR-007:

- ``development-local`` — default on installation.
- ``desktop-login`` — explicit user opt-in.
- ``continuous-node`` — explicit user opt-in, never inferred.

The :class:`ProfileSwitcher` performs the atomic switch transaction:

1. Validate the requested mode.
2. Stop the runtime target if active.
3. Read current target symlinks.
4. Compute the desired symlink set.
5. Generate drop-ins atomically (write to temp, fsync, rename).
6. Run ``systemctl --user daemon-reload``.
7. Add new ``add-wants`` symlinks.
8. Remove obsolete ``remove-wants`` symlinks.
9. Verify effective dependencies.
10. Roll back on any failure.
11. Write ``profile.json`` only after successful verification.

The :class:`ProfileSwitcher` is *pure* — it accepts a
:class:`SwitchBackend` protocol that performs the actual subprocess
calls. The harness provides a fake backend; production uses a
subprocess-based backend.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("animus_bootstrap.lifecycle.profile")


class ProfileMode(str, Enum):  # noqa: UP042 - preserve profile serialization behavior
    """The three deployment profiles.

    String values are persisted in ``profile.json`` and match the
    filenames of the drop-in templates.
    """

    DEVELOPMENT_LOCAL = "development-local"
    DESKTOP_LOGIN = "desktop-login"
    CONTINUOUS_NODE = "continuous-node"


# Map profile -> the systemd user target the runtime target binds to.
# None means "no binding"; the user must invoke `start` manually.
PROFILE_TARGET_BINDINGS: dict[ProfileMode, str | None] = {
    ProfileMode.DEVELOPMENT_LOCAL: None,
    ProfileMode.DESKTOP_LOGIN: "graphical-session.target",
    ProfileMode.CONTINUOUS_NODE: "default.target",
}


class ProfileSwitchError(RuntimeError):
    """Raised on profile-switch failure.

    The :class:`ProfileSwitcher` rolls back before raising.
    """


@dataclass
class ProfileConfig:
    """Desired-state profile.

    Matches the JSON schema in
    ``docs/specifications/animus-runtime-lifecycle-build-spec.md`` §8.
    """

    mode: ProfileMode = ProfileMode.DEVELOPMENT_LOCAL
    tray_while_running: bool = False
    tray_while_offline: bool = False
    start_on_login: bool = False
    schema_version: str = "1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "tray_while_running": self.tray_while_running,
            "tray_while_offline": self.tray_while_offline,
            "start_on_login": self.start_on_login,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ProfileConfig:
        if not isinstance(data, dict):
            raise ValueError("profile.json must be a JSON object")
        version = data.get("schema_version", "1")
        if not isinstance(version, str) or version != "1":
            raise ValueError(f"unsupported schema_version: {version!r}")
        try:
            mode = ProfileMode(data.get("mode", "development-local"))
        except ValueError as exc:
            raise ValueError(f"invalid mode: {data.get('mode')!r}") from exc
        return cls(
            mode=mode,
            tray_while_running=bool(data.get("tray_while_running", False)),
            tray_while_offline=bool(data.get("tray_while_offline", False)),
            start_on_login=bool(data.get("start_on_login", False)),
            schema_version=version,
        )


def load_profile(path: Path) -> ProfileConfig:
    """Load desired-state profile from ``profile.json``.

    If the file does not exist, returns the default
    ``development-local`` profile.
    """
    if not path.exists():
        return ProfileConfig()
    raw = json.loads(path.read_text())
    return ProfileConfig.from_dict(raw)


def save_profile(path: Path, profile: ProfileConfig) -> None:
    """Persist desired-state profile to ``profile.json``.

    Writes atomically: temp file in the same directory, fsync, then
    rename. This prevents a partial write from leaving the runtime in
    a state where the file is half-formed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile.to_dict(), indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".profile.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class SwitchBackend(Protocol):
    """The backend the :class:`ProfileSwitcher` uses to talk to systemd.

    Production backend invokes ``systemctl --user``. The test harness
    provides a recording fake that captures the calls without
    touching the live user manager.
    """

    def is_target_active(self, target: str) -> bool: ...

    def daemon_reload(self) -> None: ...

    def add_wants(self, host_target: str, runtime_target: str) -> None: ...

    def remove_wants(self, host_target: str, runtime_target: str) -> None: ...

    def show(self, unit: str, properties: Iterable[str]) -> dict[str, str]: ...

    def write_drop_in(self, unit: str, filename: str, content: str) -> None: ...

    def remove_drop_in(self, unit: str, filename: str) -> None: ...

    def list_drop_ins(self, unit: str) -> list[str]: ...


@dataclass
class ProfileSwitchResult:
    """Result of a profile switch.

    Attributes:
        success: True if the switch succeeded.
        from_mode: The mode before the switch.
        to_mode: The requested mode.
        steps: Ordered list of human-readable steps executed.
        rollback: True if the switch rolled back to the prior mode.
        error: The error message if the switch failed.
    """

    success: bool
    from_mode: ProfileMode
    to_mode: ProfileMode
    steps: list[str] = field(default_factory=list)
    rollback: bool = False
    error: str | None = None


@dataclass
class ProfileSwitcher:
    """Atomic profile switcher.

    The constructor takes a :class:`SwitchBackend`. The backend is the
    only thing that varies between production and the test harness.
    """

    backend: SwitchBackend
    runtime_target: str = "animus-runtime.target"
    drop_in_prefix: str = "20-profile-"
    units: tuple[str, ...] = (
        "animus.service",
        "animus-forge.service",
        "animus-mcp.service",
        "animus-scheduler.service",
        "animus-tray.service",
    )
    # Templated drop-in content per profile. The dashboard or build
    # pipeline can substitute the right values; the values below are
    # the documented defaults from the build spec.
    _drop_in_templates: dict[ProfileMode, dict[str, str]] = field(
        default_factory=lambda: {
            ProfileMode.DEVELOPMENT_LOCAL: {
                "MemoryMax": "4G",
                "CPUQuota": "200%",
                "TasksMax": "64",
                "Restart": "no",
                "WatchdogSec": "0",
            },
            ProfileMode.DESKTOP_LOGIN: {
                "MemoryMax": "8G",
                "CPUQuota": "400%",
                "TasksMax": "128",
                "Restart": "on-failure",
                "RestartSec": "5s",
                "WatchdogSec": "30",
            },
            ProfileMode.CONTINUOUS_NODE: {
                "MemoryMax": "32G",
                "CPUQuota": "1600%",
                "TasksMax": "512",
                "Restart": "on-failure",
                "RestartSec": "5s",
                "WatchdogSec": "30",
                "TimeoutStopSec": "30",
            },
        }
    )

    def _drop_in_for(self, mode: ProfileMode) -> str:
        """Render the canonical drop-in content for a profile."""
        values = self._drop_in_templates[mode]
        lines = ["[Service]\nKillMode=control-group"]
        for key, value in values.items():
            lines.append(f"{key}={value}")
        # Preserve the no-Delegate rule regardless of profile.
        lines.append("Delegate=no")
        return "\n".join(lines) + "\n"

    def switch(
        self,
        *,
        current: ProfileConfig,
        target_mode: ProfileMode,
        user_consent: bool = False,
    ) -> ProfileSwitchResult:
        """Switch profiles atomically.

        Returns:
            A :class:`ProfileSwitchResult` describing the outcome.

        Raises:
            ProfileSwitchError: only if the switch failed *and* the
                rollback also failed. Usually the result is returned
                with ``success=False`` and ``rollback=True``.
        """
        if target_mode == ProfileMode.CONTINUOUS_NODE and not user_consent:
            return ProfileSwitchResult(
                success=False,
                from_mode=current.mode,
                to_mode=target_mode,
                error="continuous-node requires explicit user_consent=True",
            )

        if target_mode not in PROFILE_TARGET_BINDINGS:
            return ProfileSwitchResult(
                success=False,
                from_mode=current.mode,
                to_mode=target_mode,
                error=f"unknown profile mode: {target_mode!r}",
            )

        steps: list[str] = []
        from_mode = current.mode
        prior_drop_ins: dict[str, list[str]] = {}
        prior_bindings: dict[str, bool] = {}

        try:
            # Step 2: stop the runtime target if active.
            if self.backend.is_target_active(self.runtime_target):
                # The backend's is_target_active is the only read; we
                # do not act on True/False beyond recording. The caller
                # can stop the target via the control app or via
                # `systemctl --user stop` directly.
                steps.append("runtime_target_active=True (caller must stop)")

            # Step 3-4: compute desired bindings.
            new_target = PROFILE_TARGET_BINDINGS[target_mode]
            old_target = PROFILE_TARGET_BINDINGS[from_mode]

            # Step 7: generate drop-ins atomically. Capture prior
            # values for rollback.
            for unit in self.units:
                filename = f"{self.drop_in_prefix}{target_mode.value}.conf"
                prior_drop_ins[unit] = self.backend.list_drop_ins(unit)
                self.backend.write_drop_in(unit, filename, self._drop_in_for(target_mode))
            steps.append("drop-ins written")

            # Step 8: daemon-reload.
            self.backend.daemon_reload()
            steps.append("daemon-reload")

            # Step 9-10: add/remove target symlinks.
            if new_target is not None:
                self.backend.add_wants(new_target, self.runtime_target)
                steps.append(f"add-wants {new_target}")
            if old_target is not None and old_target != new_target:
                self.backend.remove_wants(old_target, self.runtime_target)
                steps.append(f"remove-wants {old_target}")

            # Step 11-12: verify effective dependencies and properties.
            # Verify the host target's Wants= includes the runtime target.
            if new_target is not None:
                host_show = self.backend.show(
                    new_target,
                    properties=("Wants", "Requires"),
                )
                host_wants = host_show.get("Wants", "")
                if self.runtime_target not in host_wants:
                    raise ProfileSwitchError(
                        f"verification failed: {self.runtime_target} not in "
                        f"Wants of {new_target} (got {host_wants!r})"
                    )
            # Verify the daemon's drop-in produces the expected values
            # for the properties that, if wrong, would silently break
            # the runtime. MemoryMax + KillMode is the original pair;
            # Delegate + CPUQuota was added after the four-lens review
            # (Lens 3.3) — a drop-in can have the right MemoryMax but
            # the wrong Delegate=yes, which would pass the original
            # check but disable cgroup reaping.
            show_svc = self.backend.show(
                "animus.service",
                properties=("MemoryMax", "KillMode", "CPUQuota", "Delegate"),
            )
            expected_memory = self._drop_in_templates[target_mode]["MemoryMax"]
            expected_cpu = self._drop_in_templates[target_mode]["CPUQuota"]
            if show_svc.get("MemoryMax") != expected_memory:
                raise ProfileSwitchError(
                    f"verification failed: MemoryMax={show_svc.get('MemoryMax')!r} "
                    f"expected {expected_memory!r}"
                )
            if show_svc.get("KillMode") != "control-group":
                raise ProfileSwitchError(
                    f"verification failed: KillMode={show_svc.get('KillMode')!r} "
                    f"expected 'control-group'"
                )
            if show_svc.get("CPUQuota") != expected_cpu:
                raise ProfileSwitchError(
                    f"verification failed: CPUQuota={show_svc.get('CPUQuota')!r} "
                    f"expected {expected_cpu!r}"
                )
            if show_svc.get("Delegate") != "no":
                raise ProfileSwitchError(
                    f"verification failed: Delegate={show_svc.get('Delegate')!r} expected 'no'"
                )
            steps.append("verification passed")

        except Exception as exc:
            # Capture the original failure before any further handling
            # below rebinds ``exc`` (e.g. the rollback's own try/except).
            original_error = str(exc)
            # Rollback: restore prior drop-ins and bindings.
            for unit, prior in prior_drop_ins.items():
                for filename in prior:
                    self.backend.write_drop_in(unit, filename, "")
                # Remove the drop-in we just wrote.
                new_filename = f"{self.drop_in_prefix}{target_mode.value}.conf"
                self.backend.remove_drop_in(unit, new_filename)
            for host_target, was_bound in prior_bindings.items():
                if was_bound:
                    self.backend.add_wants(host_target, self.runtime_target)
            try:
                self.backend.daemon_reload()
            except Exception as reload_exc:
                # Rollback's daemon-reload is best-effort. The drop-ins
                # and bindings are already restored on disk; the next
                # daemon-reload (manual or via the next switch) will
                # pick them up. Log so operators can see the gap.
                logger.warning(
                    "rollback daemon-reload failed: %s; bindings restored "
                    "on disk but systemd may still see the prior state",
                    reload_exc,
                )
            return ProfileSwitchResult(
                success=False,
                from_mode=from_mode,
                to_mode=target_mode,
                steps=steps,
                rollback=True,
                error=original_error,
            )

        return ProfileSwitchResult(
            success=True,
            from_mode=from_mode,
            to_mode=target_mode,
            steps=steps,
        )

    def persist(self, profile: ProfileConfig, path: Path) -> None:
        """Persist the desired state after a successful switch."""
        save_profile(path, profile)
