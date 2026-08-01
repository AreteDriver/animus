"""Session management façade.

Encapsulates all imports from ``animus_kernel.head`` so the rest of Core
(and upper layers) do not reach into Kernel internals directly.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_head_repl",
    "create_session_controller",
    "SessionLifecycleEvent",
]


def get_head_repl(
    model: str | None = None,
    project_root: str | None = None,
    session_timer: Any = None,
    wrapup_threshold: float = 0.96,
):
    """Factory for ``HeadREPL``.  Deferred import preserves Core-only installability."""
    from animus_kernel.head.repl import HeadREPL  # boundary-ok: façade delegates to Kernel

    return HeadREPL(
        model=model,
        project_root=project_root,
        session_timer=session_timer,
        wrapup_threshold=wrapup_threshold,
    )


def create_session_controller(
    wrapup_threshold: float = 0.96,
    session_timer_minutes: int = 30,
    auto_restart: bool = True,
):
    """Build a ``SessionController`` from telemetry-style parameters."""
    from animus_kernel.head.session_controller import (  # boundary-ok: façade delegates to Kernel
        SessionController,
        SessionPolicy,
    )

    policy = SessionPolicy(
        wrapup_threshold=wrapup_threshold,
        session_timer=__import__("datetime", fromlist=["timedelta"]).timedelta(
            minutes=session_timer_minutes
        ),
        auto_restart=auto_restart,
    )
    return SessionController(policy=policy)


class _SessionLifecycleEventProxy:
    """Lazy proxy for ``SessionLifecycleEvent`` enum values.

    The caller does ``SessionLifecycleEvent["RUNNING"]`` and we delegate to the
    real enum on first attribute access.  This avoids importing Kernel at module
    level while still exposing the enum surface.
    """

    _real = None

    def _load(self) -> Any:
        if self._real is None:
            from animus_kernel.head.session_controller import (
                SessionLifecycleEvent as _Real,  # boundary-ok: façade delegates to Kernel
            )

            self._real = _Real
        return self._real

    def __getitem__(self, name: str) -> Any:
        return self._load()[name]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


SessionLifecycleEvent: Any = _SessionLifecycleEventProxy()
