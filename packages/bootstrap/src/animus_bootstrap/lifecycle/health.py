"""Health-state derivation and the versioned health contract.

Implements the seven-state ``HealthState`` enum from ADR-007 and the
versioned ``HealthSnapshot`` schema for ``/healthz``.

The control app and dashboard consume these primitives. The
:class:`derive_health_state` function is pure and is the primary test
surface for the three failure walkthroughs in ADR-007.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

logger = logging.getLogger("animus_bootstrap.lifecycle.health")


class HealthState(str, Enum):  # noqa: UP042 - preserve API enum string behavior
    """Seven-state health enum, per ADR-007.

    Distinct from the systemd ``ActiveState`` (``active`` / ``inactive``
    / ``failed`` / ``activating`` / ``deactivating``). ``HealthState`` is
    the *user-facing* state derived from systemd state plus the health
    contract.
    """

    OFFLINE = "offline"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


# Schema version for the health contract. Bump on backward-incompatible
# changes. Document the new version in ``docs/operations/health-contract.md``.
HEALTH_CONTRACT_VERSION = "1"


@dataclass(frozen=True)
class ServiceHealth:
    """Health input for one systemd service participating in the runtime."""

    unit: str
    is_active: bool | None  # None = unknown
    is_required: bool  # True for the daemon; False for optional
    health_probe_ok: bool | None = None  # None = no probe data


@dataclass(frozen=True)
class HealthSnapshot:
    """Versioned health response — the producer side of the contract.

    The daemon produces this JSON; the control app and dashboard parse
    it via :class:`HealthContract.parse`.
    """

    schema_version: Literal["1"]
    timestamp: datetime
    state: HealthState
    active_citizens: int
    open_jobs: int
    last_heartbeat_age_seconds: float
    detail: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "active_citizens": self.active_citizens,
            "open_jobs": self.open_jobs,
            "last_heartbeat_age_seconds": self.last_heartbeat_age_seconds,
            "detail": dict(self.detail),
        }


@dataclass
class HealthContract:
    """Producer and consumer for the versioned health contract.

    The contract is intentionally narrow: it is what the daemon promises
    to return and what the control app + dashboard promise to accept.
    """

    schema_version: Literal["1"] = "1"  # type: ignore[assignment]

    def produce(
        self,
        *,
        state: HealthState,
        active_citizens: int,
        open_jobs: int,
        last_heartbeat_age_seconds: float,
        detail: dict[str, str] | None = None,
    ) -> HealthSnapshot:
        """Producer side: the daemon wraps its state in a HealthSnapshot."""
        if active_citizens < 0 or open_jobs < 0:
            raise ValueError("active_citizens and open_jobs must be >= 0")
        if last_heartbeat_age_seconds < 0:
            raise ValueError("last_heartbeat_age_seconds must be >= 0")
        return HealthSnapshot(
            schema_version=self.schema_version,
            timestamp=datetime.now(UTC),
            state=state,
            active_citizens=active_citizens,
            open_jobs=open_jobs,
            last_heartbeat_age_seconds=last_heartbeat_age_seconds,
            detail=dict(detail or {}),
        )

    def parse(self, payload: dict[str, Any]) -> HealthSnapshot:
        """Consumer side: validate and parse the daemon's response.

        Raises:
            ValueError: if the payload is missing required fields, has
                the wrong schema_version, or has invalid types.
        """
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        version = payload.get("schema_version")
        if version != self.schema_version:
            raise ValueError(
                f"unsupported schema_version: {version!r} (expected {self.schema_version!r})"
            )
        ts_raw = payload.get("timestamp")
        if not isinstance(ts_raw, str):
            raise ValueError("timestamp must be an ISO-8601 string")
        timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must include timezone")
        state_raw = payload.get("state")
        try:
            state = HealthState(state_raw)
        except ValueError as exc:
            raise ValueError(f"invalid state: {state_raw!r}") from exc
        active_citizens = payload.get("active_citizens")
        if not isinstance(active_citizens, int) or active_citizens < 0:
            raise ValueError("active_citizens must be a non-negative int")
        open_jobs = payload.get("open_jobs")
        if not isinstance(open_jobs, int) or open_jobs < 0:
            raise ValueError("open_jobs must be a non-negative int")
        last_heartbeat = payload.get("last_heartbeat_age_seconds")
        if not isinstance(last_heartbeat, (int, float)) or last_heartbeat < 0:
            raise ValueError("last_heartbeat_age_seconds must be a non-negative number")
        detail = payload.get("detail") or {}
        if not isinstance(detail, dict):
            raise ValueError("detail must be a dict")
        for k, v in detail.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("detail keys and values must be strings")
        return HealthSnapshot(
            schema_version=version,
            timestamp=timestamp,
            state=state,
            active_citizens=active_citizens,
            open_jobs=open_jobs,
            last_heartbeat_age_seconds=float(last_heartbeat),
            detail=detail,
        )


def derive_health_state(
    *,
    runtime_target_active: bool | None,
    required_daemon: ServiceHealth,
    optional_services: list[ServiceHealth],
    health_snapshot: HealthSnapshot | None,
) -> HealthState:
    """Derive the seven-state ``HealthState`` from authoritative inputs.

    Pure function. The control app and dashboard call this with the
    data they have; the result is what the user sees.

    Rules (per ADR-007):

    1. If both authoritative signals (systemd state and the
       snapshot) are unavailable, return ``UNKNOWN``. Honest
       uncertainty.
    2. If the runtime target is inactive, return ``OFFLINE``.
    3. If the runtime target is stopping, return ``STOPPING``.
    4. If the runtime target is starting, return ``STARTING``.
    5. If the required daemon is not active, return ``FAILED``.
    6. If the health snapshot is present and reports ``FAILED`` or
       ``DEGRADED``, propagate that.
    7. If any optional service failed, return ``DEGRADED``.
    8. Otherwise return ``HEALTHY``.

    The three failure walkthroughs in ADR-007 exercise cases 5, 6, and
    7 with the optional services healthy and the health probe failing.
    """
    # Rule 1: missing both signals -> UNKNOWN
    if runtime_target_active is None and health_snapshot is None:
        return HealthState.UNKNOWN

    # If the runtime target is active and the health probe is also
    # unavailable, the result is still UNKNOWN. We have only one
    # signal at that point.
    if (
        runtime_target_active is True
        and health_snapshot is None
        and required_daemon.is_active is None
    ):
        return HealthState.UNKNOWN

    # Rule 2-4: terminal/intermediate states
    if runtime_target_active is False:
        return HealthState.OFFLINE
    if runtime_target_active is None:
        # We have partial info but no target state. If the snapshot
        # says OFFLINE/STOPPING, propagate. Otherwise UNKNOWN.
        if health_snapshot is not None:
            if health_snapshot.state == HealthState.OFFLINE:
                return HealthState.OFFLINE
            if health_snapshot.state == HealthState.STOPPING:
                return HealthState.STOPPING
        return HealthState.UNKNOWN

    # At this point runtime_target_active is True.
    # Rule 5: required daemon not active -> FAILED
    if required_daemon.is_active is False:
        return HealthState.FAILED

    # Rule 6: snapshot says FAILED
    if health_snapshot is not None and health_snapshot.state == HealthState.FAILED:
        return HealthState.FAILED

    # Snapshot says STARTING — propagate. Starting is distinct from
    # Degraded: the daemon is not yet ready, but it is on its way.
    if health_snapshot is not None and health_snapshot.state == HealthState.STARTING:
        return HealthState.STARTING

    # Rule 7: any optional service failed -> DEGRADED
    if any(s.is_active is False for s in optional_services):
        return HealthState.DEGRADED

    # Snapshot says DEGRADED
    if health_snapshot is not None and health_snapshot.state == HealthState.DEGRADED:
        return HealthState.DEGRADED

    # Required daemon is reported as None (no signal) while the
    # target is active. Honest uncertainty.
    if required_daemon.is_active is None:
        return HealthState.UNKNOWN

    # Rule 8
    return HealthState.HEALTHY
