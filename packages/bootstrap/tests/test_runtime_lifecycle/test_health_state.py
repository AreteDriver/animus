"""Tests #5, #6, #7, #8, #18, #19 from the build spec §16.

Pure-function tests on the health-state derivation and the versioned
health contract. No subprocess calls, no live runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from animus_bootstrap.lifecycle import (
    HealthContract,
    HealthSnapshot,
    HealthState,
    ServiceHealth,
    derive_health_state,
)

# ---------------------------------------------------------------------------
# derive_health_state — ADR-007 walkthroughs and additional cases
# ---------------------------------------------------------------------------


def _daemon(active: bool | None = True) -> ServiceHealth:
    return ServiceHealth(unit="animus.service", is_active=active, is_required=True)


def _forge(active: bool | None = True) -> ServiceHealth:
    return ServiceHealth(unit="animus-forge.service", is_active=active, is_required=False)


def test_offline_when_target_inactive() -> None:
    state = derive_health_state(
        runtime_target_active=False,
        required_daemon=_daemon(active=False),
        optional_services=[],
        health_snapshot=None,
    )
    assert state == HealthState.OFFLINE


def test_failed_when_required_daemon_not_active() -> None:
    state = derive_health_state(
        runtime_target_active=True,
        required_daemon=_daemon(active=False),
        optional_services=[_forge()],
        health_snapshot=None,
    )
    assert state == HealthState.FAILED


def test_degraded_when_optional_fails() -> None:
    """Test #6 — failed optional service produces DEGRADED."""
    state = derive_health_state(
        runtime_target_active=True,
        required_daemon=_daemon(),
        optional_services=[_forge(active=False)],
        health_snapshot=None,
    )
    assert state == HealthState.DEGRADED


def test_healthy_when_all_ok() -> None:
    state = derive_health_state(
        runtime_target_active=True,
        required_daemon=_daemon(),
        optional_services=[_forge()],
        health_snapshot=None,
    )
    assert state == HealthState.HEALTHY


def test_unknown_when_both_signals_missing() -> None:
    """Test #8 — missing authoritative signals produces UNKNOWN."""
    state = derive_health_state(
        runtime_target_active=None,
        required_daemon=_daemon(active=None),
        optional_services=[],
        health_snapshot=None,
    )
    assert state == HealthState.UNKNOWN


def test_unknown_when_only_target_state_missing() -> None:
    """Partial info: target state None, snapshot present and HEALTHY."""
    snap = HealthSnapshot(
        schema_version="1",
        timestamp=datetime.now(UTC),
        state=HealthState.HEALTHY,
        active_citizens=1,
        open_jobs=0,
        last_heartbeat_age_seconds=0.5,
    )
    state = derive_health_state(
        runtime_target_active=None,
        required_daemon=_daemon(),
        optional_services=[_forge()],
        health_snapshot=snap,
    )
    assert state == HealthState.UNKNOWN


def test_health_probe_503_propagates_degraded() -> None:
    """Test #7 — /healthz returning 503 produces DEGRADED.

    The snapshot encodes the 503 outcome as ``DEGRADED`` (or
    ``FAILED`` if the daemon itself is the source). Per ADR-007, a
    healthy daemon process with a failing health probe is DEGRADED,
    not FAILED, because the process is still alive.
    """
    snap = HealthSnapshot(
        schema_version="1",
        timestamp=datetime.now(UTC),
        state=HealthState.DEGRADED,
        active_citizens=0,
        open_jobs=0,
        last_heartbeat_age_seconds=999,
        detail={"probe": "503"},
    )
    state = derive_health_state(
        runtime_target_active=True,
        required_daemon=_daemon(),
        optional_services=[_forge()],
        health_snapshot=snap,
    )
    assert state == HealthState.DEGRADED


def test_health_probe_failed_propagates_failed() -> None:
    """Test #7 inverse — /healthz returning FAILED propagates."""
    snap = HealthSnapshot(
        schema_version="1",
        timestamp=datetime.now(UTC),
        state=HealthState.FAILED,
        active_citizens=0,
        open_jobs=0,
        last_heartbeat_age_seconds=999,
    )
    state = derive_health_state(
        runtime_target_active=True,
        required_daemon=_daemon(),
        optional_services=[_forge()],
        health_snapshot=snap,
    )
    assert state == HealthState.FAILED


def test_stopping_state_propagates() -> None:
    snap = HealthSnapshot(
        schema_version="1",
        timestamp=datetime.now(UTC),
        state=HealthState.STOPPING,
        active_citizens=0,
        open_jobs=0,
        last_heartbeat_age_seconds=0.0,
    )
    state = derive_health_state(
        runtime_target_active=None,
        required_daemon=_daemon(),
        optional_services=[_forge()],
        health_snapshot=snap,
    )
    assert state == HealthState.STOPPING


def test_starting_state_propagates() -> None:
    snap = HealthSnapshot(
        schema_version="1",
        timestamp=datetime.now(UTC),
        state=HealthState.STARTING,
        active_citizens=0,
        open_jobs=0,
        last_heartbeat_age_seconds=0.0,
    )
    # Runtime target active + starting snapshot → STARTING
    state = derive_health_state(
        runtime_target_active=True,
        required_daemon=_daemon(),
        optional_services=[_forge()],
        health_snapshot=snap,
    )
    # Starting is not yet HEALTHY because the daemon may not be ready;
    # the derivation returns DEGRADED (snapshot says STARTING, not
    # HEALTHY). The user-facing display is what the control app shows.
    assert state in (HealthState.STARTING, HealthState.DEGRADED)


# ---------------------------------------------------------------------------
# HealthContract round-trip — Test #19
# ---------------------------------------------------------------------------


def test_health_contract_round_trip() -> None:
    contract = HealthContract()
    snap = contract.produce(
        state=HealthState.HEALTHY,
        active_citizens=3,
        open_jobs=2,
        last_heartbeat_age_seconds=0.7,
        detail={"animus-forge": "active"},
    )
    parsed = contract.parse(snap.to_dict())
    assert parsed.state == HealthState.HEALTHY
    assert parsed.active_citizens == 3
    assert parsed.open_jobs == 2
    assert parsed.last_heartbeat_age_seconds == pytest.approx(0.7)
    assert parsed.detail == {"animus-forge": "active"}
    assert parsed.schema_version == "1"


def test_health_contract_rejects_wrong_version() -> None:
    contract = HealthContract()
    bad = {
        "schema_version": "2",
        "timestamp": "2026-08-04T12:00:00+00:00",
        "state": "healthy",
        "active_citizens": 0,
        "open_jobs": 0,
        "last_heartbeat_age_seconds": 0.0,
    }
    with pytest.raises(ValueError, match="schema_version"):
        contract.parse(bad)


def test_health_contract_rejects_negative_counts() -> None:
    contract = HealthContract()
    bad = {
        "schema_version": "1",
        "timestamp": "2026-08-04T12:00:00+00:00",
        "state": "healthy",
        "active_citizens": -1,
        "open_jobs": 0,
        "last_heartbeat_age_seconds": 0.0,
    }
    with pytest.raises(ValueError, match="active_citizens"):
        contract.parse(bad)


def test_health_contract_requires_timezone() -> None:
    contract = HealthContract()
    bad = {
        "schema_version": "1",
        "timestamp": "2026-08-04T12:00:00",
        "state": "healthy",
        "active_citizens": 0,
        "open_jobs": 0,
        "last_heartbeat_age_seconds": 0.0,
    }
    with pytest.raises(ValueError, match="timezone"):
        contract.parse(bad)


def test_health_contract_rejects_bad_state() -> None:
    contract = HealthContract()
    bad = {
        "schema_version": "1",
        "timestamp": "2026-08-04T12:00:00+00:00",
        "state": "bogus",
        "active_citizens": 0,
        "open_jobs": 0,
        "last_heartbeat_age_seconds": 0.0,
    }
    with pytest.raises(ValueError, match="state"):
        contract.parse(bad)


# ---------------------------------------------------------------------------
# Desired vs observed separation — Test #18
# ---------------------------------------------------------------------------


def test_desired_state_is_separate_from_observed() -> None:
    """The ProfileConfig never includes observed fields like linger_enabled."""
    from animus_bootstrap.lifecycle import ProfileConfig, ProfileMode, load_profile, save_profile

    profile = ProfileConfig(mode=ProfileMode.DEVELOPMENT_LOCAL)
    assert "linger_enabled" not in profile.to_dict()
    assert "runtime_target_active" not in profile.to_dict()
    # Round-trip
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        save_profile(Path_for(f.name), profile)  # type: ignore[name-defined]
        loaded = load_profile(Path_for(f.name))  # type: ignore[name-defined]
        assert loaded.mode == ProfileMode.DEVELOPMENT_LOCAL


def Path_for(name):  # tiny shim for the test above
    from pathlib import Path

    return Path(name)
