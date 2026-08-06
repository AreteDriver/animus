"""Scheduler integration tests for the governor adapter.

These tests verify the mission-level lifecycle contract:

* READY missions prepare before transitioning to RUNNING.
* Preparation failure prevents task dispatch.
* Successful preparation persists ``governor_run_id`` in mission
  metadata.
* Every task dispatched for the same mission sees the same run id.
* Restart reuses the persisted run id (no duplicate runs).
* Task retry does not create a new run.
* Separate missions do not accidentally share runs.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from animus_forge.governor import (
    GovernorAdapter,
    GovernorClient,
)
from animus_forge.governor.adapter import RunIdResolver
from animus_forge.missions.domain import Mission, MissionStatus
from animus_forge.missions.store import MissionLedger
from animus_forge.scheduler.mission_scheduler import (
    MissionScheduler,
    _MissionContractResolver,
)
from animus_forge.state.backends import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_backend() -> SQLiteBackend:
    backend = SQLiteBackend(":memory:")
    MissionLedger(backend)
    return backend


@pytest.fixture()
def ledger(memory_backend: SQLiteBackend) -> MissionLedger:
    return MissionLedger(memory_backend)


@pytest.fixture()
def ready_mission(ledger: MissionLedger, tmp_path: Path) -> Mission:
    """A READY mission tied to ``tmp_path`` as its repository."""
    mission = Mission(
        repository=str(tmp_path),
        objective="Build thing",
        risk_class="medium",
        status=MissionStatus.READY,
    )
    ledger.create_mission(mission)
    return mission


def _resolver_from_ledger(ledger: MissionLedger) -> RunIdResolver:
    """Build a RunIdResolver that reads from the mission ledger.

    Mirrors the production resolver the scheduler will wire in Step 3.
    """

    class _L(RunIdResolver):
        def lookup(self, mission_id: str | UUID) -> str | None:
            mid = mission_id if isinstance(mission_id, UUID) else UUID(str(mission_id))
            mission = ledger.get_mission(mid)
            if mission is None:
                return None
            receipt = mission.metadata.get("governor_run")
            if not receipt:
                return None
            return receipt.get("run_id")  # type: ignore[no-any-return]

    return _L()


# ---------------------------------------------------------------------------
# Adapter ↔ Mission store cooperation
# ---------------------------------------------------------------------------


def test_ensure_run_persists_receipt_in_mission_metadata(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
    tmp_path: Path,
) -> None:
    """After ``ensure_run`` the mission metadata carries a GovernorRun."""
    fake_client.set_response("start", "run-mission-1")
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )

    receipt = adapter.ensure_run(
        repository=Path(ready_mission.repository),
        mission_id=ready_mission.mission_id,
        contract_path=tmp_path / "contract.yaml",
    )

    # Persist the receipt to mission metadata (the scheduler will do
    # this in Step 3 inside the compare-and-swap; here we assert the
    # adapter's contract is sufficient for the scheduler to do so).
    mission = ledger.get_mission(ready_mission.mission_id)
    assert mission is not None
    mission.metadata["governor_run"] = receipt.model_dump(mode="json")
    ledger.update_mission(mission)

    reloaded = ledger.get_mission(ready_mission.mission_id)
    assert reloaded is not None
    assert reloaded.metadata["governor_run"]["run_id"] == "run-mission-1"


def test_persisted_receipt_reused_on_subsequent_ensure_run(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
    tmp_path: Path,
) -> None:
    """Restart / scheduler retry: persisted run id is reused, not replaced."""
    fake_client.set_response("start", "run-stable")
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )

    first = adapter.ensure_run(
        repository=Path(ready_mission.repository),
        mission_id=ready_mission.mission_id,
        contract_path=tmp_path / "contract.yaml",
    )
    mission = ledger.get_mission(ready_mission.mission_id)
    assert mission is not None
    mission.metadata["governor_run"] = first.model_dump(mode="json")
    ledger.update_mission(mission)

    # Second call — no fresh ``alg start`` should fire.
    fake_client.calls.clear()
    second = adapter.ensure_run(
        repository=Path(ready_mission.repository),
        mission_id=ready_mission.mission_id,
        contract_path=tmp_path / "contract.yaml",
    )
    assert second.run_id == first.run_id
    assert not fake_client.calls


def test_separate_missions_get_separate_runs(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
    tmp_path: Path,
) -> None:
    """Two missions on the same repo do not share a mutable run."""
    second = Mission(
        repository=ready_mission.repository,
        objective="Another task",
        status=MissionStatus.READY,
    )
    ledger.create_mission(second)

    fake_client.set_response("start", "run-m1")
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )
    receipt_1 = adapter.ensure_run(
        repository=Path(ready_mission.repository),
        mission_id=ready_mission.mission_id,
        contract_path=tmp_path / "contract.yaml",
    )

    # Persist the first receipt so the resolver returns it for m1
    # but not for m2.
    m1 = ledger.get_mission(ready_mission.mission_id)
    assert m1 is not None
    m1.metadata["governor_run"] = receipt_1.model_dump(mode="json")
    ledger.update_mission(m1)

    fake_client.calls.clear()
    fake_client.set_response("start", "run-m2")
    receipt_2 = adapter.ensure_run(
        repository=Path(second.repository),
        mission_id=second.mission_id,
        contract_path=tmp_path / "contract.yaml",
    )

    assert receipt_1.run_id == "run-m1"
    assert receipt_2.run_id == "run-m2"
    assert receipt_1.run_id != receipt_2.run_id


# ---------------------------------------------------------------------------
# Mission-status transition invariants
# ---------------------------------------------------------------------------


def test_ready_to_running_is_a_valid_transition() -> None:
    """``READY → RUNNING`` is allowed by the state machine."""
    from animus_forge.missions.transitions import ALLOWED_MISSION_TRANSITIONS

    assert MissionStatus.RUNNING in ALLOWED_MISSION_TRANSITIONS[
        MissionStatus.READY
    ]


def test_failed_is_terminal_no_implicit_recovery() -> None:
    """``FAILED`` has no outgoing transitions — no silent retry."""
    from animus_forge.missions.transitions import ALLOWED_MISSION_TRANSITIONS

    assert ALLOWED_MISSION_TRANSITIONS[MissionStatus.FAILED] == set()


def test_completed_is_terminal() -> None:
    """``COMPLETED`` has no outgoing transitions."""
    from animus_forge.missions.transitions import ALLOWED_MISSION_TRANSITIONS

    assert ALLOWED_MISSION_TRANSITIONS[MissionStatus.COMPLETED] == set()


def test_preparation_failure_keeps_mission_runnable(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
    tmp_path: Path,
) -> None:
    """If ``alg start`` raises, the mission stays in READY (not RUNNING).

    The scheduler can retry on the next tick. This is the user's
    "remain READY or enter BLOCKED" option — staying READY is the
    supported default since the enum has no BLOCKED status.
    """
    from animus_forge.governor.errors import ContractRejectedError

    fake_client.set_error(
        "start", ContractRejectedError("bad contract", exit_code=2)
    )
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )

    with pytest.raises(ContractRejectedError):
        adapter.ensure_run(
            repository=Path(ready_mission.repository),
            mission_id=ready_mission.mission_id,
            contract_path=tmp_path / "contract.yaml",
        )

    # Mission has not been transitioned; the scheduler's
    # compare-and-swap ``persist_run_and_start`` is what would have
    # moved it to RUNNING, but the adapter raised first.
    current = ledger.get_mission(ready_mission.mission_id)
    assert current is not None
    assert current.status == MissionStatus.READY
    assert "governor_run" not in current.metadata

# ---------------------------------------------------------------------------
# MissionScheduler._start_ready_mission — READY → RUNNING gating
# ---------------------------------------------------------------------------


class _StubPool:
    """Minimal stand-in for CitizenWorkerPool for _start_ready_mission tests.

    The scheduler's read-only properties are exercised in the broader
    scheduler tests; here we only need the scheduler to construct
    without raising and the lifecycle methods to no-op.
    """

    async def run_recovery_loop(self) -> None:  # pragma: no cover - unused
        return None


class _StubLease:
    """Minimal stand-in for LeaseManager."""


class _StubCost:
    """Minimal stand-in for CostEnforcer."""

    def global_spend(self) -> Decimal:  # pragma: no cover - unused
        return Decimal("0")


def _build_scheduler(
    ledger: MissionLedger,
    *,
    governor_adapter: GovernorAdapter | None,
    contract_resolver: Any | None = None,
) -> MissionScheduler:
    """Build a MissionScheduler that exercises only the governor path.

    Other collaborators are stubbed because ``_start_ready_mission``
    only touches the ledger and the governor adapter. Recovery is
    disabled so the stub pool never has to register a recovery loop.
    """
    from animus_forge.scheduler.mission_scheduler import SchedulerConfig

    return MissionScheduler(
        ledger=ledger,
        lease_manager=_StubLease(),  # type: ignore[arg-type]
        worker_pool=_StubPool(),  # type: ignore[arg-type]
        cost_enforcer=_StubCost(),  # type: ignore[arg-type]
        governor_adapter=governor_adapter,
        contract_resolver=contract_resolver,
        config=SchedulerConfig(enable_recovery=False),
    )


@pytest.mark.asyncio
async def test_start_ready_mission_promotes_after_ensure_run(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
    tmp_path: Path,
) -> None:
    """A READY mission transitions to RUNNING after ``ensure_run``."""
    fake_client.set_response("start", "run-mission-1")
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )
    scheduler = _build_scheduler(ledger, governor_adapter=adapter)

    # Write a contract so the resolver is satisfied.
    contract = tmp_path / "contract.yaml"
    contract.write_text("requirements: []\n")
    ready_mission.metadata["contract_path"] = str(contract)
    ledger.update_mission(ready_mission)

    await scheduler._start_ready_mission()

    reloaded = ledger.get_mission(ready_mission.mission_id)
    assert reloaded is not None
    assert reloaded.status == MissionStatus.RUNNING
    assert reloaded.metadata["governor_run"]["run_id"] == "run-mission-1"


@pytest.mark.asyncio
async def test_start_ready_mission_keeps_ready_on_adapter_failure(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
    tmp_path: Path,
) -> None:
    """``ensure_run`` raises → mission stays READY for the next tick."""
    from animus_forge.governor.errors import ContractRejectedError

    fake_client.set_error(
        "start", ContractRejectedError("bad", exit_code=2)
    )
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )
    scheduler = _build_scheduler(ledger, governor_adapter=adapter)

    contract = tmp_path / "contract.yaml"
    contract.write_text("requirements: []\n")
    ready_mission.metadata["contract_path"] = str(contract)
    ledger.update_mission(ready_mission)

    await scheduler._start_ready_mission()

    reloaded = ledger.get_mission(ready_mission.mission_id)
    assert reloaded is not None
    assert reloaded.status == MissionStatus.READY
    assert "governor_run" not in reloaded.metadata


@pytest.mark.asyncio
async def test_start_ready_mission_no_adapter_uses_legacy_path(
    ledger: MissionLedger,
    ready_mission: Mission,
) -> None:
    """No adapter wired → legacy path: promote directly to RUNNING."""
    scheduler = _build_scheduler(ledger, governor_adapter=None)

    await scheduler._start_ready_mission()

    reloaded = ledger.get_mission(ready_mission.mission_id)
    assert reloaded is not None
    assert reloaded.status == MissionStatus.RUNNING


@pytest.mark.asyncio
async def test_start_ready_mission_missing_contract_stays_ready(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
) -> None:
    """No contract path and no in-repo default → mission stays READY."""
    fake_client.set_response("start", "run-mission-1")
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )
    scheduler = _build_scheduler(ledger, governor_adapter=adapter)

    await scheduler._start_ready_mission()

    reloaded = ledger.get_mission(ready_mission.mission_id)
    assert reloaded is not None
    assert reloaded.status == MissionStatus.READY
    # No ``alg start`` was invoked.
    assert not fake_client.calls


@pytest.mark.asyncio
async def test_start_ready_mission_uses_resolver_when_no_explicit_path(
    ledger: MissionLedger,
    ready_mission: Mission,
    fake_client: GovernorClient,
    tmp_path: Path,
) -> None:
    """The default resolver picks up ``<repo>/.animus-loop-governor/contract.yaml``."""
    fake_client.set_response("start", "run-default-contract")
    adapter = GovernorAdapter(
        client=fake_client,
        run_id_resolver=_resolver_from_ledger(ledger),
    )
    scheduler = _build_scheduler(ledger, governor_adapter=adapter)

    # Write the in-repo default contract.
    default = (
        Path(ready_mission.repository)
        / ".animus-loop-governor"
        / "contract.yaml"
    )
    default.parent.mkdir(parents=True, exist_ok=True)
    default.write_text("requirements: []\n")

    await scheduler._start_ready_mission()

    reloaded = ledger.get_mission(ready_mission.mission_id)
    assert reloaded is not None
    assert reloaded.status == MissionStatus.RUNNING
    assert reloaded.metadata["governor_run"]["run_id"] == "run-default-contract"


def test_contract_resolver_explicit_metadata_wins(tmp_path: Path) -> None:
    """Explicit ``mission.metadata["contract_path"]`` overrides default."""
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("x: 1\n")
    mission = Mission(
        repository=str(tmp_path),
        objective="t",
        metadata={"contract_path": str(explicit)},
    )
    resolver = _MissionContractResolver()
    assert resolver.resolve(mission, tmp_path) == explicit


def test_contract_resolver_falls_back_to_in_repo_default(tmp_path: Path) -> None:
    """With no override, the in-repo default wins."""
    default = tmp_path / ".animus-loop-governor" / "contract.yaml"
    default.parent.mkdir(parents=True)
    default.write_text("x: 1\n")
    mission = Mission(repository=str(tmp_path), objective="t")
    resolver = _MissionContractResolver()
    assert resolver.resolve(mission, tmp_path) == default


def test_contract_resolver_returns_none_when_no_contract(tmp_path: Path) -> None:
    """No explicit path, no in-repo default → ``None`` (caller fails)."""
    mission = Mission(repository=str(tmp_path), objective="t")
    resolver = _MissionContractResolver()
    assert resolver.resolve(mission, tmp_path) is None
