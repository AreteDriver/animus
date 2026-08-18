"""Tests for :class:`GovernorAdapter` run-resolution algorithm.

Covers the strict idempotent resolution order:

1. Known run id from mission metadata → validate, reuse if valid.
2. Hint from filesystem → validate, reuse if valid.
3. Otherwise: ``alg start`` → persist receipt, return new run.

Plus the negative cases: stale runs, cross-repo runs, partially
initialised runs, concurrent races.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from animus_forge.governor import (
    GovernorAdapter,
    GovernorClient,
)
from animus_forge.governor.adapter import compute_compatibility_key
from animus_forge.governor.errors import (
    AlgNotFoundError,
    RunUnusableError,
)
from animus_forge.governor.models import CompatibilityKey, MissionKey


def _compat(
    repository: Path,
    mission_id: str = "mission-001",
    *,
    revision: str | None = None,
) -> CompatibilityKey:
    """Build a compatibility key for the given repository."""
    return compute_compatibility_key(
        repository=repository, mission_id=mission_id, revision=revision
    )


def _populate_ledger(run_path: Path, phase: str = "implementation") -> None:
    """Write a minimal valid ledger to ``run_path``."""
    payload = (
        '{"run_id": "' + run_path.name + '", "task_id": "t-1", '
        '"contract_hash": "h-1", "phase": "' + phase + '"}'
    )
    (run_path / "ledger.json").write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Step 1: known_run_id reuse
# ---------------------------------------------------------------------------


def test_known_run_id_valid_is_reused(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """A valid known id short-circuits the filesystem search and start."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-known"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)
    write_receipt(run_path, mission_id="mission-001")

    adapter = GovernorAdapter(client=fake_client)
    receipt = adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
        known_run_id="run-known",
    )
    assert receipt.run_id == "run-known"
    assert not fake_client.calls, "alg start must not be invoked"


def test_known_run_id_wrong_mission_rejected(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """A receipt that points to a different mission cannot be reused."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-mismatch"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)
    write_receipt(run_path, mission_id="other-mission")

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunUnusableError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-001",
            contract_path=tmp_path / "contract.yaml",
            known_run_id="run-mismatch",
        )


def test_known_run_id_other_repository_rejected(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """A receipt for a different repository path cannot be reused."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-cross-repo"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)
    write_receipt(run_path, repository_path="/some/other/path")

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunUnusableError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-001",
            contract_path=tmp_path / "contract.yaml",
            known_run_id="run-cross-repo",
        )


def test_known_run_id_stale_terminal_rejected(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """A terminal-phase ledger is rejected outright."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-stale"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path, phase="failed")
    write_receipt(run_path)

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunUnusableError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-001",
            contract_path=tmp_path / "contract.yaml",
            known_run_id="run-stale",
        )


def test_known_run_id_partially_initialised_rejected(
    tmp_path: Path, fake_client: GovernorClient
) -> None:
    """Run dir exists, ledger exists, but no receipt → ``RunUnusable``."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-partial"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunUnusableError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-001",
            contract_path=tmp_path / "contract.yaml",
            known_run_id="run-partial",
        )


def test_known_run_id_missing_dir_falls_through(
    tmp_path: Path, fake_client: GovernorClient
) -> None:
    """Known id that doesn't exist → fall through to Step 3 (``alg start``).

    Steps 1 and 2 see nothing; the fake client returns the new id;
    ``_persist_receipt`` creates the run dir + writes the receipt.
    """
    fake_client.set_response("start", "run-created")
    adapter = GovernorAdapter(client=fake_client)
    receipt = adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
        known_run_id="run-vanished",
    )
    assert receipt.run_id == "run-created"
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-created"
    assert (run_path / "adapter-receipt.json").is_file()


# ---------------------------------------------------------------------------
# Step 2: filesystem hint
# ---------------------------------------------------------------------------


def test_filesystem_hint_compatible_reused(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """A compatible on-disk run is found via ``find_active_run``."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-hint"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)
    write_receipt(run_path)

    adapter = GovernorAdapter(client=fake_client)
    receipt = adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
    )
    assert receipt.run_id == "run-hint"
    assert not fake_client.calls


def test_filesystem_hint_terminated_rejected(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """Hinted terminal-phase run is rejected, not silently reused."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-stale-hint"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path, phase="aborted")
    write_receipt(run_path)

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunUnusableError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-001",
            contract_path=tmp_path / "contract.yaml",
        )


# ---------------------------------------------------------------------------
# Step 3: create new run
# ---------------------------------------------------------------------------


def test_no_existing_run_invokes_start(tmp_path: Path, fake_client: GovernorClient) -> None:
    """No hint, no known id → ``alg start`` is called once."""
    fake_client.set_response("start", "run-newly-created")
    adapter = GovernorAdapter(client=fake_client)
    receipt = adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
    )
    assert receipt.run_id == "run-newly-created"
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0].method == "start"


def test_new_run_persists_receipt(tmp_path: Path, fake_client: GovernorClient) -> None:
    """A freshly started run has its receipt written to disk."""
    fake_client.set_response("start", "run-persisted")
    adapter = GovernorAdapter(client=fake_client)
    adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
    )
    receipt_file = (
        tmp_path / ".animus-loop-governor" / "runs" / "run-persisted" / "adapter-receipt.json"
    )
    assert receipt_file.is_file()


def test_restart_reuses_persisted_run(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """Simulate process restart: known_run_id from ledger is reused."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-restart"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)
    write_receipt(run_path)

    adapter = GovernorAdapter(client=fake_client)
    receipt = adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
        known_run_id="run-restart",
    )
    assert receipt.run_id == "run-restart"
    assert not fake_client.calls


def test_concurrent_callers_dedup_via_resolver(tmp_path: Path, fake_client: GovernorClient) -> None:
    """A resolver that returns a stable id forces every caller to reuse it.

    Simulates two scheduler workers picking up the same mission — the
    second caller must observe the first's persisted run id and skip
    ``alg start``.
    """
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-shared"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)

    # Write a receipt that matches the request — the resolver simply
    # returns its name; ``_validate_or_raise`` reuses it.
    from animus_forge.governor.adapter import (
        RunIdResolver,
        _persist_receipt,
    )
    from animus_forge.governor.models import GovernorRun

    receipt = GovernorRun(
        run_id="run-shared",
        repository=tmp_path,
        contract_path=tmp_path / "contract.yaml",
        started_at="2026-08-05T09:00:00+00:00",
        compatibility=compute_compatibility_key(repository=tmp_path, mission_id="mission-001"),
    )
    _persist_receipt(run_path, receipt)

    class _Resolver(RunIdResolver):
        def __init__(self) -> None:
            self._seen: list[str] = []

        def lookup(self, mission_id: str) -> str | None:  # noqa: ARG002
            self._seen.append("run-shared")
            return "run-shared"

    resolver = _Resolver()
    adapter = GovernorAdapter(client=fake_client, run_id_resolver=resolver)

    first = adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
    )
    second = adapter.ensure_run(
        repository=tmp_path,
        mission_id="mission-001",
        contract_path=tmp_path / "contract.yaml",
    )

    assert first.run_id == "run-shared"
    assert second.run_id == "run-shared"
    assert not fake_client.calls  # neither call invoked alg start
    assert resolver._seen == ["run-shared", "run-shared"]


def test_separate_missions_dont_share_runs(
    tmp_path: Path, fake_client: GovernorClient, write_receipt: Callable
) -> None:
    """A run for mission-A cannot be reused for mission-B."""
    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-A"
    run_path.mkdir(parents=True)
    _populate_ledger(run_path)
    write_receipt(run_path, mission_id="mission-A")

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunUnusableError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-B",
            contract_path=tmp_path / "contract.yaml",
            known_run_id="run-A",
        )


# ---------------------------------------------------------------------------
# compute_compatibility_key
# ---------------------------------------------------------------------------


def test_compute_compatibility_key_includes_canonical_path(
    tmp_path: Path,
) -> None:
    """Canonical path is resolved; non-canonical inputs are normalised."""
    key = compute_compatibility_key(repository=tmp_path, mission_id="m-1")
    assert key.repository.canonical_path == str(tmp_path.resolve())
    assert key.mission.mission_id == "m-1"
    assert key.adapter_version != ""


def test_compute_compatibility_key_rejects_degenerate_inputs() -> None:
    """Mission id must be a non-empty string."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        compute_compatibility_key(repository=Path("/tmp"), mission_id="")


# ---------------------------------------------------------------------------
# Failure: alg not installed
# ---------------------------------------------------------------------------


def test_missing_alg_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``alg`` not on PATH → :class:`AlgNotFoundError` from ensure_run."""
    monkeypatch.setenv("PATH", "")
    client = GovernorClient(alg_binary=None)
    adapter = GovernorAdapter(client=client)
    with pytest.raises(AlgNotFoundError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-001",
            contract_path=tmp_path / "contract.yaml",
        )


# ---------------------------------------------------------------------------
# Coverage-closing tests: corrupt ledger / corrupt receipt / known run
# with no ledger.
# ---------------------------------------------------------------------------


def test_known_run_id_with_corrupt_ledger_rejected(
    tmp_path: Path, fake_client: GovernorClient
) -> None:
    """A known run whose ledger is corrupt JSON is rejected loudly."""
    from animus_forge.governor.errors import RunStateInvalidError
    from animus_forge.governor.models import CompatibilityKey

    run_id = "run-corrupt-ledger"
    runs = tmp_path / ".animus-loop-governor" / "runs" / run_id
    runs.mkdir(parents=True)
    (runs / "ledger.json").write_text("{not valid json", encoding="utf-8")

    # Receipt also written so the failure lands on ledger parse.
    from animus_forge.governor.models import GovernorRun
    from animus_forge.governor.models import RepositoryKey as RepositoryKeyModel

    compat = CompatibilityKey(
        repository=RepositoryKeyModel(canonical_path=str(tmp_path)),
        mission=MissionKey(mission_id="mission-x"),
        policy_version=1,
        adapter_version="0.1.0",
    )
    receipt = GovernorRun(
        run_id=run_id,
        repository=tmp_path,
        contract_path=tmp_path / "contract.yaml",
        started_at="2026-08-05T10:00:00+00:00",
        compatibility=compat,
    )
    (runs / "adapter-receipt.json").write_text(receipt.model_dump_json(), encoding="utf-8")

    from animus_forge.governor.adapter import GovernorAdapter

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunStateInvalidError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-x",
            contract_path=tmp_path / "contract.yaml",
            known_run_id=run_id,
        )


def test_known_run_id_with_no_ledger_rejected(tmp_path: Path, fake_client: GovernorClient) -> None:
    """A known run that exists but has no parseable ledger is rejected."""
    from animus_forge.governor.errors import RunUnusableError
    from animus_forge.governor.models import (
        CompatibilityKey,
        GovernorRun,
    )
    from animus_forge.governor.models import (
        RepositoryKey as RepositoryKeyModel,
    )

    run_id = "run-no-ledger"
    runs = tmp_path / ".animus-loop-governor" / "runs" / run_id
    runs.mkdir(parents=True)
    # No ledger.json — exercise the "no parseable ledger" branch.
    compat = CompatibilityKey(
        repository=RepositoryKeyModel(canonical_path=str(tmp_path)),
        mission=MissionKey(mission_id="mission-y"),
        policy_version=1,
        adapter_version="0.1.0",
    )
    receipt = GovernorRun(
        run_id=run_id,
        repository=tmp_path,
        contract_path=tmp_path / "contract.yaml",
        started_at="2026-08-05T10:01:00+00:00",
        compatibility=compat,
    )
    (runs / "adapter-receipt.json").write_text(receipt.model_dump_json(), encoding="utf-8")

    from animus_forge.governor.adapter import GovernorAdapter

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunUnusableError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-y",
            contract_path=tmp_path / "contract.yaml",
            known_run_id=run_id,
        )


def test_known_run_id_corrupt_receipt_rejected(tmp_path: Path, fake_client: GovernorClient) -> None:
    """A known run with a corrupt ``adapter-receipt.json`` is rejected."""
    from animus_forge.governor.errors import RunStateInvalidError

    run_id = "run-corrupt-receipt"
    runs = tmp_path / ".animus-loop-governor" / "runs" / run_id
    runs.mkdir(parents=True)
    # Valid ledger so the receipt-parse branch is exercised.
    (runs / "ledger.json").write_text(
        '{"run_id":"' + run_id + '","task_id":"t","contract_hash":"h","phase":"contracted"}',
        encoding="utf-8",
    )
    (runs / "adapter-receipt.json").write_text("{not json", encoding="utf-8")

    from animus_forge.governor.adapter import GovernorAdapter

    adapter = GovernorAdapter(client=fake_client)
    with pytest.raises(RunStateInvalidError):
        adapter.ensure_run(
            repository=tmp_path,
            mission_id="mission-z",
            contract_path=tmp_path / "contract.yaml",
            known_run_id=run_id,
        )
