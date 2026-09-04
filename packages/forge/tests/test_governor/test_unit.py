"""Tests for pure-Python helpers (errors, paths, models, protocol).

No subprocess, no fixtures — just unit-level contract coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from animus_forge.governor.errors import (
    AlgNotFoundError,
    ContractIntegrityError,
    ContractRejectedError,
    GovernorAdapterError,
    GovernorError,
    GovernorTimeoutError,
    PermissionDeniedError,
    RunNotFoundError,
    RunStateInvalidError,
    RunUnusableError,
    VerifyDeniedError,
)
from animus_forge.governor.models import (
    CompatibilityKey,
    GovernorRun,
    MissionKey,
    RepositoryKey,
)
from animus_forge.governor.paths import (
    GOVERNOR_DIRNAME,
    RUNS_DIRNAME,
    find_active_run,
    run_dir,
    run_dir_or_raise,
    runs_root,
)
from animus_forge.governor.protocol import (
    CompletionDecision,
    RunLedger,
    WatchdogFinding,
    WatchdogReport,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_all_inherit_from_base() -> None:
    """One ``except GovernorAdapterError`` catches the whole module."""
    for cls in [
        AlgNotFoundError,
        GovernorError,
        ContractRejectedError,
        VerifyDeniedError,
        PermissionDeniedError,
        ContractIntegrityError,
        RunNotFoundError,
        RunStateInvalidError,
        RunUnusableError,
        GovernorTimeoutError,
    ]:
        assert issubclass(cls, GovernorAdapterError)


def test_governor_error_subcommand_default() -> None:
    """``GovernorError`` carries ``exit_code`` and ``subcommand``."""
    exc = GovernorError("boom", exit_code=1, subcommand="verify")
    assert exc.exit_code == 1
    assert exc.subcommand == "verify"
    assert exc.stderr == "boom"


def test_timeout_carries_timeout() -> None:
    exc = GovernorTimeoutError("slow", timeout=30.0)
    assert exc.timeout == 30.0


def test_base_default_message() -> None:
    """``GovernorAdapterError()`` with no message uses class name."""
    exc = GovernorAdapterError()
    assert "GovernorAdapterError" in str(exc)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_runs_root(tmp_path: Path) -> None:
    assert runs_root(tmp_path) == tmp_path.resolve() / GOVERNOR_DIRNAME


def test_run_dir_layout(tmp_path: Path) -> None:
    assert run_dir(tmp_path, "run-x") == runs_root(tmp_path) / RUNS_DIRNAME / "run-x"


def test_run_dir_or_raise_missing(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError):
        run_dir_or_raise(tmp_path, "missing")


def test_run_dir_or_raise_present(tmp_path: Path) -> None:
    target = tmp_path / ".animus-loop-governor" / "runs" / "run-y"
    target.mkdir(parents=True)
    assert run_dir_or_raise(tmp_path, "run-y") == target


def test_find_active_run_no_governor_dir(tmp_path: Path) -> None:
    assert find_active_run(tmp_path) is None


def test_find_active_run_no_runs_dir(tmp_path: Path) -> None:
    (tmp_path / ".animus-loop-governor").mkdir()
    assert find_active_run(tmp_path) is None


def test_find_active_run_empty_runs(tmp_path: Path) -> None:
    (tmp_path / ".animus-loop-governor" / "runs").mkdir(parents=True)
    assert find_active_run(tmp_path) is None


def test_find_active_run_returns_most_recent(tmp_path: Path, populate_runs_root) -> None:
    import time

    populate_runs_root("run-old")
    time.sleep(0.02)
    populate_runs_root("run-new")
    result = find_active_run(tmp_path)
    assert result is not None
    assert result.name == "run-new"


def test_find_active_run_ignores_files(tmp_path: Path) -> None:
    runs = tmp_path / ".animus-loop-governor" / "runs"
    runs.mkdir(parents=True)
    (runs / "stray.txt").write_text("noise")
    assert find_active_run(tmp_path) is None


# ---------------------------------------------------------------------------
# Protocol models (Pydantic mirrors)
# ---------------------------------------------------------------------------


def test_completion_decision_done_true() -> None:
    d = CompletionDecision(done=True, reasons=["ok"])
    assert d.done is True


def test_completion_decision_rejects_extra() -> None:
    with pytest.raises(ValidationError):
        CompletionDecision(done=True, reasons=[], bogus="x")


def test_completion_decision_requires_done_reasons() -> None:
    with pytest.raises(ValidationError):
        CompletionDecision(done=True)  # type: ignore[call-arg]


def test_watchdog_finding_score_bounded() -> None:
    with pytest.raises(ValidationError):
        WatchdogFinding(code="x", severity="info", message="m", score=1.5)


def test_watchdog_severity_literal() -> None:
    with pytest.raises(ValidationError):
        WatchdogFinding(code="x", severity="catastrophic", message="m")  # type: ignore[arg-type]


def test_watchdog_required_action_default_null() -> None:
    r = WatchdogReport(drift_score=0.1, stagnation=False)
    assert r.required_action is None


def test_run_ledger_minimal() -> None:
    ledger = RunLedger(run_id="r", task_id="t", contract_hash="c")
    assert ledger.phase == "contracted"


def test_run_ledger_phase_literal() -> None:
    with pytest.raises(ValidationError):
        RunLedger(run_id="r", task_id="t", contract_hash="c", phase="bogus")


# ---------------------------------------------------------------------------
# Adapter-side models
# ---------------------------------------------------------------------------


def test_repository_key_resolves_path() -> None:
    key = RepositoryKey(canonical_path="/tmp/repo")
    assert key.canonical_path == "/tmp/repo"
    assert key.remote_identity is None


def test_compatibility_key_default_policy_version() -> None:
    """``CompatibilityKey`` defaults ``policy_version`` to 1."""
    key = CompatibilityKey(
        repository=RepositoryKey(canonical_path="/tmp/r"),
        mission=MissionKey(mission_id="m-1"),
        adapter_version="0.1.0",
    )
    assert key.policy_version == 1


def test_governor_run_roundtrip() -> None:
    """Receipt round-trips through JSON losslessly."""
    run = GovernorRun(
        run_id="run-x",
        repository=Path("/tmp/repo"),
        contract_path=Path("/tmp/repo/contract.yaml"),
        started_at="2026-08-05T09:00:00+00:00",
        compatibility=CompatibilityKey(
            repository=RepositoryKey(canonical_path="/tmp/repo"),
            mission=MissionKey(mission_id="m-1"),
            adapter_version="0.1.0",
        ),
    )
    as_json = run.model_dump_json()
    parsed = GovernorRun.model_validate_json(as_json)
    assert parsed == run


def test_governor_run_extra_ignored() -> None:
    """Adapter receipts tolerate diagnostic extras (forward compat)."""
    run = GovernorRun(
        run_id="run-x",
        repository=Path("/tmp/repo"),
        contract_path=Path("/tmp/repo/contract.yaml"),
        started_at="2026-08-05T09:00:00+00:00",
        compatibility=CompatibilityKey(
            repository=RepositoryKey(canonical_path="/tmp/repo"),
            mission=MissionKey(mission_id="m-1"),
            adapter_version="0.1.0",
        ),
    )
    payload = json.loads(run.model_dump_json())
    payload["future_field"] = "ignored"
    GovernorRun.model_validate(payload)  # no raise


# ---------------------------------------------------------------------------
# Direct coverage for adapter module-level helpers
# ---------------------------------------------------------------------------


def test_persist_ledger_stub_skips_when_real_ledger_present(
    tmp_path: Path,
) -> None:
    """``_persist_ledger_stub`` is a no-op when a real ledger exists.

    The production ``alg start`` writes the ledger before the adapter
    persists the receipt. The stub then runs but must not clobber the
    production ledger.
    """
    from animus_forge.governor.adapter import _persist_ledger_stub
    from animus_forge.governor.protocol import RunLedger

    run_path = tmp_path / "runs" / "run-x"
    run_path.mkdir(parents=True)
    real = RunLedger(
        run_id="run-x",
        task_id="real-task",
        contract_hash="real-hash",
        phase="contracted",
    )
    real_path = run_path / "ledger.json"
    real_path.write_text(real.model_dump_json(), encoding="utf-8")

    _persist_ledger_stub(run_path, run_id="run-x")

    # The real ledger must be preserved verbatim.
    after = RunLedger.model_validate(
        __import__("json").loads(real_path.read_text(encoding="utf-8"))
    )
    assert after.task_id == "real-task"
    assert after.contract_hash == "real-hash"


def test_run_state_reader_watchdog_corrupt_raises(tmp_path: Path) -> None:
    """Corrupt ``watchdog-latest.json`` raises :class:`RunStateInvalidError`."""
    from animus_forge.governor.adapter import RunStateReader
    from animus_forge.governor.errors import RunStateInvalidError

    run_path = tmp_path / ".animus-loop-governor" / "runs" / "run-w"
    run_path.mkdir(parents=True)
    (run_path / "watchdog-latest.json").write_text("{not valid json", encoding="utf-8")
    reader = RunStateReader()
    with pytest.raises(RunStateInvalidError):
        reader.read_watchdog(tmp_path, "run-w")
