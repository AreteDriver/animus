"""Tests for :class:`GovernorVerifierCitizen`.

Covers the verifier's translation of ``alg verify`` outcomes into
:class:`CitizenOutput`:
* rc 0 + clean watchdog → ``completed``
* rc 0 + watchdog ``required_action`` → ``needs_repair``
* rc 3 (``VerifyDeniedError``) → ``needs_repair`` with ``missing_evidence``
* infrastructure failures → ``failed``
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from animus_forge.governor import GovernorClient, GovernorVerifierCitizen
from animus_forge.governor.errors import (
    AlgNotFoundError,
    GovernorError,
    GovernorTimeoutError,
    VerifyDeniedError,
)
from animus_forge.missions.domain import Task, TaskContext


def _task(mission_id: str = "m-1") -> Task:
    return Task(
        task_id=uuid4(),
        mission_id=uuid4(),
        citizen_role="loop_governor",
        description="Verify mission completion",
        metadata={"mission_id_text": mission_id},
    )


def _context(
    repository: Path | None, *, governor_run_id: str | None = None
) -> TaskContext:
    extras: dict[str, object] = {}
    if governor_run_id is not None:
        extras["governor_run_id"] = governor_run_id
    ctx = TaskContext(
        mission_objective="complete the thing",
        task_description="verify",
        repository=str(repository) if repository else "",
    )
    if extras:
        # ``TaskContext`` uses ``extra='forbid'``; the citizen reads
        # governor_run_id from a side-channel so tests inject via
        # direct attribute.
        object.__setattr__(ctx, "_extras", extras)  # type: ignore[attr-defined]
    return ctx


# ---------------------------------------------------------------------------
# Missing inputs
# ---------------------------------------------------------------------------


def test_missing_repository_returns_failed(
    fake_client: GovernorClient,
) -> None:
    citizen = GovernorVerifierCitizen(client=fake_client)
    task = _task()
    context = _context(repository=None, governor_run_id="run-x")
    output = citizen.run(task, context)
    assert output.status == "failed"
    assert output.confidence == 0.0
    assert any(r.get("type") == "no_repository" for r in output.risks)


def test_missing_run_id_returns_failed(fake_client: GovernorClient) -> None:
    citizen = GovernorVerifierCitizen(client=fake_client)
    task = _task()
    context = _context(repository=Path("/tmp"), governor_run_id=None)
    output = citizen.run(task, context)
    assert output.status == "failed"
    assert any(r.get("type") == "no_governor_run" for r in output.risks)


# ---------------------------------------------------------------------------
# Approval path
# ---------------------------------------------------------------------------


def test_verify_approved_returns_completed(
    tmp_path: Path,
    fake_client: GovernorClient,
    populate_runs_root: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rc 0 + no required_action → ``status='completed'``."""
    populate_runs_root("run-x", files={})
    # ``populate_runs_root`` returns ``tmp_path``; the actual run dir
    # is at ``tmp_path / .animus-loop-governor / runs / run-x``.
    run_dir_path = (
        tmp_path / ".animus-loop-governor" / "runs" / "run-x"
    )
    from shutil import copyfile

    copyfile(
        Path(__file__).parent / "fixtures/runs/run-approve/watchdog-latest.json",
        run_dir_path / "watchdog-latest.json",
    )
    citizen = GovernorVerifierCitizen(client=fake_client)
    task = _task()
    context = _context(repository=tmp_path, governor_run_id="run-x")

    monkeypatch.setattr(
        "animus_forge.governor.adapter._resolve_run_id_for_task",
        lambda ctx: "run-x",
    )

    output = citizen.run(task, context)
    assert output.status == "completed"
    assert output.confidence == 1.0
    assert fake_client.calls and fake_client.calls[0].method == "verify"


def test_verify_approved_with_required_action_returns_needs_repair(
    tmp_path: Path,
    fake_client: GovernorClient,
    populate_runs_root: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rc 0 + watchdog ``required_action`` → ``status='needs_repair'``."""
    populate_runs_root("run-w")
    run_dir_path = (
        tmp_path / ".animus-loop-governor" / "runs" / "run-w"
    )
    from shutil import copyfile

    copyfile(
        Path(__file__).parent
        / "fixtures/runs/run-watchdog-halt/watchdog-latest.json",
        run_dir_path / "watchdog-latest.json",
    )

    citizen = GovernorVerifierCitizen(client=fake_client)
    task = _task()
    context = _context(repository=tmp_path, governor_run_id="run-w")
    monkeypatch.setattr(
        "animus_forge.governor.adapter._resolve_run_id_for_task",
        lambda ctx: "run-w",
    )

    output = citizen.run(task, context)
    assert output.status == "needs_repair"
    assert output.follow_up_tasks
    assert any(
        r.get("type") == "watchdog" for r in output.risks
    )


# ---------------------------------------------------------------------------
# Denial path
# ---------------------------------------------------------------------------


def test_verify_denied_returns_needs_repair(
    tmp_path: Path,
    fake_client: GovernorClient,
    populate_runs_root: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rc 3 (denial) → ``status='needs_repair'`` with explicit reasons."""
    populate_runs_root("run-deny")
    run_dir_path = (
        tmp_path / ".animus-loop-governor" / "runs" / "run-deny"
    )
    from shutil import copyfile

    copyfile(
        Path(__file__).parent / "fixtures/runs/run-deny/completion-latest.json",
        run_dir_path / "completion-latest.json",
    )

    fake_client.set_error(
        "verify",
        VerifyDeniedError("denied", exit_code=3),
    )

    citizen = GovernorVerifierCitizen(client=fake_client)
    task = _task()
    context = _context(repository=tmp_path, governor_run_id="run-deny")
    monkeypatch.setattr(
        "animus_forge.governor.adapter._resolve_run_id_for_task",
        lambda ctx: "run-deny",
    )

    output = citizen.run(task, context)
    assert output.status == "needs_repair"
    assert output.confidence == 1.0
    # The repair tasks come from completion-latest.json's
    # missing_evidence + blocking_findings.
    assert any("cargo clippy" in t for t in output.follow_up_tasks)
    assert any("cargo test" in t for t in output.follow_up_tasks)


# ---------------------------------------------------------------------------
# Infrastructure failures
# ---------------------------------------------------------------------------


def test_alg_missing_returns_failed(
    tmp_path: Path,
    fake_client: GovernorClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AlgNotFoundError`` → ``status='failed'`` with diagnostic risk."""
    fake_client.set_error("verify", AlgNotFoundError("no alg"))
    monkeypatch.setattr(
        "animus_forge.governor.adapter._resolve_run_id_for_task",
        lambda ctx: "run-x",
    )
    citizen = GovernorVerifierCitizen(client=fake_client)
    output = citizen.run(_task(), _context(repository=tmp_path))
    assert output.status == "failed"
    assert any(
        r.get("type") == "governor_error" for r in output.risks
    )


def test_timeout_returns_failed(
    tmp_path: Path,
    fake_client: GovernorClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GovernorTimeoutError`` → ``status='failed'``."""
    fake_client.set_error(
        "verify", GovernorTimeoutError("slow", timeout=30.0)
    )
    monkeypatch.setattr(
        "animus_forge.governor.adapter._resolve_run_id_for_task",
        lambda ctx: "run-x",
    )
    citizen = GovernorVerifierCitizen(client=fake_client)
    output = citizen.run(_task(), _context(repository=tmp_path))
    assert output.status == "failed"


def test_unexpected_governor_error_returns_failed(
    tmp_path: Path,
    fake_client: GovernorClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic :class:`GovernorError` → ``status='failed'``."""
    fake_client.set_error(
        "verify", GovernorError("oops", exit_code=99, subcommand="verify")
    )
    monkeypatch.setattr(
        "animus_forge.governor.adapter._resolve_run_id_for_task",
        lambda ctx: "run-x",
    )
    citizen = GovernorVerifierCitizen(client=fake_client)
    output = citizen.run(_task(), _context(repository=tmp_path))
    assert output.status == "failed"


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------


def test_citizen_role_and_capabilities() -> None:
    """``GovernorVerifierCitizen`` declares correct Forge identity."""
    citizen = GovernorVerifierCitizen()
    assert citizen.role == "loop_governor"
    assert citizen.can_modify_code is False
    assert citizen.can_approve is False
    assert "verify" in citizen.capabilities

# ---------------------------------------------------------------------------
# RunStateReader direct coverage (push adapter coverage above 97%)
# ---------------------------------------------------------------------------


def test_run_state_reader_read_completion(
    tmp_path: Path, populate_runs_root: Callable
) -> None:
    """``read_completion`` parses a valid ``completion-latest.json``."""
    from animus_forge.governor.adapter import RunStateReader
    from animus_forge.governor.protocol import CompletionDecision

    populate_runs_root(
        "run-c",
        files={
            "completion-latest.json": json.dumps(
                {
                    "done": True,
                    "reasons": ["all evidence captured"],
                    "missing_evidence": [],
                    "blocking_findings": [],
                }
            )
        },
    )
    reader = RunStateReader()
    decision = reader.read_completion(tmp_path, "run-c")
    assert isinstance(decision, CompletionDecision)
    assert decision.done is True
    assert "all evidence captured" in decision.reasons


def test_run_state_reader_read_completion_missing_file(tmp_path: Path) -> None:
    """Missing ``completion-latest.json`` raises :class:`RunStateInvalidError`."""
    from animus_forge.governor.adapter import RunStateReader
    from animus_forge.governor.errors import RunStateInvalidError

    reader = RunStateReader()
    with pytest.raises(RunStateInvalidError):
        reader.read_completion(tmp_path, "missing")


def test_run_state_reader_read_watchdog_missing_returns_none(
    tmp_path: Path,
) -> None:
    """Missing ``watchdog-latest.json`` returns ``None`` (not an error)."""
    from animus_forge.governor.adapter import RunStateReader

    reader = RunStateReader()
    assert reader.read_watchdog(tmp_path, "anything") is None


def test_run_state_reader_read_completion_corrupt_raises(
    tmp_path: Path, populate_runs_root: Callable
) -> None:
    """Corrupt JSON in ``completion-latest.json`` raises."""
    from animus_forge.governor.adapter import RunStateReader
    from animus_forge.governor.errors import RunStateInvalidError

    populate_runs_root(
        "run-corrupt", files={"completion-latest.json": "{not json"}
    )
    reader = RunStateReader()
    with pytest.raises(RunStateInvalidError):
        reader.read_completion(tmp_path, "run-corrupt")
