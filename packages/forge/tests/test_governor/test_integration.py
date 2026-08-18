"""End-to-end integration tests against a real ``alg`` binary.

Gated by the environment variable ``ANIMUS_LOOP_GOVERNOR_INTEGRATION=1``.
By default the suite is skipped so unit tests never accidentally hit
the real CLI.

Run with::

    ANIMUS_LOOP_GOVERNOR_INTEGRATION=1 \
        PYTHONPATH=src \
        pytest tests/test_governor/test_integration.py -v

These tests prove the adapter's external contract against the
real CLI, not just a test double. They cover:

* ``alg compile`` produces a contract accepted by ``alg start``.
* ``alg start`` creates the canonical run dir layout the adapter
  expects (ledger.json, events.jsonl, contract.yaml, contract.sha256).
* ``ensure_run`` reads the on-disk run state via the adapter's
  filesystem-hint step and returns a valid :class:`GovernorRun`.
* ``alg verify`` on an unsatisfied contract emits rc 3 and a
  ``completion-latest.json`` matching the adapter's
  :class:`CompletionDecision` Pydantic mirror.

Pre-requisites:

* ``alg`` on ``PATH`` (the wheel is installable from
  ``~/Downloads/animus_loop_governor-0.1.0-py3-none-any.whl``).
* Each smoke repo must be a git repo — the watchdog inspector runs
  ``git diff`` and crashes on non-git roots.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from animus_forge.governor import GovernorAdapter, GovernorClient
from animus_forge.governor.errors import VerifyDeniedError
from animus_forge.governor.models import GovernorRun

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    os.environ.get("ANIMUS_LOOP_GOVERNOR_INTEGRATION") != "1",
    reason="Set ANIMUS_LOOP_GOVERNOR_INTEGRATION=1 to run real-binary tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def alg_binary() -> str:
    """Locate the real ``alg`` binary on PATH."""
    binary = shutil.which("alg")
    if binary is None:
        pytest.skip("`alg` not on PATH; install animus_loop_governor wheel")
    return binary


@pytest.fixture()
def git_smoke_repo(tmp_path: Path) -> Iterator[Path]:
    """A real git-initialized repo at ``tmp_path``.

    The watchdog inspector in ``alg verify`` shells out to ``git diff``
    — a non-git root triggers an unhandled ``RuntimeError`` in the
    Governor. Initializing here makes the integration test independent
    of which directory pytest hands us.
    """
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Integration Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "--allow-empty", "-q", "-m", "init"],
        check=True,
    )
    yield tmp_path


@pytest.fixture()
def minimal_contract(tmp_path: Path) -> Path:
    """A normalized contract accepted by ``alg start``.

    Copy of the loop-governor's ``hangar-contract.yaml`` example; we
    use it because it is the only public, non-secret reference
    contract in the loop-governor repo.
    """
    repo = Path(
        os.environ.get(
            "ANIMUS_LOOP_GOVERNOR_REPO",
            str(Path.home() / "projects/animus-loop-governor"),
        )
    )
    src = repo / "examples" / "hangar-contract.yaml"
    if not src.is_file():
        pytest.skip(f"loop-governor example contract missing at {src}")
    dst = tmp_path / "contract.yaml"
    shutil.copyfile(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Compile + start
# ---------------------------------------------------------------------------


def test_alg_compile_produces_normalized_contract(
    alg_binary: str,
    minimal_contract: Path,
    tmp_path: Path,
) -> None:
    """``alg compile`` exits 0 and writes a valid normalized contract."""
    request_path = tmp_path / "request.md"
    request_path.write_text("# Integration test request\n", encoding="utf-8")
    output = tmp_path / "compiled.yaml"

    result = subprocess.run(
        [
            alg_binary,
            "compile",
            "--request",
            str(request_path),
            "--draft",
            str(minimal_contract),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file(), "compiled contract not written"
    # ``alg compile --output`` writes YAML, not JSON.
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert "contract_version" in data


def test_alg_start_creates_canonical_run_dir(
    alg_binary: str,
    minimal_contract: Path,
    git_smoke_repo: Path,
) -> None:
    """``alg start`` seals the contract and creates the run dir."""
    result = subprocess.run(
        [
            alg_binary,
            "start",
            "--contract",
            str(minimal_contract),
            "--root",
            str(git_smoke_repo),
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert result.returncode == 0, result.stderr

    # Use the adapter's parser — line 1 is ``Created run <id>``,
    # line 2 is the run dir path; the parser canonicalises both.
    from animus_forge.governor.client import _parse_run_id_from_start_stdout

    run_id = _parse_run_id_from_start_stdout(result.stdout)

    runs_root = git_smoke_repo / ".animus-loop-governor" / "runs" / run_id
    assert runs_root.is_dir(), f"run dir not created at {runs_root}"

    # Canonical layout.
    for filename in ("ledger.json", "events.jsonl", "contract.yaml", "contract.sha256"):
        assert (runs_root / filename).is_file(), f"missing {filename}"

    # Ledger parses as the adapter's Pydantic mirror.
    from animus_forge.governor.protocol import RunLedger

    ledger = RunLedger.model_validate_json((runs_root / "ledger.json").read_text(encoding="utf-8"))
    assert ledger.run_id == run_id
    assert ledger.phase == "contracted"


# ---------------------------------------------------------------------------
# ensure_run round-trip
# ---------------------------------------------------------------------------


def test_ensure_run_round_trip_with_real_cli(
    alg_binary: str,
    minimal_contract: Path,
    git_smoke_repo: Path,
) -> None:
    """``GovernorAdapter.ensure_run`` cooperates with a real ``alg start``."""
    client = GovernorClient(alg_binary=alg_binary)
    adapter = GovernorAdapter(client=client)

    receipt = adapter.ensure_run(
        repository=git_smoke_repo,
        mission_id="integration-mission-001",
        contract_path=minimal_contract,
    )
    assert isinstance(receipt, GovernorRun)
    assert receipt.repository == git_smoke_repo
    assert receipt.compatibility.mission.mission_id == "integration-mission-001"

    run_dir = git_smoke_repo / ".animus-loop-governor" / "runs" / receipt.run_id
    assert run_dir.is_dir()


def test_ensure_run_reuses_existing_run_id(
    alg_binary: str,
    minimal_contract: Path,
    git_smoke_repo: Path,
) -> None:
    """Restart reuses a known run id; no second ``alg start`` is invoked."""
    client = GovernorClient(alg_binary=alg_binary)
    adapter = GovernorAdapter(client=client)

    first = adapter.ensure_run(
        repository=git_smoke_repo,
        mission_id="integration-mission-002",
        contract_path=minimal_contract,
    )

    second = adapter.ensure_run(
        repository=git_smoke_repo,
        mission_id="integration-mission-002",
        contract_path=minimal_contract,
        known_run_id=first.run_id,
    )
    assert second.run_id == first.run_id
    # No second run dir appeared.
    runs_root = git_smoke_repo / ".animus-loop-governor" / "runs"
    assert sum(1 for _ in runs_root.iterdir()) == 1


# ---------------------------------------------------------------------------
# Verify denial path
# ---------------------------------------------------------------------------


def test_alg_verify_denied_raises_verify_denied(
    alg_binary: str,
    minimal_contract: Path,
    git_smoke_repo: Path,
) -> None:
    """``alg verify`` on an empty run returns rc 3 → ``VerifyDeniedError``."""
    # First create a run.
    start = subprocess.run(
        [
            alg_binary,
            "start",
            "--contract",
            str(minimal_contract),
            "--root",
            str(git_smoke_repo),
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=True,
    )
    from animus_forge.governor.client import _parse_run_id_from_start_stdout

    run_id = _parse_run_id_from_start_stdout(start.stdout)

    # Then verify — no evidence, expect denial.
    client = GovernorClient(alg_binary=alg_binary)
    with pytest.raises(VerifyDeniedError):
        client.verify(run_id, cwd=git_smoke_repo, timeout=30.0)

    # completion-latest.json must exist with done=false.
    completion_path = (
        git_smoke_repo / ".animus-loop-governor" / "runs" / run_id / "completion-latest.json"
    )
    assert completion_path.is_file()
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    assert payload["done"] is False
    assert payload["missing_evidence"], "missing_evidence must be populated"
