"""Shared fixtures for the governor adapter tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from animus_forge.governor.client import GovernorClient
from animus_forge.governor.models import CompatibilityKey, GovernorRun

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RUNS_FIXTURES = FIXTURES_DIR / "runs"


@dataclass
class CallRecord:
    """Single recorded invocation of a fake client method."""

    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeGovernorClient(GovernorClient):
    """Records calls and returns canned values; no subprocess.

    Tests configure responses via :meth:`set_response` and
    :meth:`set_error`. ``binary`` is bypassed so the fake never
    invokes :func:`shutil.which`.
    """

    def __init__(self) -> None:  # noqa: D401 — test double
        self.calls: list[CallRecord] = []
        self.responses: dict[str, Any] = {}
        self.errors: dict[str, BaseException] = {}

    def set_response(self, method: str, value: Any) -> None:
        self.responses[method] = value

    def set_error(self, method: str, error: BaseException) -> None:
        self.errors[method] = error

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(CallRecord(method=method, args=args, kwargs=kwargs))

    @property
    def binary(self) -> str:
        return "/fake/alg"

    def compile(  # noqa: ARG002 — test double ignores shapes
        self,
        request: Path,
        draft: Path,
        output: Path,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> Path:
        self._record("compile", request, draft, output, cwd, timeout)
        if "compile" in self.errors:
            raise self.errors["compile"]
        return self.responses.get("compile", output)

    def start(  # noqa: ARG002
        self,
        contract_path: Path,
        *,
        cwd: Path,
        run_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        self._record("start", contract_path, cwd, run_id, timeout)
        if "start" in self.errors:
            raise self.errors["start"]
        return self.responses.get("start", "run-fake-001")

    def verify(  # noqa: ARG002
        self,
        run_id: str,
        *,
        cwd: Path,
        timeout: float | None = None,
    ) -> None:
        self._record("verify", run_id, cwd, timeout)
        if "verify" in self.errors:
            raise self.errors["verify"]
        return self.responses.get("verify")


@pytest.fixture
def fake_client() -> FakeGovernorClient:
    """Fresh :class:`FakeGovernorClient` for each test."""
    return FakeGovernorClient()


@pytest.fixture(autouse=True)
def _isolate_alg_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ``PATH`` so unit tests cannot accidentally invoke ``alg``."""
    if os.environ.get("ANIMUS_LOOP_GOVERNOR_INTEGRATION") == "1":
        return
    monkeypatch.setenv("PATH", "")


@pytest.fixture
def fixture_run_dir() -> Callable[[str], Path]:
    def factory(name: str) -> Path:
        path = RUNS_FIXTURES / name
        if not path.is_dir():
            raise AssertionError(f"Missing fixture: {path}")
        return path

    return factory


@pytest.fixture
def write_receipt() -> Callable[..., Path]:
    """Persist a :class:`GovernorRun` receipt into a fixture run dir.

    Use with ``fixture_run_dir`` to seed an existing run that the
    adapter will validate.
    """

    def factory(
        run_path: Path,
        *,
        mission_id: str = "mission-001",
        repository_path: str | None = None,
        revision: str | None = None,
        remote_identity: str | None = None,
        worktree: str | None = None,
        contract_digest: str | None = None,
        adapter_version: str = "0.1.0",
    ) -> Path:
        repo = repository_path or str(run_path.parent.parent.parent)
        compat = CompatibilityKey(
            repository={
                "canonical_path": repo,
                "remote_identity": remote_identity,
                "revision": revision,
                "worktree": worktree,
            },
            mission={
                "mission_id": mission_id,
                "contract_digest": contract_digest,
            },
            policy_version=1,
            adapter_version=adapter_version,
        )
        receipt = GovernorRun(
            run_id=run_path.name,
            repository=Path(repo),
            contract_path=run_path / "contract.yaml",
            started_at="2026-08-05T09:00:00+00:00",
            compatibility=compat,
        )
        (run_path / "adapter-receipt.json").write_text(
            receipt.model_dump_json(indent=2), encoding="utf-8"
        )
        return run_path

    return factory


@pytest.fixture
def populate_runs_root(tmp_path: Path) -> Callable[..., Path]:
    """Create ``<tmp>/.animus-loop-governor/runs/<id>/`` with files."""

    def factory(
        run_id: str,
        files: dict[str, str | bytes] | None = None,
    ) -> Path:
        runs = tmp_path / ".animus-loop-governor" / "runs" / run_id
        runs.mkdir(parents=True)
        if files:
            for name, content in files.items():
                target = runs / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    target.write_bytes(content)
                else:
                    target.write_text(content, encoding="utf-8")
        return tmp_path

    return factory


@pytest.fixture
def mock_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> Callable[..., MagicMock]:
    """Patch :func:`subprocess.run`; return the mock for assertions."""
    mock = MagicMock()
    mock.return_value = MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(
        "animus_forge.governor.client.subprocess.run", mock
    )
    return mock
