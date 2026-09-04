"""Tests for :class:`GovernorClient` subprocess wrapper.

Covers the construction matrix (paths with spaces, missing alg,
no shell) and the output parsing matrix (valid, missing run id,
malformed JSON-like, oversized stderr).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from animus_forge.governor.client import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_BYTES,
    SAFE_ENV_KEYS,
    GovernorClient,
    _sanitized_environment,
)
from animus_forge.governor.errors import (
    AlgNotFoundError,
    ContractRejectedError,
    GovernorTimeoutError,
    VerifyDeniedError,
)


@pytest.fixture
def fake_alg_path(tmp_path: Path) -> Path:
    """An existing executable file at ``tmp_path/alg``.

    Production code checks ``is_file()`` on the explicit-binary path;
    unit tests that mock ``subprocess.run`` need a real file so that
    check passes. The contents are inert — they never run.
    """
    path = tmp_path / "alg"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# Environment sanitisation
# ---------------------------------------------------------------------------


def test_sanitized_environment_drops_secrets() -> None:
    """``ANTHROPIC_API_KEY`` and similar are stripped."""
    env = _sanitized_environment()
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "GH_TOKEN" not in env


def test_sanitized_environment_keeps_safe_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every safe key set on the host survives sanitisation."""
    for key in SAFE_ENV_KEYS:
        monkeypatch.setenv(key, f"value-for-{key}")
    env = _sanitized_environment()
    for key in SAFE_ENV_KEYS:
        assert env.get(key) == f"value-for-{key}"


def test_sanitized_environment_extra_is_merged() -> None:
    """Caller-provided extras are added verbatim."""
    env = _sanitized_environment({"GOVERNOR_DEBUG": "1"})
    assert env["GOVERNOR_DEBUG"] == "1"


def test_safe_env_keys_does_not_include_secrets() -> None:
    """The safe-keys list never accidentally whitelists a secret."""
    for key in SAFE_ENV_KEYS:
        assert "KEY" not in key
        assert "SECRET" not in key
        assert "TOKEN" not in key


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------


def test_binary_missing_raises(tmp_path: Path) -> None:
    """Explicit binary that does not exist → :class:`AlgNotFoundError`."""
    client = GovernorClient(alg_binary=tmp_path / "no-such-binary")
    with pytest.raises(AlgNotFoundError):
        _ = client.binary


def test_binary_not_on_path_raises(tmp_path: Path) -> None:
    """Empty PATH → :class:`AlgNotFoundError`."""
    client = GovernorClient(alg_binary=None)
    with pytest.raises(AlgNotFoundError):
        _ = client.binary


# ---------------------------------------------------------------------------
# Subprocess construction (never shell=True, sequence args, sanitized env)
# ---------------------------------------------------------------------------


def test_run_passes_args_as_sequence(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``subprocess.run`` receives a sequence, not a string."""
    client = GovernorClient(alg_binary=str(fake_alg_path))
    mock_subprocess_run.return_value.stdout = ""
    mock_subprocess_run.return_value.stderr = ""
    mock_subprocess_run.return_value.returncode = 0
    client._run(["status", "run-x"], cwd=tmp_path, timeout=10.0)
    call = mock_subprocess_run.call_args
    args = call.args[0]
    assert args[0] == str(fake_alg_path)
    assert args[1] == "status"
    assert args[2] == "run-x"
    assert call.kwargs.get("shell", False) is False


def test_run_never_uses_shell(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``shell`` keyword is never truthy."""
    mock_subprocess_run.return_value.stdout = ""
    mock_subprocess_run.return_value.stderr = ""
    mock_subprocess_run.return_value.returncode = 0
    client = GovernorClient(alg_binary=str(fake_alg_path))
    client._run(["verify", "run-x"], cwd=tmp_path, timeout=None)
    assert mock_subprocess_run.call_args.kwargs["shell"] is False


def test_run_strips_secrets_from_env(
    tmp_path: Path,
    mock_subprocess_run: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    fake_alg_path: Path,
) -> None:
    """Host secrets must not reach the subprocess."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    mock_subprocess_run.return_value.returncode = 0
    client = GovernorClient(alg_binary=str(fake_alg_path))
    client._run(["status", "run-x"], cwd=tmp_path, timeout=None)
    env = mock_subprocess_run.call_args.kwargs["env"]
    assert env.get("ANTHROPIC_API_KEY") is None
    assert env.get("OPENAI_API_KEY") is None


def test_run_handles_paths_with_spaces(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """Repository paths containing spaces reach the subprocess intact."""
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    mock_subprocess_run.return_value.returncode = 0
    client = GovernorClient(alg_binary=str(fake_alg_path))
    client._run(["verify", "run-x"], cwd=repo, timeout=None)
    # cwd is passed as kwarg, never as an argv element. ``alg`` finds
    # the repository via cwd, not via the args list.
    assert mock_subprocess_run.call_args.kwargs["cwd"] == str(repo)
    # ``shell=False`` is the only thing preventing shell metacharacter
    # interpretation of the space-containing path.
    assert mock_subprocess_run.call_args.kwargs["shell"] is False


def test_run_resolves_relative_cwd(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``cwd`` is passed as a stringified path."""
    mock_subprocess_run.return_value.returncode = 0
    client = GovernorClient(alg_binary=str(fake_alg_path))
    client._run(["status", "x"], cwd=tmp_path, timeout=None)
    assert mock_subprocess_run.call_args.kwargs["cwd"] == str(tmp_path)


def test_run_uses_explicit_timeout(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    mock_subprocess_run.return_value.returncode = 0
    client = GovernorClient(alg_binary=str(fake_alg_path))
    client._run(["verify", "x"], cwd=tmp_path, timeout=15.0)
    assert mock_subprocess_run.call_args.kwargs["timeout"] == 15.0


def test_run_default_timeout_when_none(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    mock_subprocess_run.return_value.returncode = 0
    client = GovernorClient(alg_binary=str(fake_alg_path))
    client._run(["verify", "x"], cwd=tmp_path, timeout=None)
    assert mock_subprocess_run.call_args.kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS


def test_run_truncates_oversized_stdout(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """stdout exceeding ``MAX_OUTPUT_BYTES`` is truncated."""
    huge = "A" * (MAX_OUTPUT_BYTES * 2)
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout=huge, stderr="")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    result = client._run(["status", "x"], cwd=tmp_path, timeout=None)
    assert len(result.stdout) == MAX_OUTPUT_BYTES


def test_run_truncates_oversized_stderr(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    huge = "B" * (MAX_OUTPUT_BYTES * 2)
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr=huge)
    client = GovernorClient(alg_binary=str(fake_alg_path))
    # Exit 1 + huge stderr → GovernorError with truncated stderr
    mock_subprocess_run.return_value.returncode = 1
    from animus_forge.governor.errors import GovernorError

    with pytest.raises(GovernorError) as excinfo:
        client._run(["verify", "x"], cwd=tmp_path, timeout=None)
    assert len(excinfo.value.stderr) == MAX_OUTPUT_BYTES


def test_run_handles_filenotfound(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``FileNotFoundError`` from subprocess → :class:`AlgNotFoundError`."""
    mock_subprocess_run.side_effect = FileNotFoundError("no alg")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(AlgNotFoundError):
        client._run(["verify", "x"], cwd=tmp_path, timeout=None)


def test_run_handles_timeout(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``TimeoutExpired`` → :class:`GovernorTimeoutError`."""
    mock_subprocess_run.side_effect = subprocess.TimeoutExpired(
        cmd=[str(fake_alg_path), "verify"], timeout=5.0
    )
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(GovernorTimeoutError) as excinfo:
        client._run(["verify", "x"], cwd=tmp_path, timeout=5.0)
    assert excinfo.value.timeout == 5.0


def test_run_handles_permission_error(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``PermissionError`` from subprocess → :class:`AlgNotFoundError`."""
    mock_subprocess_run.side_effect = PermissionError("not executable")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(AlgNotFoundError):
        client._run(["verify", "x"], cwd=tmp_path, timeout=None)


def test_run_rejects_empty_args(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """Empty arg list is a programming error, not a runtime condition."""
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(ValueError):
        client._run([], cwd=tmp_path, timeout=None)


# ---------------------------------------------------------------------------
# Output parsing — ``alg start`` stdout
# ---------------------------------------------------------------------------


def test_start_parses_two_line_output(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``alg start`` prints ``Created run run-x`` then the run dir."""
    run_dir_path = tmp_path / ".animus-loop-governor" / "runs" / "run-abc"
    mock_subprocess_run.return_value = MagicMock(
        returncode=0,
        stdout=(f"Created run [bold]run-abc[/bold]\n{run_dir_path}\n"),
        stderr="",
    )
    client = GovernorClient(alg_binary=str(fake_alg_path))
    run_id = client.start(
        contract_path=tmp_path / "contract.yaml",
        cwd=tmp_path,
    )
    assert run_id == "run-abc"


def test_start_strips_rich_ansi(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """ANSI escape sequences in stdout do not corrupt the run id."""
    run_dir_path = tmp_path / ".animus-loop-governor" / "runs" / "run-x"
    mock_subprocess_run.return_value = MagicMock(
        returncode=0,
        stdout=(f"\x1b[1mCreated run run-x\x1b[0m\n{run_dir_path}\n"),
        stderr="",
    )
    client = GovernorClient(alg_binary=str(fake_alg_path))
    run_id = client.start(
        contract_path=tmp_path / "contract.yaml",
        cwd=tmp_path,
    )
    assert run_id == "run-x"


def test_start_single_line_stdout_is_sufficient(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """The parser extracts the run id from the canonical ``Created run`` marker.

    The path line is *advisory* (we use the marker directly, not the path
    leaf), so a stdout containing only ``Created run <id>`` is sufficient.
    This is intentional: even when the path line is too long to fit on a
    narrow terminal and Rich wraps it across many lines, the parser
    recovers by anchoring on the canonical marker.
    """
    mock_subprocess_run.return_value = MagicMock(
        returncode=0, stdout="Created run run-x\n", stderr=""
    )
    client = GovernorClient(alg_binary=str(fake_alg_path))
    run_id = client.start(
        contract_path=tmp_path / "contract.yaml",
        cwd=tmp_path,
    )
    assert run_id == "run-x"


def test_start_empty_stdout_raises_value_error(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(ValueError):
        client.start(
            contract_path=tmp_path / "contract.yaml",
            cwd=tmp_path,
        )


def test_start_parses_wrapped_long_path(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """Path line too long for the terminal — Rich soft-wraps onto many lines.

    Regression: the parser used to take ``lines[1].name`` and only worked
    by accident when the run id was shorter than the line width. When
    the run id wrapped across multiple lines, only the first fragment
    was returned. The fix anchors on the canonical ``Created run <id>``
    marker; the path line is informational.
    """
    wrapped = (
        "Created run \nrun-c442326cccf6\n/tmp/pytest-of-arete\n"
        "/pytest-100/test_alg\n_start_creates_canon\n"
        "ical0c76yhjxs/.animu\ns-loop-governor/runs\n/run-c442326cccf6\n"
    )
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout=wrapped, stderr="")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    run_id = client.start(
        contract_path=tmp_path / "contract.yaml",
        cwd=tmp_path,
    )
    assert run_id == "run-c442326cccf6"


def test_start_strips_ansi_escapes(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """Rich ANSI bold escapes around the run id are stripped."""
    wrapped = "Created run \x1b[1mrun-abc123\x1b[0m\n/tmp/.animus-loop-governor/runs/run-abc123\n"
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout=wrapped, stderr="")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    run_id = client.start(
        contract_path=tmp_path / "contract.yaml",
        cwd=tmp_path,
    )
    assert run_id == "run-abc123"


def test_start_strips_rich_markup_tags(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """Rich markup tags (``[bold]...[/bold]``) are stripped, not just ANSI."""
    wrapped = "Created run [bold]run-mno789[/bold]\n/tmp/.animus-loop-governor/runs/run-mno789\n"
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout=wrapped, stderr="")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    run_id = client.start(
        contract_path=tmp_path / "contract.yaml",
        cwd=tmp_path,
    )
    assert run_id == "run-mno789"


def test_start_rejects_stdout_without_run_id_marker(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """Stdout missing the ``Created run`` marker raises :class:`ValueError`."""
    mock_subprocess_run.return_value = MagicMock(
        returncode=0,
        stdout="/tmp/.animus-loop-governor/runs/run-abc\n",
        stderr="",
    )
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(ValueError):
        client.start(
            contract_path=tmp_path / "contract.yaml",
            cwd=tmp_path,
        )


def test_start_with_explicit_run_id(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """``--run-id`` is passed when supplied."""
    run_dir_path = tmp_path / ".animus-loop-governor" / "runs" / "run-given"
    mock_subprocess_run.return_value = MagicMock(
        returncode=0,
        stdout=f"Created run run-given\n{run_dir_path}\n",
        stderr="",
    )
    client = GovernorClient(alg_binary=str(fake_alg_path))
    client.start(
        contract_path=tmp_path / "contract.yaml",
        cwd=tmp_path,
        run_id="run-given",
    )
    cmd = mock_subprocess_run.call_args.args[0]
    assert "--run-id" in cmd
    assert "run-given" in cmd


# ---------------------------------------------------------------------------
# Exit mapping at the client surface
# ---------------------------------------------------------------------------


def test_compile_exit_2_raises_contract_rejected(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    mock_subprocess_run.return_value = MagicMock(returncode=2, stdout="", stderr="bad requirement")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(ContractRejectedError):
        client.compile(
            request=tmp_path / "req.yaml",
            draft=tmp_path / "draft.yaml",
            output=tmp_path / "out.yaml",
            cwd=tmp_path,
        )


def test_verify_exit_3_raises_verify_denied(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    mock_subprocess_run.return_value = MagicMock(
        returncode=3, stdout="NOT DONE", stderr="missing evidence"
    )
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(VerifyDeniedError):
        client.verify(run_id="run-x", cwd=tmp_path)


def test_compile_unexpected_exit_crashes(
    tmp_path: Path, mock_subprocess_run: MagicMock, fake_alg_path: Path
) -> None:
    """Unmapped exit code (rc=99) on a successful path is a bug."""
    mock_subprocess_run.return_value = MagicMock(returncode=99, stdout="", stderr="???")
    client = GovernorClient(alg_binary=str(fake_alg_path))
    with pytest.raises(RuntimeError):
        client.compile(
            request=tmp_path / "req.yaml",
            draft=tmp_path / "draft.yaml",
            output=tmp_path / "out.yaml",
            cwd=tmp_path,
        )
