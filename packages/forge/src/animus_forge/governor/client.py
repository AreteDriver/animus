"""Subprocess wrapper around the ``alg`` CLI.

Owns the process-level contract:

* never ``shell=True`` — arguments are passed as a sequence
* bounded captured output (stdout + stderr truncated to ``MAX_OUTPUT_BYTES``)
* explicit timeout (no runaway ``alg``)
* sanitized environment — strips secrets that may be on the host's
  env and re-adds only the whitelisted variables needed for the CLI
  to function
* targeted exceptions: :class:`AlgNotFoundError`,
  :class:`GovernorTimeoutError`, exit-code → typed via
  :func:`exit_codes.map_exit_code`
* exit-code 0 stdout is parsed by the public methods (``start``,
  ``verify``) — the second plain line of ``alg start`` output is the
  run directory and the only canonical way to learn the new run id

The class is constructed once and reused. The ``alg`` binary is
resolved on first use via :func:`shutil.which` and cached.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from animus_forge.governor.errors import (
    AlgNotFoundError,
    GovernorTimeoutError,
)
from animus_forge.governor.exit_codes import map_exit_code

# 256 KiB cap on captured output. ``alg`` emits human-friendly Rich
# output which is bounded; truncation is paranoid defence against a
# future regression that emits unbounded output.
MAX_OUTPUT_BYTES = 262_144

# Default timeout for ``alg`` invocations. ``alg verify`` runs
# watchdog + completion compute which can be slow on large repos.
DEFAULT_TIMEOUT_SECONDS = 120.0

# Env vars we forward from the host. The Governor is vendor-neutral
# and only needs minimal env. Secrets (e.g. ``ANTHROPIC_API_KEY``,
# ``OPENAI_API_KEY``) are stripped — the Governor never makes model
# calls; passing them is a leak surface.
SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TZ",
        "USER",
        "LOGNAME",
        "TMPDIR",
    }
)


def _sanitized_environment(
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a minimal environment for ``alg`` invocation.

    Strips every host env var except those in :data:`SAFE_ENV_KEYS`.
    Callers may add additional variables via ``extra``; this is the
    only legitimate way to extend the env (no shell injection
    surface).
    """
    safe = {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS}
    if extra:
        safe.update(extra)
    return safe


class GovernorClient:
    """Thin subprocess wrapper for the ``alg`` CLI.

    The class is intentionally narrow: it only knows how to invoke
    ``alg`` and map exit codes to typed exceptions. The orchestration
    logic (when to compile, when to start, when to verify) lives in
    :mod:`adapter`.

    Args:
        alg_binary: Absolute path to ``alg``. When ``None``, the
            binary is resolved on first use via :func:`shutil.which`
            (``alg`` on ``PATH``).
        default_timeout: Subprocess timeout in seconds. Defaults to
            :data:`DEFAULT_TIMEOUT_SECONDS`.
        env_extra: Additional environment variables to add on every
            invocation.
    """

    def __init__(
        self,
        alg_binary: str | Path | None = None,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        env_extra: Mapping[str, str] | None = None,
    ) -> None:
        self._explicit_binary = str(alg_binary) if alg_binary is not None else None
        self._resolved_binary: str | None = None
        self._default_timeout = default_timeout
        self._env_extra = dict(env_extra) if env_extra else {}

    @property
    def binary(self) -> str:
        """Return the resolved ``alg`` binary path.

        Resolves on first access; cached for the client's lifetime.
        Raises :class:`AlgNotFoundError` if not on ``PATH``.
        """
        if self._resolved_binary is None:
            self._resolved_binary = self._resolve_binary()
        return self._resolved_binary

    def _resolve_binary(self) -> str:
        """Resolve the ``alg`` binary path, raising if missing.

        When the binary is not found on PATH or the explicit path
        does not exist, raise :class:`AlgNotFoundError` immediately —
        the caller almost always wants to surface this before any
        subprocess work. Mocks that bypass ``subprocess.run`` also
        bypass this check by overriding :attr:`binary`.
        """
        if self._explicit_binary is not None:
            if not Path(self._explicit_binary).is_file():
                raise AlgNotFoundError(f"alg binary not found at {self._explicit_binary}")
            return self._explicit_binary
        located = shutil.which("alg")
        if located is None:
            raise AlgNotFoundError("`alg` not on PATH; install animus_loop_governor wheel")
        return located

    def _run(
        self,
        args: list[str],
        *,
        cwd: Path | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[str]:
        """Invoke ``alg`` with the given args; map exit code to typed.

        Never uses ``shell=True`` — ``args`` is a sequence passed
        directly to :func:`subprocess.run`. The first element must be
        a subcommand (``compile``, ``start``, ``verify``, etc.).
        """
        if not args:
            raise ValueError("args must include at least one element")
        binary = self.binary  # raises AlgNotFoundError on miss
        cmd = [binary, *args]
        effective_timeout = timeout if timeout is not None else self._default_timeout
        env = _sanitized_environment(self._env_extra)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AlgNotFoundError(f"alg binary not found at {binary}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GovernorTimeoutError(
                f"alg {' '.join(args)} timed out after {effective_timeout}s",
                timeout=effective_timeout,
            ) from exc
        except PermissionError as exc:
            raise AlgNotFoundError(f"alg binary at {binary} is not executable") from exc

        # Truncate captured output to the documented bound before
        # any caller parses it.
        result.stdout = (result.stdout or "")[-MAX_OUTPUT_BYTES:]
        result.stderr = (result.stderr or "")[-MAX_OUTPUT_BYTES:]

        map_exit_code(
            returncode=result.returncode,
            stderr=result.stderr,
            subcommand=args[0],
        )
        return result

    # ----- Subcommand helpers --------------------------------------------

    def compile(
        self,
        request: Path,
        draft: Path,
        output: Path,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> Path:
        """Invoke ``alg compile``. Returns the output contract path.

        On exit 2 raises :class:`ContractRejectedError`.
        """
        result = self._run(
            [
                "compile",
                "--request",
                str(request),
                "--draft",
                str(draft),
                "--output",
                str(output),
            ],
            cwd=cwd,
            timeout=timeout,
        )
        _ensure_success(result, "compile")
        return output

    def start(
        self,
        contract_path: Path,
        *,
        cwd: Path,
        run_id: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Invoke ``alg start``. Returns the new run id.

        Parses the second plain line of stdout for the run directory;
        the run id is the leaf name. (See
        ``animus-loop-governor/src/animus_loop_governor/cli.py`` —
        ``console.print(str(run_dir))``.)
        """
        args = [
            "start",
            "--contract",
            str(contract_path),
            "--root",
            str(cwd),
        ]
        if run_id is not None:
            args.extend(["--run-id", run_id])
        result = self._run(args, cwd=cwd, timeout=timeout)
        return _parse_run_id_from_start_stdout(result.stdout)

    def verify(
        self,
        run_id: str,
        *,
        cwd: Path,
        timeout: float | None = None,
    ) -> None:
        """Invoke ``alg verify``.

        On exit 3 (normal denial) raises :class:`VerifyDeniedError`.
        The caller is expected to read ``completion-latest.json``
        separately (see :mod:`adapter`).
        """
        result = self._run(
            ["verify", run_id, "--root", str(cwd)],
            cwd=cwd,
            timeout=timeout,
        )
        _ensure_success(result, "verify")


def _ensure_success(
    result: subprocess.CompletedProcess[str],
    subcommand: str,
) -> None:
    """Surface unexpected non-zero exit codes from successful paths.

    ``map_exit_code`` already raised a typed exception for known
    codes. This helper exists to crash loudly on any other failure
    mode rather than silently returning garbage.
    """
    if result.returncode != 0:
        # map_exit_code is called first inside _run — reaching here
        # means a future exit code appeared that we forgot to map.
        raise RuntimeError(
            f"alg {subcommand} returned {result.returncode} without a "
            "typed exception mapping; stderr was: "
            f"{result.stderr.strip()}"
        )


_RUN_ID_PREFIX = re.compile(r"Createdrun(run-[A-Za-z0-9]+)")


def _parse_run_id_from_start_stdout(stdout: str) -> str:
    """Extract the new run id from ``alg start`` output.

    ``alg start`` prints two pieces of information:

    * ``Created run <id>`` — the canonical run id (no Rich markup)
    * the absolute path of the run dir (Rich-formatted)

    Rich's ``Console`` soft-wraps long lines at the terminal width and
    may also break the run id across multiple lines when the terminal
    is narrow (e.g. CI runners, tmux panes with constrained width).
    We therefore strip ANSI escapes, collapse all whitespace into
    nothing, and search for the canonical ``Createdrun<id>`` marker.
    The run-id format (``run-<hex>``) is unambiguous and survives any
    soft-wrap pattern — including one that breaks the run id itself
    across multiple lines.
    """
    cleaned = _strip_rich(stdout)
    collapsed = re.sub(r"\s+", "", cleaned)
    match = _RUN_ID_PREFIX.search(collapsed)
    if match is None:
        raise ValueError(
            f"alg start emitted unexpected stdout; cannot parse run id. stdout was: {stdout!r}"
        )
    return match.group(1)


def _strip_rich(line: str) -> str:
    """Remove Rich decorations from a stdout line.

    * ANSI CSI sequences (``ESC [ ... letter``).
    * Rich markup tags (``[bold]text[/bold]`` -> ``text``).
    """
    # Strip ANSI CSI sequences (ESC [ ... letter).
    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", line)
    # Strip Rich opening/closing markup tags like [bold] or [/bold].
    cleaned = re.sub(r"\[/?[a-zA-Z][a-zA-Z0-9_]*\]", "", cleaned)
    return cleaned


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "GovernorClient",
    "MAX_OUTPUT_BYTES",
    "SAFE_ENV_KEYS",
    "_sanitized_environment",
]
