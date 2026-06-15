"""General-purpose subprocess runner with timeout, output capture, and safety gating."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from animus_kernel.tools.safety import SecurityError
from animus_kernel.utils.validation import validate_shell_command

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_LIMIT_BYTES: int = 10 * 1024 * 1024
_READ_CHUNK_SIZE: int = 65536

# Additional dangerous patterns beyond validation.DANGEROUS_SHELL_PATTERNS.
_EXTRA_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+.*if=/dev/zero", re.IGNORECASE),
    re.compile(r"\brm\s+-[rf]*\s+[/~]", re.IGNORECASE),
]

# Shell metacharacter sequences blocked as defense-in-depth (we never use
# shell=True, but rejecting these prevents accidental injection if the
# implementation ever changes).
_BLOCKED_SEQUENCES: tuple[str, ...] = (";", "|", "&&")


@dataclass(frozen=True)
class CommandResult:
    """Result of a command execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    timeout: bool
    truncated: bool


def _validate_command(cmd: str) -> list[str]:
    """Parse and validate a command string for dangerous patterns.

    Args:
        cmd: Raw command string.

    Returns:
        Parsed token list from ``shlex.split``.

    Raises:
        SecurityError: If the command is empty, syntactically invalid,
            or contains dangerous patterns.
    """
    if not cmd or not cmd.strip():
        raise SecurityError("Command cannot be empty")

    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        raise SecurityError(f"Invalid command syntax: {exc}") from exc

    if not tokens:
        raise SecurityError("Command parsed to empty tokens")

    for seq in _BLOCKED_SEQUENCES:
        if seq in tokens:
            raise SecurityError(
                f"Command contains forbidden shell sequence: {seq!r}"
            )

    try:
        validate_shell_command(cmd, allow_dangerous=False)
    except Exception as exc:
        raise SecurityError(f"Dangerous command blocked: {exc}") from exc

    for pattern in _EXTRA_DANGEROUS_PATTERNS:
        if pattern.search(cmd):
            raise SecurityError(
                f"Dangerous command blocked: matches {pattern.pattern!r}"
            )

    return tokens


def _validate_cwd(cwd: str) -> Path:
    """Validate a working-directory path.

    Args:
        cwd: Directory that must exist.

    Returns:
        Resolved absolute path.

    Raises:
        SecurityError: If the path does not exist or is not a directory.
    """
    path = Path(cwd)
    if not path.exists():
        raise SecurityError(f"Working directory does not exist: {cwd}")
    if not path.is_dir():
        raise SecurityError(f"Path is not a directory: {cwd}")
    return path.resolve()


def _truncate_bytes(data: bytes, limit: int) -> tuple[bytes, bool]:
    """Truncate byte data to *limit* bytes.

    Returns:
        Tuple of (possibly truncated data, whether truncation occurred).
    """
    if len(data) > limit:
        return data[:limit], True
    return data, False


def run(
    cmd: str,
    cwd: str,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Execute a command synchronously with safety gating.

    Args:
        cmd: Command string to execute. Parsed with ``shlex.split``.
        cwd: Working directory for the subprocess.
        timeout: Maximum execution time in seconds, or ``None`` for no limit.
        env: Optional environment dictionary. If ``None``, the current
            process environment is inherited.

    Returns:
        ``CommandResult`` with exit code, captured stdout/stderr,
        duration, timeout flag, and truncation flag.

    Raises:
        SecurityError: If the command or working directory fails validation.
    """
    tokens = _validate_command(cmd)
    resolved_cwd = _validate_cwd(cwd)

    timed_out = False
    proc: subprocess.Popen[bytes] | None = None

    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            tokens,
            cwd=str(resolved_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout_data, stderr_data = proc.communicate()
    except FileNotFoundError as exc:
        raise SecurityError(f"Command not found: {exc}") from exc
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()

    duration_ms = (time.perf_counter() - start) * 1000.0
    stdout_bytes, stdout_trunc = _truncate_bytes(stdout_data, DEFAULT_OUTPUT_LIMIT_BYTES)
    stderr_bytes, stderr_trunc = _truncate_bytes(stderr_data, DEFAULT_OUTPUT_LIMIT_BYTES)

    return CommandResult(
        exit_code=proc.returncode or 0,
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        duration_ms=round(duration_ms, 3),
        timeout=timed_out,
        truncated=stdout_trunc or stderr_trunc,
    )


async def _read_limited(
    reader: asyncio.StreamReader, limit: int
) -> tuple[bytes, bool]:
    """Read from an async stream up to *limit* bytes.

    Args:
        reader: Async stream reader.
        limit: Maximum bytes to retain.

    Returns:
        Tuple of (captured bytes, whether truncation occurred).
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await reader.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            excess = total - limit
            chunks.append(chunk[:-excess])
            # Drain the remainder without storing.
            while await reader.read(_READ_CHUNK_SIZE):
                pass
            return b"".join(chunks), True
        chunks.append(chunk)
    return b"".join(chunks), False


async def arun(
    cmd: str,
    cwd: str,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Execute a command asynchronously with safety gating.

    Args:
        cmd: Command string to execute. Parsed with ``shlex.split``.
        cwd: Working directory for the subprocess.
        timeout: Maximum execution time in seconds, or ``None`` for no limit.
        env: Optional environment dictionary. If ``None``, the current
            process environment is inherited.

    Returns:
        ``CommandResult`` with exit code, captured stdout/stderr,
        duration, timeout flag, and truncation flag.

    Raises:
        SecurityError: If the command or working directory fails validation.
    """
    tokens = _validate_command(cmd)
    resolved_cwd = _validate_cwd(cwd)

    timed_out = False
    stdout_trunc = False
    stderr_trunc = False

    start = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *tokens,
        cwd=str(resolved_cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        (
            (stdout_bytes, stdout_trunc),
            (stderr_bytes, stderr_trunc),
        ) = await asyncio.wait_for(
            asyncio.gather(
                _read_limited(process.stdout, DEFAULT_OUTPUT_LIMIT_BYTES),
                _read_limited(process.stderr, DEFAULT_OUTPUT_LIMIT_BYTES),
            ),
            timeout=timeout,
        )
        exit_code = await process.wait()
    except TimeoutError:
        timed_out = True
        process.kill()
        # Attempt to drain any remaining buffered output after the process dies.
        try:
            (
                (stdout_bytes, stdout_trunc),
                (stderr_bytes, stderr_trunc),
            ) = await asyncio.wait_for(
                asyncio.gather(
                    _read_limited(process.stdout, DEFAULT_OUTPUT_LIMIT_BYTES),
                    _read_limited(process.stderr, DEFAULT_OUTPUT_LIMIT_BYTES),
                ),
                timeout=5.0,
            )
        except TimeoutError:
            stdout_bytes, stderr_bytes = b"", b""
            stdout_trunc, stderr_trunc = False, False
        exit_code = await process.wait()

    duration_ms = (time.perf_counter() - start) * 1000.0

    return CommandResult(
        exit_code=exit_code or 0,
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        duration_ms=round(duration_ms, 3),
        timeout=timed_out,
        truncated=stdout_trunc or stderr_trunc,
    )
