"""Bash tool — spec §4.5 + default-deny policy R11."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from ..types import ToolError

# Spec R11 default-deny patterns. Plain-substring match where unambiguous,
# regex where substring would false-positive (curl|sh).
DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "sudo",
    "mkfs",
    "dd if=",
)

# Match `curl ... | sh` (with optional flags/args, optional whitespace).
_CURL_SH_RX = re.compile(r"\bcurl\b[^|]*\|\s*sh\b")

# Spec R11: "any command containing literal `~` or `$HOME`".
_HOME_RX = re.compile(r"(?<!\\)\$HOME\b|(?<!\w)~(?=/|\s|$)")

_TRUNC_MARKER = "\n[... truncated, {n} bytes total]"
_MAX_STREAM_BYTES = 8 * 1024  # 8 KB per stream per spec §4.5


def _truncate(stream: str, limit: int = _MAX_STREAM_BYTES) -> str:
    """Truncate a stdout/stderr stream and append a marker."""
    data = stream.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return stream
    head = data[:limit].decode("utf-8", errors="replace")
    return head + _TRUNC_MARKER.format(n=len(data))


def _policy_violation(command: str, denylist: tuple[str, ...]) -> str | None:
    """Return a human-readable reason if `command` violates policy, else None."""
    for pattern in denylist:
        if pattern in command:
            return f"matches {pattern!r}"
    if _CURL_SH_RX.search(command):
        return "matches curl | sh pattern"
    if _HOME_RX.search(command):
        return "contains literal '~' or '$HOME'"
    return None


def bash(
    command: str,
    project_root: Path,
    timeout_s: int = 30,
    denylist: tuple[str, ...] = DEFAULT_DENY_PATTERNS,
) -> dict:
    """Execute a shell command in `project_root` with policy enforcement.

    Args:
        command: Shell command. Subject to denylist + `~`/`$HOME` ban.
        project_root: cwd for the subprocess.
        timeout_s: Wall-time cap. Silently clamped to 300 per spec.
        denylist: Substring patterns that always deny. Defaults to spec R11.

    Returns:
        `{"exit_code": int, "stdout": str, "stderr": str, "wall_seconds": float}`
        Stdout/stderr capped at 8 KB each with truncation marker.

    Raises:
        ToolError: on policy violation. Subprocess timeout → non-raising
            return with `exit_code=124` per Unix convention, marker in stderr.
    """
    reason = _policy_violation(command, denylist)
    if reason:
        raise ToolError(f"command denied by policy: {reason}")

    timeout_s = min(max(1, timeout_s), 300)

    start = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S602 — shell intentional, gated by policy
            command,
            shell=True,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        wall = time.monotonic() - start
        return {
            "exit_code": completed.returncode,
            "stdout": _truncate(completed.stdout),
            "stderr": _truncate(completed.stderr),
            "wall_seconds": wall,
        }
    except subprocess.TimeoutExpired as e:
        wall = time.monotonic() - start
        stderr = (e.stderr.decode("utf-8", errors="replace") if e.stderr else "") + (
            f"\n[timeout after {timeout_s}s]"
        )
        stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
        return {
            "exit_code": 124,
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr),
            "wall_seconds": wall,
        }
