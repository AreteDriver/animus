"""Starter tools exposed to the agent (spec R5, §4.2-§4.6).

Five tools — ReadFile, WriteFile, EditFile, Bash, Grep — all path-confined
to `--project` root, all raising `ToolError` on policy / input violations.
"""

from __future__ import annotations

from .base import resolve_within_root
from .bash import DEFAULT_DENY_PATTERNS, bash
from .edit_file import edit_file
from .grep import grep
from .read_file import read_file
from .write_file import write_file

__all__ = [
    "DEFAULT_DENY_PATTERNS",
    "bash",
    "edit_file",
    "grep",
    "read_file",
    "resolve_within_root",
    "write_file",
]
