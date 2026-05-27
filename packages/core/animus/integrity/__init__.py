"""Tampering detection for critical-path hardening code (10/10 polish).

If an adversary lands daemon-execution privileges and patches
``redaction.py``, ``egress.py``, ``mcp_server.py``, etc. to no-op, the
existing hardening gates silently fail. ``verify_hardening`` catches
this at test time; this module catches it at daemon startup.

Baseline manifest at ``<data_dir>/integrity-baseline.json``. Daemon
calls :func:`verify_or_raise` at startup; refuses to boot if any
tracked file's SHA-256 has drifted from the baseline.

Legitimate updates regenerate the baseline via :func:`regenerate_baseline`.
"""

from animus.integrity.checker import (
    IntegrityMismatchError,
    IntegrityNotInitializedError,
    regenerate_baseline,
    verify_or_raise,
)

__all__ = [
    "IntegrityMismatchError",
    "IntegrityNotInitializedError",
    "regenerate_baseline",
    "verify_or_raise",
]
