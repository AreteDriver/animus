"""Mapping of ``alg`` exit codes to typed exceptions.

Single source of truth for the exit-code contract. Imported by
:mod:`client` and exercised by the exit-mapping test matrix.
"""

from __future__ import annotations

import re

from animus_forge.governor.errors import (
    ContractIntegrityError,
    ContractRejectedError,
    GovernorError,
    PermissionDeniedError,
    VerifyDeniedError,
)

PERMISSION_DENIED_HINT = re.compile(r"PermissionDenied", re.IGNORECASE)
INTEGRITY_HINT = re.compile(r"integrity", re.IGNORECASE)


def map_exit_code(
    *,
    returncode: int,
    stderr: str,
    subcommand: str,
) -> None:
    """Raise the typed exception that matches an ``alg`` exit code.

    Returns ``None`` on success. Called from
    :meth:`client.GovernorClient._run` after :func:`subprocess.run`.

    Sniffing rules:

    * rc 1 + stderr matches ``PermissionDenied`` →
      :class:`PermissionDeniedError`
    * rc 1 + stderr matches ``integrity`` →
      :class:`ContractIntegrityError`
    * rc 1 otherwise → :class:`GovernorError`
    * rc 2 + subcommand ``compile`` → :class:`ContractRejectedError`
    * rc 2 otherwise → :class:`GovernorError`
    * rc 3 + subcommand ``verify`` → :class:`VerifyDeniedError`
    * rc 3 otherwise → :class:`GovernorError`
    * rc in ``[4, 99]`` (a plausible ``alg`` future code) →
      :class:`GovernorError` (caller decides recovery)
    * rc ≥ 100 (impossible exit code, signal-killed, etc.) →
      :class:`RuntimeError` — fail loud

    The ``RuntimeError`` branch is fail-loud: a wildly out-of-range
    code is almost certainly a process-management bug, not a normal
    ``alg`` failure mode that downstream code knows how to recover
    from.
    """
    if returncode == 0:
        return

    text = (stderr or "").strip()

    if returncode == 1:
        if PERMISSION_DENIED_HINT.search(text):
            raise PermissionDeniedError(text, exit_code=1)
        if INTEGRITY_HINT.search(text):
            raise ContractIntegrityError(text, exit_code=1)
        raise GovernorError(text, exit_code=1, subcommand=subcommand)

    if returncode == 2:
        if subcommand == "compile":
            raise ContractRejectedError(text, exit_code=2)
        raise GovernorError(text, exit_code=2, subcommand=subcommand)

    if returncode == 3:
        if subcommand == "verify":
            raise VerifyDeniedError(text, exit_code=3)
        raise GovernorError(text, exit_code=3, subcommand=subcommand)

    if 4 <= returncode < 99:
        raise GovernorError(text, exit_code=returncode, subcommand=subcommand)

    raise RuntimeError(
        f"alg {subcommand} returned impossible exit code "
        f"{returncode}; stderr={text!r}. Likely a subprocess "
        "management bug — update animus_forge.governor.exit_codes."
    )


__all__ = ["map_exit_code"]
