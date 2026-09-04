"""Typed exceptions for animus_forge.governor.

Hierarchy maps ``alg`` exit codes and infrastructure failures to specific
exception types so callers can distinguish "the Governor denied completion"
(the *normal* path that drives a retry) from "the Governor is broken" (a
hard infrastructure failure).

Every typed exception inherits from :class:`GovernorAdapterError`. One
``except GovernorAdapterError`` clause catches the whole module.

Mapping (see :mod:`exit_codes`):

* 0 → success (no exception)
* 1 → :class:`GovernorError` (sniffed to :class:`PermissionDeniedError`
  or :class:`ContractIntegrityError` based on stderr)
* 2 → :class:`ContractRejectedError` (only on ``alg compile``)
* 3 → :class:`VerifyDeniedError` (only on ``alg verify``)

Non-exit-code failures:

* :class:`AlgNotFoundError` — ``alg`` binary missing or not executable
* :class:`GovernorTimeoutError` — subprocess exceeded timeout
* :class:`RunNotFoundError` — run directory missing on disk
* :class:`RunStateInvalidError` — run-state JSON corrupt
* :class:`RunUnusableError` — known run id points to terminated /
  cross-repo / incompatible / uninitialized run
* :class:`ConcurrentPreparationError` — another caller raced to prepare
  the same mission and we lost the compare-and-swap
"""

from __future__ import annotations


class GovernorAdapterError(Exception):
    """Base class for every adapter exception.

    Catching this once catches every failure the adapter can surface.
    """

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message


class AlgNotFoundError(GovernorAdapterError):
    """``alg`` binary is missing from PATH or not executable."""


class GovernorError(GovernorAdapterError):
    """Generic non-zero exit from ``alg`` that does not match a more
    specific exception below.

    Carries the stderr text, exit code, and the subcommand that produced
    the failure for diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        exit_code: int,
        subcommand: str,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.subcommand = subcommand
        self.stderr = message


class ContractRejectedError(GovernorAdapterError):
    """``alg compile`` rejected the contract YAML (exit 2)."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = message


class VerifyDeniedError(GovernorAdapterError):
    """``alg verify`` denied completion (exit 3). Normal retry path."""

    def __init__(self, message: str, *, exit_code: int = 3) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = message


class PermissionDeniedError(GovernorAdapterError):
    """``alg`` refused an operation due to role permission (sniffed from
    exit-1 stderr containing ``"PermissionDenied"``)."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = message


class ContractIntegrityError(GovernorAdapterError):
    """Sealed contract ``contract.sha256`` drift detected (sniffed from
    exit-1 stderr containing ``"integrity"``). The run is unrecoverable
    without a new contract."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = message


class RunNotFoundError(GovernorAdapterError):
    """Expected run directory is missing on disk."""


class RunStateInvalidError(GovernorAdapterError):
    """A run-state JSON file is missing or corrupt."""


class RunUnusableError(GovernorAdapterError):
    """A known run id was supplied but the run cannot be reused.

    Reasons: terminated, belongs to another repository, on an
    incompatible branch/commit, partially initialized, or policy-
    incompatible. The adapter raises this instead of silently using a
    broken run; the caller must create a fresh run.
    """


class GovernorTimeoutError(GovernorAdapterError):
    """``subprocess.run`` exceeded the timeout while invoking ``alg``."""

    def __init__(self, message: str, *, timeout: float) -> None:
        super().__init__(message)
        self.timeout = timeout


class ConcurrentPreparationError(GovernorAdapterError):
    """Another caller raced to prepare the same mission and won.

    The adapter's compare-and-swap transition returned "already prepared
    by someone else" — this is normal in a multi-worker scheduler and
    not a fault. The caller should reload the mission and proceed.
    """


__all__ = [
    "AlgNotFoundError",
    "ConcurrentPreparationError",
    "ContractIntegrityError",
    "ContractRejectedError",
    "GovernorAdapterError",
    "GovernorError",
    "GovernorTimeoutError",
    "PermissionDeniedError",
    "RunNotFoundError",
    "RunStateInvalidError",
    "RunUnusableError",
    "VerifyDeniedError",
]
