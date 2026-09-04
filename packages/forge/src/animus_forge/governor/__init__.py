"""Adapter that wraps the ``alg`` CLI as a Forge-side verifier.

Public surface (everything else is implementation detail):

* :class:`GovernorClient` — subprocess wrapper around ``alg``
* :class:`GovernorAdapter` — :meth:`ensure_run` orchestrator
* :class:`GovernorVerifierCitizen` — verifier Forge citizen
* :class:`GovernorRun`, :class:`CompatibilityKey` — receipt / key models
* :func:`compute_compatibility_key` — derive a key from a repository
* Exit-code mapping (:func:`map_exit_code`) and the exception hierarchy
  in :mod:`errors`

See :mod:`adapter` for the entry point used by the scheduler.
"""

from __future__ import annotations

from animus_forge.governor.adapter import (
    GovernorAdapter,
    GovernorVerifierCitizen,
    RunStateReader,
    compute_compatibility_key,
)
from animus_forge.governor.client import GovernorClient
from animus_forge.governor.errors import (
    AlgNotFoundError,
    ConcurrentPreparationError,
    ContractIntegrityError,
    ContractRejectedError,
    GovernorAdapterError,
    GovernorError,
    GovernorTimeoutError,
    PermissionDeniedError,
    RunNotFoundError,
    RunStateInvalidError,
    RunUnusableError,
    VerifyDeniedError,
)
from animus_forge.governor.exit_codes import map_exit_code
from animus_forge.governor.models import (
    CompatibilityKey,
    GovernorRun,
    MissionKey,
    RepositoryKey,
)

__all__ = [
    "AlgNotFoundError",
    "CompatibilityKey",
    "ConcurrentPreparationError",
    "ContractIntegrityError",
    "ContractRejectedError",
    "GovernorAdapter",
    "GovernorAdapterError",
    "GovernorClient",
    "GovernorError",
    "GovernorRun",
    "GovernorTimeoutError",
    "GovernorVerifierCitizen",
    "MissionKey",
    "RunStateReader",
    "PermissionDeniedError",
    "RepositoryKey",
    "RunNotFoundError",
    "RunStateInvalidError",
    "RunUnusableError",
    "VerifyDeniedError",
    "compute_compatibility_key",
    "map_exit_code",
]
