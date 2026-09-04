"""Adapter-side models: compatibility keys and receipts.

These are *not* mirrors of Governor JSON schemas — they describe the
adapter's view of a run (what's reusable, what was created, what
contract was sealed). The Governor knows nothing about them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _AdapterModel(BaseModel):
    """Local base for adapter models.

    Uses ``extra="ignore"`` (not ``"forbid"``) because the adapter
    occasionally decorates receipts with diagnostic fields from
    callers; rejecting them would break legitimate extensions.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class RepositoryKey(_AdapterModel):
    """Identity of a workspace the Governor is being asked to govern.

    A run is only reusable across calls when this key matches
    exactly. Two missions on different branches or worktrees produce
    different keys and must not silently share a mutable run.
    """

    canonical_path: str
    remote_identity: str | None = None
    revision: str | None = None
    worktree: str | None = None


class MissionKey(_AdapterModel):
    """Identity of the mission asking for a Governor run.

    Reuse within the same mission is allowed; reuse across missions
    sharing the same repository is **not** the default. Sharing would
    require an explicit policy field — out of scope for v0.1.0.
    """

    mission_id: str
    contract_digest: str | None = None

    @field_validator("mission_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("mission_id must be a non-empty string")
        return value


class CompatibilityKey(_AdapterModel):
    """Composite key used by :meth:`adapter.ensure_run` to decide
    whether an existing run may be reused.

    Two runs are compatible only if all four sub-keys match:

    * :class:`RepositoryKey` — same workspace identity
    * :class:`MissionKey` — same mission
    * ``policy_version`` — Governor policy revision expected
    * ``adapter_version`` — this adapter's version

    Mismatch on any field → :class:`RunUnusableError`. The adapter
    never silently demotes a strict check to a soft one.
    """

    repository: RepositoryKey
    mission: MissionKey
    policy_version: int = 1
    adapter_version: str


class GovernorRun(_AdapterModel):
    """Result of :meth:`adapter.ensure_run`.

    Persisted to ``mission.metadata["governor_run"]`` so the scheduler
    survives restarts. The dict form (``model_dump(mode="json")``) is
    what mission metadata receives — the schema migration policy is
    ``model_config = ConfigDict(extra="ignore")`` so older versions
    tolerate newer fields.
    """

    run_id: str
    repository: Path
    contract_path: Path
    started_at: str  # ISO-8601 UTC, serialised for JSON safety
    compatibility: CompatibilityKey
    diagnostics: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CompatibilityKey",
    "GovernorRun",
    "MissionKey",
    "RepositoryKey",
]
