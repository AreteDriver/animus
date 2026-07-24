"""Cognitive Event Ledger — models.

Pydantic models for the append-only event store with chained SHA-256 integrity.
Compatible with ``ledger_event.schema.json`` and ``common.schema.json``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, conint, constr


class EventType(StrEnum):
    """Canonical event types for the Cognitive Event Ledger."""

    created = "created"
    updated = "updated"
    superseded = "superseded"
    approved = "approved"
    rejected = "rejected"
    deleted = "deleted"
    restored = "restored"
    exported = "exported"
    imported = "imported"


class LedgerEvent(BaseModel):
    """Single immutable event record.

    Matches ``ledger_event.schema.json`` with an optional ``parent_event_id``
    linking this event to its predecessor in the integrity chain.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: constr(pattern=r"^evt-[a-z0-9_-]+$")
    event_type: EventType
    object_id: constr(pattern=r"^[a-z][a-z0-9_-]{2,127}$")
    object_version: conint(ge=1)
    principal: constr(min_length=3)
    workspace_id: constr(pattern=r"^ws-[a-z0-9_-]+$")
    payload: dict[str, Any]
    integrity_hash: constr(pattern=r"^[a-f0-9]{64}$")
    tx_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parent_event_id: str | None = None


class LedgerEntry(LedgerEvent):
    """Internal representation with database metadata.

    Extends :class:`LedgerEvent` with fields managed by the store:
    ``chain_hash`` (crytographic link to previous entry) and
    ``db_id`` (SQLite row id, not part of the canonical event).
    """

    model_config = ConfigDict(extra="forbid")

    chain_hash: constr(pattern=r"^[a-f0-9]{64}$")
    db_id: int | None = None


class IntegrityChain:
    """Compute and verify chained SHA-256 integrity hashes.

    The chain works like a lightweight blockchain: each entry's
    ``chain_hash`` is ``SHA256(prev_chain_hash || serialized_entry)``.
    The first entry uses ``GENESIS_HASH`` as its predecessor.
    """

    GENESIS_HASH: str = "0" * 64

    @classmethod
    def hash_payload(cls, payload: dict[str, Any]) -> str:
        """Return SHA-256 hex digest of a normalized JSON payload."""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def compute_chain_hash(
        cls,
        event: LedgerEvent,
        previous_chain_hash: str | None = None,
    ) -> str:
        """Compute the chain hash for *event* given *previous_chain_hash*."""
        prev = previous_chain_hash if previous_chain_hash is not None else cls.GENESIS_HASH
        # Serialize the event deterministically (exclude chain_hash and db_id)
        data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "object_id": event.object_id,
            "object_version": event.object_version,
            "principal": event.principal,
            "workspace_id": event.workspace_id,
            "payload": event.payload,
            "integrity_hash": event.integrity_hash,
            "tx_time": event.tx_time.isoformat(),
            "parent_event_id": event.parent_event_id,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()

    @classmethod
    def verify(cls, entries: list[LedgerEntry]) -> bool:
        """Verify the integrity chain for a sequence of entries.

        Returns ``True`` if every entry's ``chain_hash`` matches the
        recomputed value based on the previous entry. Returns ``False``
        on the first broken link.
        """
        prev_hash: str | None = None
        for entry in entries:
            expected = cls.compute_chain_hash(entry, prev_hash)
            if entry.chain_hash != expected:
                return False
            prev_hash = entry.chain_hash
        return True
