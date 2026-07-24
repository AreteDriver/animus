"""Cognitive Event Ledger — append-only event store with chained integrity.

Public API
----------
:class:`LedgerEvent` — Canonical event model (matches ``ledger_event.schema.json``).
:class:`LedgerEntry` — Event plus store metadata (``chain_hash``, ``db_id``).
:class:`IntegrityChain` — Compute and verify SHA-256 chain hashes.
:class:`LedgerStore` — SQLite-backed atomic append-only store.
"""

from __future__ import annotations

from animus_bootstrap.ledger.models import EventType, IntegrityChain, LedgerEntry, LedgerEvent
from animus_bootstrap.ledger.store import LedgerStore

__all__ = [
    "EventType",
    "IntegrityChain",
    "LedgerEntry",
    "LedgerEvent",
    "LedgerStore",
]
