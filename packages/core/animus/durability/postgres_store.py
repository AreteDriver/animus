"""PostgreSQL-backed durable object store with bitemporal event ledger.

Optional dependency gated by ``sqlalchemy`` availability. When unavailable,
:class:`DurableObjectStore` raises :exc:`RuntimeError` on instantiation with a
helpful install message.

Upstreamed from ``animus-mind`` (Mind-class architecture, 2026-07-06).

Usage::

    store = DurableObjectStore(database_url=os.getenv("ANIMUS_DATABASE_URL"))
    store.create_tables()  # Run once during setup
    obj_id, event_id = store.store(ObjectRecord(
        object_id="mem-001",
        schema_id="memory_candidate",
        payload={"content": "hello"},
    ))
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from animus.logging import get_logger

logger = get_logger("durability.postgres_store")

try:
    from sqlalchemy import (
        Column,
        DateTime,
        Integer,
        JSON,
        String,
        create_engine,
        func,
        select,
    )
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session, declarative_base, sessionmaker

    _HAS_SQLALCHEMY = True
except ImportError:  # pragma: no cover
    _HAS_SQLALCHEMY = False
    # Provide placeholders so type-checking passes even without sqlalchemy
    Engine = Any  # type: ignore[misc,assignment]
    Session = Any  # type: ignore[misc,assignment]
    declarative_base = object  # type: ignore[misc,assignment]
    sessionmaker = object  # type: ignore[misc,assignment]


# ------------------------------------------------------------------
# Domain enums (Mind-class, not kernel memory types)
# ------------------------------------------------------------------


class ObjectType(str, Enum):
    MEMORY = "memory"
    SOURCE = "source"
    CLAIM = "claim"
    FORECAST = "forecast"
    DECISION = "decision"
    ACTION = "action"
    AGENT_CONTRACT = "agent_contract"


class StorageTier(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class SecurityClass(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class EpistemicStatus(str, Enum):
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    REFUTED = "refuted"


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DELETED = "deleted"


class EventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    SUPERSEDED = "superseded"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELETED = "deleted"
    RESTORED = "restored"
    EXPORTED = "exported"
    IMPORTED = "imported"


# ------------------------------------------------------------------
# Domain dataclass
# ------------------------------------------------------------------


@dataclass
class ObjectRecord:
    """A canonical object record in the bitemporal registry."""

    object_id: str
    schema_id: str
    schema_version: str = "1.0.0"
    owner_id: str = "owner-default"
    workspace_id: str = "ws-default"
    subject_domain: str = "self"
    artifact_type: str = ObjectType.MEMORY.value
    cognitive_role: str = "memory"
    workflow_status: str = "active"
    epistemic_status: str = EpistemicStatus.SUPPORTED.value
    lifecycle_status: str = LifecycleStatus.ACTIVE.value
    storage_tier: str = StorageTier.WARM.value
    presentation: str = "canonical"
    security_class: str = SecurityClass.INTERNAL.value
    payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_by: str = "animus"
    trace_id: str | None = None
    version: int = 1


# ------------------------------------------------------------------
# SQLAlchemy models
# ------------------------------------------------------------------

if _HAS_SQLALCHEMY:
    Base = declarative_base()

    class _ObjectRegistryRow(Base):  # type: ignore[valid-type,misc]
        """Canonical object registry with bitemporal state.

        *valid_time* — when the object was true in the real world.
        *transaction_time* — when the system recorded the fact.
        """

        __tablename__ = "object_registry"

        id = Column(Integer, primary_key=True, autoincrement=True)
        object_id = Column(String(128), nullable=False, index=True)
        object_version = Column(Integer, nullable=False, default=1)
        schema_id = Column(String(256), nullable=False)
        schema_version = Column(String(32), nullable=False)
        owner_id = Column(String(128), nullable=False)
        workspace_id = Column(String(128), nullable=False, index=True)
        subject_domain = Column(String(32), nullable=False)
        artifact_type = Column(String(64), nullable=False)
        cognitive_role = Column(String(32), nullable=False)
        workflow_status = Column(String(32), nullable=False)
        epistemic_status = Column(String(32), nullable=False)
        lifecycle_status = Column(String(32), nullable=False)
        storage_tier = Column(String(16), nullable=False)
        presentation = Column(String(32), nullable=False)
        security_class = Column(String(32), nullable=False)

        # Bitemporal — valid time (real-world truth interval)
        valid_from = Column(DateTime(timezone=True), nullable=True)
        valid_to = Column(DateTime(timezone=True), nullable=True)

        # Bitemporal — transaction time (system record interval)
        recorded_at = Column(
            DateTime(timezone=True), nullable=False, server_default=func.now()
        )
        superseded_at = Column(DateTime(timezone=True), nullable=True)

        created_by = Column(String(256), nullable=False)
        trace_id = Column(String(256), nullable=True)
        content_sha256 = Column(String(64), nullable=False)
        payload = Column(JSON, nullable=False)
        tags = Column(JSON, nullable=False, server_default="[]")

    class _LedgerEventRow(Base):  # type: ignore[valid-type,misc]
        """Immutable append-only event ledger."""

        __tablename__ = "event_ledger"

        id = Column(Integer, primary_key=True, autoincrement=True)
        event_id = Column(String(128), nullable=False, unique=True)
        event_type = Column(String(64), nullable=False)
        object_id = Column(String(128), nullable=False, index=True)
        object_version = Column(Integer, nullable=False)
        principal = Column(String(256), nullable=False)
        workspace_id = Column(String(128), nullable=False)
        payload = Column(JSON, nullable=False, server_default="{}")
        integrity_hash = Column(String(64), nullable=False)
        tx_time = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
        parent_event_id = Column(String(128), nullable=True)

    class _OutboxEntryRow(Base):  # type: ignore[valid-type,misc]
        """Transactional outbox entry for async consumers."""

        __tablename__ = "outbox_entries"

        id = Column(Integer, primary_key=True, autoincrement=True)
        entry_id = Column(String(128), nullable=False, unique=True)
        topic = Column(String(128), nullable=False)
        payload = Column(JSON, nullable=False)
        headers = Column(JSON, nullable=False, server_default="{}")
        created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
        claimed_at = Column(DateTime(timezone=True), nullable=True)
        claimed_by = Column(String(128), nullable=True)
        retry_count = Column(Integer, nullable=False, default=0)
        processed_at = Column(DateTime(timezone=True), nullable=True)
        error_message = Column(String(512), nullable=True)
else:
    Base = object  # type: ignore[misc,assignment]


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------


class ConcurrencyError(RuntimeError):
    """Raised when optimistic concurrency check fails."""


class LedgerValidationError(Exception):
    """Raised when a ledger event fails validation."""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _generate_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _row_to_record(row: Any) -> ObjectRecord:
    """Convert an _ObjectRegistryRow to an ObjectRecord."""
    return ObjectRecord(
        object_id=row.object_id,
        schema_id=row.schema_id,
        schema_version=row.schema_version,
        owner_id=row.owner_id,
        workspace_id=row.workspace_id,
        subject_domain=row.subject_domain,
        artifact_type=row.artifact_type,
        cognitive_role=row.cognitive_role,
        workflow_status=row.workflow_status,
        epistemic_status=row.epistemic_status,
        lifecycle_status=row.lifecycle_status,
        storage_tier=row.storage_tier,
        presentation=row.presentation,
        security_class=row.security_class,
        payload=row.payload,
        tags=row.tags or [],
        created_by=row.created_by,
        trace_id=row.trace_id,
        version=row.object_version,
    )


# ------------------------------------------------------------------
# Store
# ------------------------------------------------------------------


class DurableObjectStore:
    """PostgreSQL-backed durable object store.

    This is the **canonical authority** in the bitemporal architecture.
    All consequential state transitions flow through here.

    Requires ``sqlalchemy`` and a running PostgreSQL instance.
    Connection string is read from ``database_url`` parameter or
    ``ANIMUS_DATABASE_URL`` environment variable.
    """

    def __init__(
        self,
        database_url: str | None = None,
        owner_id: str = "owner-default",
        workspace_id: str = "ws-default",
    ):
        if not _HAS_SQLALCHEMY:
            raise RuntimeError(
                "DurableObjectStore requires sqlalchemy. "
                "Install: pip install animus[postgres]"
            )

        self.database_url = database_url or os.getenv("ANIMUS_DATABASE_URL")
        if not self.database_url:
            raise RuntimeError(
                "DurableObjectStore requires database_url or ANIMUS_DATABASE_URL."
            )

        self.owner_id = owner_id
        self.workspace_id = workspace_id
        self._engine: Engine = create_engine(self.database_url, echo=False)
        self._session_factory = sessionmaker(bind=self._engine)
        logger.debug(f"DurableObjectStore initialized: {self.database_url}")

    def create_tables(self) -> None:
        """Create all tables. Call once during setup."""
        Base.metadata.create_all(self._engine)
        logger.info("Created bitemporal tables (object_registry, event_ledger, outbox_entries)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_integrity_hash(self, record: ObjectRecord) -> str:
        payload = {
            "object_id": record.object_id,
            "version": record.version,
            "schema_id": record.schema_id,
            "schema_version": record.schema_version,
            "owner_id": record.owner_id,
            "workspace_id": record.workspace_id,
            "artifact_type": record.artifact_type,
            "payload": record.payload,
            "tags": record.tags,
        }
        return _sha256(payload)

    def _write_ledger_event(
        self,
        session: Session,
        event_type: str,
        record: ObjectRecord,
        parent_event_id: str | None = None,
    ) -> str:
        """Append an immutable event to the ledger. Returns event_id."""
        event_id = _generate_id("evt")
        payload = {
            "artifact_type": record.artifact_type,
            "schema_id": record.schema_id,
            "tags": record.tags,
        }
        now = _now_utc()
        integrity = _sha256({
            "event_id": event_id,
            "event_type": event_type,
            "object_id": record.object_id,
            "version": record.version,
            "payload": payload,
        })

        row = _LedgerEventRow(
            event_id=event_id,
            event_type=event_type,
            object_id=record.object_id,
            object_version=record.version,
            principal=record.created_by,
            workspace_id=record.workspace_id,
            payload=payload,
            integrity_hash=integrity,
            tx_time=now,
            parent_event_id=parent_event_id,
        )
        session.add(row)
        session.flush()
        return event_id

    def _enqueue_outbox(
        self,
        session: Session,
        topic: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> str:
        """Enqueue an outbox entry for async processing. Returns entry_id."""
        entry_id = _generate_id("obx")
        headers_dict = headers or {}

        row = _OutboxEntryRow(
            entry_id=entry_id,
            topic=topic,
            payload=payload,
            headers=headers_dict,
        )
        session.add(row)
        session.flush()
        return entry_id

    # ------------------------------------------------------------------
    # CRUD + ledger
    # ------------------------------------------------------------------

    def store(self, record: ObjectRecord) -> tuple[str, str]:
        """Store a new object. Returns (object_id, event_id).

        Writes:
        1. Object registry row (current projection)
        2. Ledger event (immutable)
        3. Outbox entry (async projection update)
        """
        record.version = 1
        integrity = self._compute_integrity_hash(record)
        now = _now_utc()

        with self._session_factory() as session:
            row = _ObjectRegistryRow(
                object_id=record.object_id,
                object_version=record.version,
                schema_id=record.schema_id,
                schema_version=record.schema_version,
                owner_id=record.owner_id,
                workspace_id=record.workspace_id,
                subject_domain=record.subject_domain,
                artifact_type=record.artifact_type,
                cognitive_role=record.cognitive_role,
                workflow_status=record.workflow_status,
                epistemic_status=record.epistemic_status,
                lifecycle_status=record.lifecycle_status,
                storage_tier=record.storage_tier,
                presentation=record.presentation,
                security_class=record.security_class,
                valid_from=now,
                created_by=record.created_by,
                trace_id=record.trace_id,
                content_sha256=integrity,
                payload=record.payload,
                tags=record.tags,
            )
            session.add(row)
            event_id = self._write_ledger_event(session, EventType.CREATED.value, record)
            self._enqueue_outbox(
                session,
                topic="object.created",
                payload={
                    "object_id": record.object_id,
                    "version": record.version,
                    "event_id": event_id,
                },
            )
            session.commit()
            logger.debug(f"Stored object {record.object_id} with event {event_id}")
            return record.object_id, event_id

    def update(
        self, record: ObjectRecord, expected_version: int | None = None
    ) -> tuple[bool, str]:
        """Update an object with optimistic concurrency control.

        If *expected_version* is provided and does not match the current
        version, raises :exc:`ConcurrencyError`.
        """
        with self._session_factory() as session:
            current = session.execute(
                select(_ObjectRegistryRow).where(
                    _ObjectRegistryRow.object_id == record.object_id,
                    _ObjectRegistryRow.superseded_at.is_(None),
                )
            ).scalar_one_or_none()

            if not current:
                return False, ""

            if expected_version is not None and current.object_version != expected_version:
                raise ConcurrencyError(
                    f"Expected version {expected_version}, found {current.object_version}"
                )

            # Supersede old version
            now = _now_utc()
            current.superseded_at = now
            current.valid_to = now
            current.lifecycle_status = LifecycleStatus.SUPERSEDED.value

            # Increment version
            record.version = current.object_version + 1
            integrity = self._compute_integrity_hash(record)
            self._validate_object_version(record, integrity, valid_from=now, recorded_at=now)

            new_row = _ObjectRegistryRow(
                object_id=record.object_id,
                object_version=record.version,
                schema_id=record.schema_id,
                schema_version=record.schema_version,
                owner_id=record.owner_id,
                workspace_id=record.workspace_id,
                subject_domain=record.subject_domain,
                artifact_type=record.artifact_type,
                cognitive_role=record.cognitive_role,
                workflow_status=record.workflow_status,
                epistemic_status=record.epistemic_status,
                lifecycle_status=record.lifecycle_status,
                storage_tier=record.storage_tier,
                presentation=record.presentation,
                security_class=record.security_class,
                valid_from=now,
                created_by=record.created_by,
                trace_id=record.trace_id,
                content_sha256=integrity,
                payload=record.payload,
                tags=record.tags,
            )
            session.add(new_row)
            event_id = self._write_ledger_event(
                session, EventType.UPDATED.value, record, parent_event_id=None
            )
            self._enqueue_outbox(
                session,
                topic="object.updated",
                payload={
                    "object_id": record.object_id,
                    "version": record.version,
                    "event_id": event_id,
                },
            )
            session.commit()
            logger.debug(f"Updated object {record.object_id} v{record.version} with event {event_id}")
            return True, event_id

    def _validate_object_version(
        self,
        record: ObjectRecord,
        integrity_hash: str,
        valid_from: datetime | None = None,
        recorded_at: datetime | None = None,
    ) -> None:
        """Validate an object record. Fail-closed: raises LedgerValidationError."""
        # Schema validation hook — reserved for future integration with
        # contracts.validator. Currently a no-op so the store works without
        # the contracts package.
        pass

    def retrieve(self, object_id: str) -> ObjectRecord | None:
        """Retrieve the current (non-superseded) version of an object."""
        with self._session_factory() as session:
            row = session.execute(
                select(_ObjectRegistryRow).where(
                    _ObjectRegistryRow.object_id == object_id,
                    _ObjectRegistryRow.superseded_at.is_(None),
                )
            ).scalar_one_or_none()

            if not row:
                return None
            return _row_to_record(row)

    def retrieve_version(self, object_id: str, version: int) -> ObjectRecord | None:
        """Retrieve a specific historical version of an object."""
        with self._session_factory() as session:
            row = session.execute(
                select(_ObjectRegistryRow).where(
                    _ObjectRegistryRow.object_id == object_id,
                    _ObjectRegistryRow.object_version == version,
                )
            ).scalar_one_or_none()

            if not row:
                return None
            return _row_to_record(row)

    def delete(self, object_id: str, principal: str = "animus") -> tuple[bool, str]:
        """Soft-delete an object (mark superseded + ledger event)."""
        with self._session_factory() as session:
            row = session.execute(
                select(_ObjectRegistryRow).where(
                    _ObjectRegistryRow.object_id == object_id,
                    _ObjectRegistryRow.superseded_at.is_(None),
                )
            ).scalar_one_or_none()

            if not row:
                return False, ""

            now = _now_utc()
            row.superseded_at = now
            row.valid_to = now
            row.lifecycle_status = LifecycleStatus.DELETED.value

            record = self.retrieve(object_id)
            if record:
                record.created_by = principal
                event_id = self._write_ledger_event(session, EventType.DELETED.value, record)
                self._enqueue_outbox(
                    session,
                    topic="object.deleted",
                    payload={"object_id": object_id, "event_id": event_id},
                )
                session.commit()
                logger.debug(f"Deleted object {object_id} with event {event_id}")
                return True, event_id
            return False, ""

    def list_current(self, artifact_type: str | None = None) -> list[ObjectRecord]:
        """List all current (non-superseded) objects."""
        with self._session_factory() as session:
            stmt = select(_ObjectRegistryRow).where(_ObjectRegistryRow.superseded_at.is_(None))
            if artifact_type:
                stmt = stmt.where(_ObjectRegistryRow.artifact_type == artifact_type)

            rows = session.execute(stmt).scalars().all()
            return [_row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Bitemporal queries
    # ------------------------------------------------------------------

    def as_of_valid_time(self, object_id: str, vt: datetime) -> ObjectRecord | None:
        """Retrieve the version valid at *vt* (valid time)."""
        with self._session_factory() as session:
            row = session.execute(
                select(_ObjectRegistryRow).where(
                    _ObjectRegistryRow.object_id == object_id,
                    _ObjectRegistryRow.valid_from <= vt,
                    _ObjectRegistryRow.valid_to.is_(None)
                    | (_ObjectRegistryRow.valid_to > vt),
                )
            ).scalar_one_or_none()

            if not row:
                return None
            return _row_to_record(row)

    def as_of_transaction_time(self, object_id: str, tt: datetime) -> ObjectRecord | None:
        """Retrieve the version as known at *tt* (transaction time)."""
        with self._session_factory() as session:
            row = session.execute(
                select(_ObjectRegistryRow).where(
                    _ObjectRegistryRow.object_id == object_id,
                    _ObjectRegistryRow.recorded_at <= tt,
                    _ObjectRegistryRow.superseded_at.is_(None)
                    | (_ObjectRegistryRow.superseded_at > tt),
                )
            ).scalar_one_or_none()

            if not row:
                return None
            return _row_to_record(row)

    # ------------------------------------------------------------------
    # Ledger access
    # ------------------------------------------------------------------

    def get_ledger_events(self, object_id: str) -> list[dict[str, Any]]:
        """Retrieve all ledger events for an object, ordered by tx_time."""
        with self._session_factory() as session:
            rows = session.execute(
                select(_LedgerEventRow)
                .where(_LedgerEventRow.object_id == object_id)
                .order_by(_LedgerEventRow.tx_time)
            ).scalars().all()

            return [
                {
                    "event_id": r.event_id,
                    "event_type": r.event_type,
                    "object_id": r.object_id,
                    "object_version": r.object_version,
                    "principal": r.principal,
                    "workspace_id": r.workspace_id,
                    "payload": r.payload,
                    "integrity_hash": r.integrity_hash,
                    "tx_time": r.tx_time.isoformat() if r.tx_time else None,
                    "parent_event_id": r.parent_event_id,
                }
                for r in rows
            ]

    def verify_integrity(self, event_id: str) -> bool:
        """Verify the integrity hash of a ledger event."""
        with self._session_factory() as session:
            row = session.execute(
                select(_LedgerEventRow).where(_LedgerEventRow.event_id == event_id)
            ).scalar_one_or_none()

            if not row:
                return False

            expected = _sha256({
                "event_id": row.event_id,
                "event_type": row.event_type,
                "object_id": row.object_id,
                "version": row.object_version,
                "payload": row.payload,
            })
            return row.integrity_hash == expected

    # ------------------------------------------------------------------
    # Outbox processing
    # ------------------------------------------------------------------

    def claim_outbox_entries(self, worker_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Claim unprocessed outbox entries for a worker."""
        with self._session_factory() as session:
            rows = session.execute(
                select(_OutboxEntryRow)
                .where(_OutboxEntryRow.processed_at.is_(None))
                .where(_OutboxEntryRow.claimed_at.is_(None))
                .limit(limit)
            ).scalars().all()

            now = _now_utc()
            entries = []
            for row in rows:
                row.claimed_at = now
                row.claimed_by = worker_id
                entries.append({
                    "entry_id": row.entry_id,
                    "topic": row.topic,
                    "payload": row.payload,
                    "headers": row.headers,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                })
            session.commit()
            return entries

    def acknowledge_outbox_entry(self, entry_id: str, error: str | None = None) -> bool:
        """Mark an outbox entry as processed (or failed)."""
        with self._session_factory() as session:
            row = session.execute(
                select(_OutboxEntryRow).where(_OutboxEntryRow.entry_id == entry_id)
            ).scalar_one_or_none()

            if not row:
                return False

            if error:
                row.retry_count += 1
                row.error_message = error
                row.claimed_at = None
                row.claimed_by = None
            else:
                row.processed_at = _now_utc()
                row.error_message = None

            session.commit()
            return True


__all__ = [
    "DurableObjectStore",
    "ObjectRecord",
    "ObjectType",
    "StorageTier",
    "SecurityClass",
    "EpistemicStatus",
    "LifecycleStatus",
    "EventType",
    "ConcurrencyError",
    "LedgerValidationError",
]
