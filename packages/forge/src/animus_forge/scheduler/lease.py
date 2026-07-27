"""Lease manager — atomic task claim with heartbeats and expiry recovery.

Every task assigned to a citizen gets a lease.  The lease has a TTL; if the
worker dies or hangs, the lease expires and the task returns to READY for
another worker to claim.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_leases (
    lease_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE,
    mission_id TEXT NOT NULL,
    citizen_role TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    heartbeat_at TEXT,
    outcome TEXT
);

CREATE INDEX IF NOT EXISTS idx_leases_status ON task_leases(status);
CREATE INDEX IF NOT EXISTS idx_leases_expires ON task_leases(expires_at);
CREATE INDEX IF NOT EXISTS idx_leases_mission ON task_leases(mission_id);
"""


class LeaseStatus(str):
    """Lease lifecycle states."""

    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"


class Lease:
    """Runtime representation of a task lease."""

    def __init__(
        self,
        lease_id: str,
        task_id: str,
        mission_id: str,
        citizen_role: str,
        worker_id: str,
        acquired_at: datetime,
        expires_at: datetime,
        status: str = LeaseStatus.ACTIVE,
        heartbeat_at: datetime | None = None,
        outcome: str | None = None,
    ):
        self.lease_id = lease_id
        self.task_id = task_id
        self.mission_id = mission_id
        self.citizen_role = citizen_role
        self.worker_id = worker_id
        self.acquired_at = acquired_at
        self.expires_at = expires_at
        self.status = status
        self.heartbeat_at = heartbeat_at
        self.outcome = outcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "mission_id": self.mission_id,
            "citizen_role": self.citizen_role,
            "worker_id": self.worker_id,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "outcome": self.outcome,
        }


class LeaseManager:
    """Atomic lease operations backed by DatabaseBackend.

    Args:
        backend: Shared ``DatabaseBackend``.
        default_ttl_seconds: Default lease TTL if not overridden per-acquire.
    """

    def __init__(self, backend: DatabaseBackend, default_ttl_seconds: int = 300):
        self._backend = backend
        self.default_ttl_seconds = default_ttl_seconds
        self._init_schema()

    def _init_schema(self) -> None:
        with self._backend.transaction():
            self._backend.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(
        self,
        task_id: str,
        mission_id: str,
        citizen_role: str,
        worker_id: str,
        ttl_seconds: int | None = None,
    ) -> Lease | None:
        """Atomically acquire a lease for *task_id*.

        Returns:
            The new ``Lease`` if successful, or ``None`` if the task already
            has an active lease (another worker claimed it first).
        """
        ttl = ttl_seconds or self.default_ttl_seconds
        now = datetime.now()
        expires = now + timedelta(seconds=ttl)
        lease_id = str(uuid.uuid4())

        try:
            with self._backend.transaction():
                # Atomic insert — UNIQUE constraint on task_id prevents duplicates
                self._backend.execute(
                    """
                    INSERT INTO task_leases
                        (lease_id, task_id, mission_id, citizen_role, worker_id,
                         acquired_at, expires_at, status, heartbeat_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lease_id,
                        task_id,
                        mission_id,
                        citizen_role,
                        worker_id,
                        now.isoformat(),
                        expires.isoformat(),
                        LeaseStatus.ACTIVE,
                        now.isoformat(),
                    ),
                )
        except Exception:
            logger.debug("Lease acquire failed for task %s (race or constraint)", task_id)
            return None

        logger.info(
            "Lease %s acquired for task %s by worker %s (expires %s)",
            lease_id,
            task_id,
            worker_id,
            expires,
        )
        return Lease(
            lease_id=lease_id,
            task_id=task_id,
            mission_id=mission_id,
            citizen_role=citizen_role,
            worker_id=worker_id,
            acquired_at=now,
            expires_at=expires,
            status=LeaseStatus.ACTIVE,
            heartbeat_at=now,
        )

    def renew(self, lease_id: str, ttl_seconds: int | None = None) -> Lease | None:
        """Renew an existing lease (heartbeat).

        Returns:
            The updated ``Lease``, or ``None`` if the lease is gone.
        """
        ttl = ttl_seconds or self.default_ttl_seconds
        now = datetime.now()
        expires = now + timedelta(seconds=ttl)

        with self._backend.transaction():
            row = self._backend.fetchone(
                "SELECT * FROM task_leases WHERE lease_id = ? AND status = ?",
                (lease_id, LeaseStatus.ACTIVE),
            )
            if not row:
                return None

            self._backend.execute(
                """
                UPDATE task_leases
                SET expires_at = ?, heartbeat_at = ?
                WHERE lease_id = ?
                """,
                (expires.isoformat(), now.isoformat(), lease_id),
            )

        lease = self._parse_row(row)
        lease.expires_at = expires
        lease.heartbeat_at = now
        logger.debug("Lease %s renewed, expires %s", lease_id, expires)
        return lease

    def release(self, lease_id: str, outcome: str = "completed") -> Lease | None:
        """Release a lease (task completed or failed).

        Returns:
            The released ``Lease``, or ``None`` if not found.
        """
        with self._backend.transaction():
            row = self._backend.fetchone(
                "SELECT * FROM task_leases WHERE lease_id = ?",
                (lease_id,),
            )
            if not row:
                return None

            self._backend.execute(
                """
                UPDATE task_leases
                SET status = ?, outcome = ?
                WHERE lease_id = ?
                """,
                (LeaseStatus.RELEASED, outcome, lease_id),
            )

        lease = self._parse_row(row)
        lease.status = LeaseStatus.RELEASED
        lease.outcome = outcome
        logger.info("Lease %s released with outcome %s", lease_id, outcome)
        return lease

    def recover_expired(self, as_of: datetime | None = None) -> list[str]:
        """Find expired leases and mark them EXPIRED.

        Returns:
            List of ``task_id`` strings that were recovered.
        """
        now = as_of or datetime.now()
        with self._backend.transaction():
            rows = self._backend.fetchall(
                "SELECT lease_id, task_id FROM task_leases WHERE status = ? AND expires_at < ?",
                (LeaseStatus.ACTIVE, now.isoformat()),
            )
            recovered = [r["task_id"] for r in rows]
            for r in rows:
                self._backend.execute(
                    "UPDATE task_leases SET status = ? WHERE lease_id = ?",
                    (LeaseStatus.EXPIRED, r["lease_id"]),
                )

        if recovered:
            logger.info("Recovered %d expired lease(s): %s", len(recovered), recovered)
        return recovered

    def get_active_leases(self) -> list[Lease]:
        """Return all currently active leases."""
        rows = self._backend.fetchall(
            "SELECT * FROM task_leases WHERE status = ? ORDER BY acquired_at",
            (LeaseStatus.ACTIVE,),
        )
        return [self._parse_row(r) for r in rows]

    def get_lease_for_task(self, task_id: str) -> Lease | None:
        """Fetch the lease (active or otherwise) for a task."""
        row = self._backend.fetchone(
            "SELECT * FROM task_leases WHERE task_id = ?",
            (task_id,),
        )
        return self._parse_row(row) if row else None

    def get_lease(self, lease_id: str) -> Lease | None:
        """Fetch a lease by ID."""
        row = self._backend.fetchone(
            "SELECT * FROM task_leases WHERE lease_id = ?",
            (lease_id,),
        )
        return self._parse_row(row) if row else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_row(row: dict) -> Lease:
        return Lease(
            lease_id=row["lease_id"],
            task_id=row["task_id"],
            mission_id=row["mission_id"],
            citizen_role=row["citizen_role"],
            worker_id=row["worker_id"],
            acquired_at=datetime.fromisoformat(row["acquired_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            status=row["status"],
            heartbeat_at=datetime.fromisoformat(row["heartbeat_at"]) if row.get("heartbeat_at") else None,
            outcome=row.get("outcome"),
        )
