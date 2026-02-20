"""Skill deprecation lifecycle: flag → deprecate → retire."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from animus_forge.self_improve.approval import ApprovalGate, ApprovalStage
from animus_forge.state.backends import DatabaseBackend

from .models import DeprecationRecord, SkillMetrics

logger = logging.getLogger(__name__)


class SkillDeprecator:
    """Manages the deprecation lifecycle for underperforming skills.

    Three-stage lifecycle: flagged → deprecated → retired.
    Retirement requires approval via ``ApprovalGate``.

    Args:
        backend: A ``DatabaseBackend`` instance.
        approval_gate: An ``ApprovalGate`` for retirement approvals.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        approval_gate: ApprovalGate | None = None,
    ) -> None:
        self._backend = backend
        self._approval_gate = approval_gate
        self._lock = threading.Lock()

    def flag_for_deprecation(
        self,
        skill_name: str,
        reason: str,
        metrics: SkillMetrics | None = None,
    ) -> DeprecationRecord:
        """Flag a skill for deprecation review.

        Args:
            skill_name: Skill to flag.
            reason: Why the skill is being flagged.
            metrics: Current metrics at time of flagging.

        Returns:
            The ``DeprecationRecord`` that was created.
        """
        now = datetime.now(UTC).isoformat()
        success_rate = metrics.success_rate if metrics else 0.0
        invocations = metrics.total_invocations if metrics else 0

        query = (
            "INSERT OR REPLACE INTO skill_deprecations "
            "(skill_name, status, flagged_at, reason, "
            "success_rate_at_flag, invocations_at_flag) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        with self._lock:
            with self._backend.transaction():
                self._backend.execute(
                    query,
                    (skill_name, "flagged", now, reason, success_rate, invocations),
                )

        logger.info("Flagged skill %s for deprecation: %s", skill_name, reason)

        return DeprecationRecord(
            skill_name=skill_name,
            status="flagged",
            flagged_at=now,
            reason=reason,
            success_rate_at_flag=success_rate,
            invocations_at_flag=invocations,
        )

    def deprecate(self, skill_name: str) -> bool:
        """Move a flagged skill to deprecated status.

        Args:
            skill_name: Skill to deprecate.

        Returns:
            True if the status was updated.
        """
        now = datetime.now(UTC).isoformat()
        query = (
            "UPDATE skill_deprecations SET status = ?, deprecated_at = ? "
            "WHERE skill_name = ? AND status = ?"
        )
        with self._lock:
            with self._backend.transaction():
                self._backend.execute(query, ("deprecated", now, skill_name, "flagged"))

        # Check if any row was affected
        row = self._backend.fetchone(
            "SELECT status FROM skill_deprecations WHERE skill_name = ?",
            (skill_name,),
        )
        success = row is not None and row.get("status") == "deprecated"
        if success:
            logger.info("Deprecated skill %s", skill_name)
        return success

    def request_retirement(self, skill_name: str) -> str:
        """Request approval to retire a deprecated skill.

        Args:
            skill_name: Skill to retire.

        Returns:
            Approval request ID, or empty string if no gate configured.
        """
        if not self._approval_gate:
            return ""

        row = self._backend.fetchone(
            "SELECT * FROM skill_deprecations WHERE skill_name = ?",
            (skill_name,),
        )
        if not row or row.get("status") != "deprecated":
            logger.warning("Cannot retire %s: not in deprecated state", skill_name)
            return ""

        request = self._approval_gate.request_approval(
            stage=ApprovalStage.APPLY,
            title=f"Retire skill: {skill_name}",
            description=(
                f"Requesting retirement of skill '{skill_name}'. "
                f"Reason: {row.get('reason', 'N/A')}. "
                f"Success rate at flag: {row.get('success_rate_at_flag', 0):.0%}."
            ),
            details={"skill_name": skill_name, "record": dict(row)},
        )
        return request.id

    def retire(self, skill_name: str, approval_id: str = "") -> bool:
        """Retire a deprecated skill (requires approval).

        Args:
            skill_name: Skill to retire.
            approval_id: Approval request ID (verified if gate exists).

        Returns:
            True if the skill was retired.
        """
        if self._approval_gate and approval_id:
            if not self._approval_gate.is_approved(approval_id):
                logger.warning("Cannot retire %s: approval %s not granted", skill_name, approval_id)
                return False

        now = datetime.now(UTC).isoformat()
        query = (
            "UPDATE skill_deprecations SET status = ?, retired_at = ?, approval_id = ? "
            "WHERE skill_name = ? AND status = ?"
        )
        with self._lock:
            with self._backend.transaction():
                self._backend.execute(
                    query, ("retired", now, approval_id, skill_name, "deprecated")
                )

        row = self._backend.fetchone(
            "SELECT status FROM skill_deprecations WHERE skill_name = ?",
            (skill_name,),
        )
        success = row is not None and row.get("status") == "retired"
        if success:
            logger.info("Retired skill %s", skill_name)
        return success

    def get_flagged_skills(self) -> list[DeprecationRecord]:
        """Get all skills in 'flagged' status.

        Returns:
            List of ``DeprecationRecord`` objects.
        """
        return self._query_by_status("flagged")

    def get_deprecated_skills(self) -> list[DeprecationRecord]:
        """Get all skills in 'deprecated' status.

        Returns:
            List of ``DeprecationRecord`` objects.
        """
        return self._query_by_status("deprecated")

    def _query_by_status(self, status: str) -> list[DeprecationRecord]:
        query = "SELECT * FROM skill_deprecations WHERE status = ?"
        with self._lock:
            rows = self._backend.fetchall(query, (status,))

        return [
            DeprecationRecord(
                skill_name=str(row["skill_name"]),
                status=str(row["status"]),
                flagged_at=str(row.get("flagged_at", "")),
                deprecated_at=str(row.get("deprecated_at", "") or ""),
                retired_at=str(row.get("retired_at", "") or ""),
                reason=str(row.get("reason", "")),
                success_rate_at_flag=float(row.get("success_rate_at_flag", 0.0)),
                invocations_at_flag=int(row.get("invocations_at_flag", 0)),
                replacement_skill=str(row.get("replacement_skill", "") or ""),
                approval_id=str(row.get("approval_id", "") or ""),
            )
            for row in rows
        ]
