"""Identity patch approval gate — Forge → Core mutation path.

Enforces the P4 constraint: "Forge may NEVER write to Core directly."
Provides a controlled propose → approve → reject gate for identity changes.

All mutations are staged in-memory + persisted to identity_patches.json,
with every action appended to forge_audit.jsonl for audit trail.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IdentityPatch:
    """A proposed identity change from Forge."""

    proposed_changes: dict[str, Any]
    reasoning: str
    impact_assessment: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    patch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: str = "pending"
    rejection_reason: str | None = None


class IdentityPatchGate:
    """Approval gate for Forge identity change proposals.

    Stages patches in-memory and persists to identity_patches.json.
    Every propose/approve/reject action is appended to forge_audit.jsonl.

    Optionally delegates approval to Bootstrap's IdentityProposalManager
    when available, otherwise writes to a staging file.
    """

    def __init__(
        self,
        patches_path: Path | None = None,
        audit_log_path: Path | None = None,
        proposal_manager: Any | None = None,
    ):
        """Initialize the identity patch gate.

        Args:
            patches_path: Path to persist pending patches JSON.
                          Defaults to forge/identity_patches.json.
            audit_log_path: Path to append audit events.
                            Defaults to forge/forge_audit.jsonl.
            proposal_manager: Optional Bootstrap IdentityProposalManager
                              for delegating approvals.
        """
        self._patches_path = patches_path or Path("forge/identity_patches.json")
        self._audit_log_path = audit_log_path or Path("forge/forge_audit.jsonl")
        self._proposal_manager = proposal_manager
        self._pending: dict[str, IdentityPatch] = {}
        self._load_patches()

    def propose(self, patch: IdentityPatch) -> str:
        """Stage an identity patch for approval.

        Args:
            patch: The proposed identity change.

        Returns:
            The patch_id for later approve/reject calls.
        """
        patch.status = "pending"
        self._pending[patch.patch_id] = patch
        self._persist_patches()

        self._emit_audit(
            "identity_patch_proposed",
            {
                "patch_id": patch.patch_id,
                "reasoning": patch.reasoning,
                "impact_assessment": patch.impact_assessment,
                "changes_keys": list(patch.proposed_changes.keys()),
            },
        )

        logger.info("Identity patch proposed: %s", patch.patch_id)
        return patch.patch_id

    def approve(self, patch_id: str) -> bool:
        """Approve a pending identity patch.

        Applies via Bootstrap's IdentityProposalManager if available,
        otherwise writes proposed changes to a staging file.

        Args:
            patch_id: The patch to approve.

        Returns:
            True if approved successfully, False if patch not found.
        """
        patch = self._pending.get(patch_id)
        if patch is None:
            logger.warning("Approve failed: patch %s not found", patch_id)
            return False

        # Attempt to apply via Bootstrap if available
        applied = False
        if self._proposal_manager is not None:
            try:
                for file_key, content in patch.proposed_changes.items():
                    self._proposal_manager.create(
                        file=file_key,
                        proposed_content=str(content),
                        reason=patch.reasoning,
                    )
                applied = True
            except Exception:
                logger.exception(
                    "Failed to apply patch %s via proposal manager", patch_id
                )

        # Fallback: write to staging file
        if not applied:
            staging_path = self._patches_path.parent / f"staged_{patch_id}.json"
            staging_path.parent.mkdir(parents=True, exist_ok=True)
            staging_path.write_text(
                json.dumps(
                    {
                        "patch_id": patch_id,
                        "proposed_changes": patch.proposed_changes,
                        "reasoning": patch.reasoning,
                        "applied_at": datetime.now(UTC).isoformat(),
                    },
                    default=str,
                )
            )

        patch.status = "approved"
        del self._pending[patch_id]
        self._persist_patches()

        self._emit_audit(
            "identity_patch_approved",
            {
                "patch_id": patch_id,
                "via_proposal_manager": applied,
            },
        )

        logger.info("Identity patch approved: %s", patch_id)
        return True

    def reject(self, patch_id: str, reason: str = "") -> bool:
        """Reject a pending identity patch.

        Args:
            patch_id: The patch to reject.
            reason: Why the patch was rejected.

        Returns:
            True if rejected successfully, False if patch not found.
        """
        patch = self._pending.get(patch_id)
        if patch is None:
            logger.warning("Reject failed: patch %s not found", patch_id)
            return False

        patch.status = "rejected"
        patch.rejection_reason = reason
        del self._pending[patch_id]
        self._persist_patches()

        self._emit_audit(
            "identity_patch_rejected",
            {
                "patch_id": patch_id,
                "reason": reason,
            },
        )

        logger.info("Identity patch rejected: %s — %s", patch_id, reason)
        return True

    def list_pending(self) -> list[IdentityPatch]:
        """Return all pending identity patches.

        Returns:
            List of pending IdentityPatch objects.
        """
        return [p for p in self._pending.values() if p.status == "pending"]

    def _load_patches(self) -> None:
        """Load pending patches from disk."""
        if not self._patches_path.exists():
            return
        try:
            data = json.loads(self._patches_path.read_text())
            for entry in data:
                patch = IdentityPatch(
                    proposed_changes=entry.get("proposed_changes", {}),
                    reasoning=entry.get("reasoning", ""),
                    impact_assessment=entry.get("impact_assessment", ""),
                    created_at=datetime.fromisoformat(entry["created_at"]),
                    patch_id=entry["patch_id"],
                    status=entry.get("status", "pending"),
                )
                if patch.status == "pending":
                    self._pending[patch.patch_id] = patch
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to load identity patches from %s", self._patches_path)

    def _persist_patches(self) -> None:
        """Write pending patches to disk."""
        self._patches_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "patch_id": p.patch_id,
                "proposed_changes": p.proposed_changes,
                "reasoning": p.reasoning,
                "impact_assessment": p.impact_assessment,
                "created_at": p.created_at.isoformat(),
                "status": p.status,
            }
            for p in self._pending.values()
        ]
        self._patches_path.write_text(json.dumps(data, default=str, indent=2))

    def _emit_audit(self, event_type: str, data: dict[str, Any]) -> None:
        """Append to audit log."""
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            **data,
        }
        with open(self._audit_log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
