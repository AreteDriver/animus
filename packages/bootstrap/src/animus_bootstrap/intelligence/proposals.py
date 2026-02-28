"""Typed identity proposal manager — clean abstraction over ImprovementStore."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    """A typed identity change proposal."""

    id: int
    file: str
    current: str
    proposed: str
    diff: str
    reason: str
    created_at: str
    status: str = "pending"
    resolved_at: str | None = None
    rejection_reason: str | None = None

    @staticmethod
    def from_store_dict(d: dict, current_content: str = "") -> Proposal:
        """Create a Proposal from an ImprovementStore dict."""
        area = d.get("area", "")
        filename = area.split(":", 1)[1] if ":" in area else area
        proposed = d.get("patch", "")

        # Reconstruct diff if we have both sides
        diff = ""
        if current_content or proposed:
            diff = _compute_diff(filename, current_content, proposed)

        # Map store statuses to proposal statuses
        status_map = {"proposed": "pending", "approved": "approved", "rejected": "rejected"}
        raw_status = d.get("status", "proposed")

        return Proposal(
            id=d.get("id", 0),
            file=filename,
            current=current_content,
            proposed=proposed,
            diff=diff,
            reason=d.get("description", ""),
            created_at=d.get("timestamp", ""),
            status=status_map.get(raw_status, raw_status),
            resolved_at=d.get("applied_at"),
        )


def _compute_diff(filename: str, current: str, proposed: str) -> str:
    """Compute a unified diff between current and proposed content."""
    current_lines = current.splitlines(keepends=True)
    proposed_lines = proposed.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            current_lines,
            proposed_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


@dataclass
class _ManagerState:
    """Internal state container — avoids mutable default args."""

    improvement_store: object | None = None
    identity_manager: object | None = None


class IdentityProposalManager:
    """Typed layer over ImprovementStore for identity-specific proposals.

    Wraps the general-purpose ImprovementStore with identity-focused methods,
    typed Proposal objects, and automatic diff computation.

    Args:
        improvement_store: The ImprovementStore instance for persistence.
        identity_manager: The IdentityFileManager for reading/writing files.
    """

    def __init__(self, improvement_store: object, identity_manager: object) -> None:
        self._store = improvement_store
        self._mgr = identity_manager

    def create(self, file: str, proposed_content: str, reason: str) -> Proposal:
        """Create a new identity change proposal.

        Computes a unified diff, persists via ImprovementStore, and returns
        a typed Proposal object.

        Args:
            file: Identity filename (e.g., "CONTEXT.md").
            proposed_content: The proposed new content for the file.
            reason: Why this change is being proposed.

        Returns:
            A Proposal with status "pending".

        Raises:
            PermissionError: If file is CORE_VALUES.md.
        """
        if file == "CORE_VALUES.md":
            msg = (
                "CORE_VALUES.md is immutable and cannot be modified by Animus. "
                "Edit manually or via the dashboard."
            )
            raise PermissionError(msg)

        current = self._mgr.read(file)  # type: ignore[union-attr]
        diff = _compute_diff(file, current, proposed_content)
        now = datetime.now(UTC).isoformat()

        store_dict = {
            "area": f"identity:{file}",
            "description": f"Proposed change to {file}: {reason}",
            "status": "proposed",
            "timestamp": now,
            "analysis": (
                f"Current length: {len(current)} chars, proposed: {len(proposed_content)} chars"
            ),
            "patch": proposed_content,
        }
        proposal_id = self._store.save(store_dict)  # type: ignore[union-attr]

        return Proposal(
            id=proposal_id,
            file=file,
            current=current,
            proposed=proposed_content,
            diff=diff,
            reason=reason,
            created_at=now,
            status="pending",
        )

    def list_pending(self) -> list[Proposal]:
        """Return all pending identity proposals."""
        all_proposals = self._store.list_all()  # type: ignore[union-attr]
        results = []
        for p in all_proposals:
            area = p.get("area", "")
            if area.startswith("identity:") and p["status"] == "proposed":
                filename = area.split(":", 1)[1]
                current = self._mgr.read(filename)  # type: ignore[union-attr]
                results.append(Proposal.from_store_dict(p, current))
        return results

    def approve(self, proposal_id: int) -> Proposal:
        """Approve a proposal — write the file and update status.

        Args:
            proposal_id: The proposal ID to approve.

        Returns:
            The updated Proposal with status "approved".

        Raises:
            ValueError: If proposal not found or not an identity proposal.
            PermissionError: If the target file is locked.
        """
        raw = self._store.get(proposal_id)  # type: ignore[union-attr]
        if raw is None:
            msg = f"Proposal #{proposal_id} not found."
            raise ValueError(msg)

        area = raw.get("area", "")
        if not area.startswith("identity:"):
            msg = f"Proposal #{proposal_id} is not an identity proposal."
            raise ValueError(msg)

        filename = area.split(":", 1)[1]
        content = raw.get("patch", "")

        # Write via identity manager (respects LOCKED_FILES)
        self._mgr.write(filename, content)  # type: ignore[union-attr]

        now = datetime.now(UTC).isoformat()
        self._store.update_status(proposal_id, "approved", now)  # type: ignore[union-attr]

        current = self._mgr.read(filename)  # type: ignore[union-attr]
        return Proposal.from_store_dict(
            {**raw, "status": "approved", "applied_at": now},
            current,
        )

    def reject(self, proposal_id: int, reason: str = "") -> Proposal:
        """Reject a proposal — update status and log rejection reason.

        Args:
            proposal_id: The proposal ID to reject.
            reason: Why the proposal was rejected.

        Returns:
            The updated Proposal with status "rejected".

        Raises:
            ValueError: If proposal not found.
        """
        raw = self._store.get(proposal_id)  # type: ignore[union-attr]
        if raw is None:
            msg = f"Proposal #{proposal_id} not found."
            raise ValueError(msg)

        now = datetime.now(UTC).isoformat()
        self._store.update_status(proposal_id, "rejected", now)  # type: ignore[union-attr]

        if reason:
            analysis = raw.get("analysis", "") or ""
            rejection_note = f"{analysis}\nRejection reason: {reason}".strip()
            self._store.update_analysis(proposal_id, rejection_note)  # type: ignore[union-attr]

        logger.info("Proposal #%d rejected", proposal_id)

        proposal = Proposal.from_store_dict(
            {**raw, "status": "rejected", "applied_at": now},
        )
        proposal.rejection_reason = reason
        return proposal

    def history(self) -> list[Proposal]:
        """Return all identity proposals sorted by date (newest first)."""
        all_proposals = self._store.list_all()  # type: ignore[union-attr]
        results = []
        for p in all_proposals:
            if p.get("area", "").startswith("identity:"):
                results.append(Proposal.from_store_dict(p))
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results

    def get(self, proposal_id: int) -> Proposal | None:
        """Get a single proposal by ID, or None if not found."""
        raw = self._store.get(proposal_id)  # type: ignore[union-attr]
        if raw is None:
            return None
        area = raw.get("area", "")
        if not area.startswith("identity:"):
            return None
        filename = area.split(":", 1)[1]
        current = self._mgr.read(filename)  # type: ignore[union-attr]
        return Proposal.from_store_dict(raw, current)
