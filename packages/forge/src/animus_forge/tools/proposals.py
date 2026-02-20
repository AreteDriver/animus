"""Edit proposal management for filesystem tools.

Proposals allow agents to suggest file changes that require user approval
before being applied.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from animus_forge.tools.models import EditProposal, ProposalStatus
from animus_forge.tools.safety import PathValidator

if TYPE_CHECKING:
    from animus_forge.state.backends import DatabaseBackend


class ProposalManager:
    """Manages edit proposals for a chat session.

    Proposals are stored in the database and can be approved/rejected by users.
    """

    def __init__(
        self,
        backend: DatabaseBackend,
        validator: PathValidator,
    ):
        """Initialize the proposal manager.

        Args:
            backend: Database backend for persistence.
            validator: Path validator for security checks.
        """
        self.backend = backend
        self.validator = validator
        self.project_root = validator.get_project_root()

    def create_proposal(
        self,
        session_id: str,
        file_path: str,
        new_content: str,
        old_content: str | None = None,
        description: str = "",
    ) -> EditProposal:
        """Create a new edit proposal.

        Args:
            session_id: Chat session ID.
            file_path: Relative path to the file.
            new_content: Proposed new content.
            old_content: Original content (None for new files).
            description: Description of the change.

        Returns:
            Created EditProposal.

        Raises:
            SecurityError: If path fails validation.
        """
        # Validate the path is writable
        resolved = self.validator.validate_file_for_write(file_path)
        rel_path = str(resolved.relative_to(self.project_root))

        # If file exists and old_content not provided, read it
        if old_content is None and resolved.is_file():
            try:
                old_content = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass  # Non-critical fallback: file unreadable, proceed with None old_content

        proposal_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        # Store in database
        query = self.backend.adapt_query("""
            INSERT INTO edit_proposals
            (id, session_id, file_path, old_content, new_content, description, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self.backend.execute(
            query,
            (
                proposal_id,
                session_id,
                rel_path,
                old_content,
                new_content,
                description,
                ProposalStatus.PENDING.value,
                now.isoformat(),
            ),
        )

        return EditProposal(
            id=proposal_id,
            session_id=session_id,
            file_path=rel_path,
            old_content=old_content,
            new_content=new_content,
            description=description,
            status=ProposalStatus.PENDING,
            created_at=now,
        )

    def get_proposal(self, proposal_id: str) -> EditProposal | None:
        """Get a proposal by ID.

        Args:
            proposal_id: Proposal ID.

        Returns:
            EditProposal if found, None otherwise.
        """
        query = self.backend.adapt_query("""
            SELECT id, session_id, file_path, old_content, new_content,
                   description, status, created_at, applied_at, error_message
            FROM edit_proposals WHERE id = ?
        """)
        row = self.backend.fetchone(query, (proposal_id,))
        if not row:
            return None

        return self._row_to_proposal(row)

    def get_session_proposals(
        self,
        session_id: str,
        status: ProposalStatus | None = None,
    ) -> list[EditProposal]:
        """Get all proposals for a session.

        Args:
            session_id: Chat session ID.
            status: Optional status filter.

        Returns:
            List of proposals.
        """
        if status:
            query = self.backend.adapt_query("""
                SELECT id, session_id, file_path, old_content, new_content,
                       description, status, created_at, applied_at, error_message
                FROM edit_proposals
                WHERE session_id = ? AND status = ?
                ORDER BY created_at DESC
            """)
            rows = self.backend.fetchall(query, (session_id, status.value))
        else:
            query = self.backend.adapt_query("""
                SELECT id, session_id, file_path, old_content, new_content,
                       description, status, created_at, applied_at, error_message
                FROM edit_proposals
                WHERE session_id = ?
                ORDER BY created_at DESC
            """)
            rows = self.backend.fetchall(query, (session_id,))

        return [self._row_to_proposal(row) for row in rows]

    def approve_proposal(self, proposal_id: str) -> EditProposal:
        """Approve and apply a proposal.

        Args:
            proposal_id: Proposal ID.

        Returns:
            Updated proposal.

        Raises:
            ValueError: If proposal not found or not pending.
            SecurityError: If file path validation fails.
        """
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(f"Proposal is not pending: {proposal.status.value}")

        try:
            # Validate path again before writing
            resolved = self.validator.validate_file_for_write(proposal.file_path)

            # Create backup if file exists
            if resolved.is_file():
                backup_path = resolved.with_suffix(resolved.suffix + ".bak")
                shutil.copy2(resolved, backup_path)

            # Write new content
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(proposal.new_content, encoding="utf-8")

            # Update status
            now = datetime.now(UTC)
            query = self.backend.adapt_query("""
                UPDATE edit_proposals
                SET status = ?, applied_at = ?
                WHERE id = ?
            """)
            self.backend.execute(
                query,
                (ProposalStatus.APPLIED.value, now.isoformat(), proposal_id),
            )

            proposal.status = ProposalStatus.APPLIED
            proposal.applied_at = now

        except Exception as e:
            # Mark as failed
            error_msg = str(e)
            query = self.backend.adapt_query("""
                UPDATE edit_proposals
                SET status = ?, error_message = ?
                WHERE id = ?
            """)
            self.backend.execute(
                query,
                (ProposalStatus.FAILED.value, error_msg, proposal_id),
            )
            proposal.status = ProposalStatus.FAILED
            proposal.error_message = error_msg
            raise

        return proposal

    def reject_proposal(self, proposal_id: str) -> EditProposal:
        """Reject a proposal.

        Args:
            proposal_id: Proposal ID.

        Returns:
            Updated proposal.

        Raises:
            ValueError: If proposal not found or not pending.
        """
        proposal = self.get_proposal(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal not found: {proposal_id}")

        if proposal.status != ProposalStatus.PENDING:
            raise ValueError(f"Proposal is not pending: {proposal.status.value}")

        query = self.backend.adapt_query("""
            UPDATE edit_proposals SET status = ? WHERE id = ?
        """)
        self.backend.execute(query, (ProposalStatus.REJECTED.value, proposal_id))

        proposal.status = ProposalStatus.REJECTED
        return proposal

    def _row_to_proposal(self, row: dict) -> EditProposal:
        """Convert a database row to an EditProposal."""
        created_at = row.get("created_at")
        applied_at = row.get("applied_at")

        return EditProposal(
            id=row["id"],
            session_id=row["session_id"],
            file_path=row["file_path"],
            old_content=row.get("old_content"),
            new_content=row["new_content"],
            description=row.get("description") or "",
            status=ProposalStatus(row["status"]),
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(),
            applied_at=datetime.fromisoformat(applied_at) if applied_at else None,
            error_message=row.get("error_message"),
        )


def log_file_access(
    backend: DatabaseBackend,
    session_id: str,
    tool: str,
    file_path: str,
    operation: str,
    success: bool = True,
    error_message: str | None = None,
) -> None:
    """Log a file access for audit purposes.

    Args:
        backend: Database backend.
        session_id: Chat session ID.
        tool: Tool name used.
        file_path: Path accessed.
        operation: Operation type (read, list, search).
        success: Whether operation succeeded.
        error_message: Error message if failed.
    """
    log_id = str(uuid.uuid4())
    query = backend.adapt_query("""
        INSERT INTO file_access_log
        (id, session_id, tool, file_path, operation, timestamp, success, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """)
    backend.execute(
        query,
        (
            log_id,
            session_id,
            tool,
            file_path,
            operation,
            datetime.now(UTC).isoformat(),
            success,
            error_message,
        ),
    )
