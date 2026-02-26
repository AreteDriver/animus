"""Tests for edit proposal management."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.tools.models import ProposalStatus
from animus_forge.tools.proposals import ProposalManager, log_file_access


@pytest.fixture
def mock_backend():
    """Create a mock database backend."""
    backend = MagicMock()
    backend.adapt_query.side_effect = lambda q: q
    return backend


@pytest.fixture
def mock_validator(tmp_path):
    """Create a mock path validator."""
    validator = MagicMock()
    validator.get_project_root.return_value = tmp_path
    validator.validate_file_for_write.side_effect = lambda p: tmp_path / p
    return validator


@pytest.fixture
def manager(mock_backend, mock_validator):
    """Create a ProposalManager."""
    return ProposalManager(mock_backend, mock_validator)


class TestProposalManager:
    """Tests for ProposalManager."""

    def test_init(self, manager, mock_backend, mock_validator, tmp_path):
        assert manager.backend is mock_backend
        assert manager.validator is mock_validator
        assert manager.project_root == tmp_path

    def test_create_proposal(self, manager, mock_backend):
        proposal = manager.create_proposal(
            session_id="sess-1",
            file_path="test.py",
            new_content="new code",
            old_content="old code",
            description="Update test",
        )

        assert proposal.session_id == "sess-1"
        assert proposal.file_path == "test.py"
        assert proposal.new_content == "new code"
        assert proposal.old_content == "old code"
        assert proposal.description == "Update test"
        assert proposal.status == ProposalStatus.PENDING
        assert proposal.id  # UUID assigned
        mock_backend.execute.assert_called_once()

    def test_create_proposal_reads_existing_file(self, manager, tmp_path):
        """When old_content is None and file exists, it reads the file."""
        test_file = tmp_path / "existing.py"
        test_file.write_text("existing content", encoding="utf-8")
        manager.validator.validate_file_for_write.side_effect = lambda p: test_file

        proposal = manager.create_proposal(
            session_id="sess-1",
            file_path="existing.py",
            new_content="new content",
        )

        assert proposal.old_content == "existing content"

    def test_create_proposal_unreadable_file(self, manager, tmp_path):
        """When file exists but can't be read, old_content stays None."""
        test_file = tmp_path / "binary.dat"
        test_file.write_bytes(b"\x80\x81\x82")
        manager.validator.validate_file_for_write.side_effect = lambda p: test_file

        # Patch read_text to raise
        with patch.object(Path, "read_text", side_effect=OSError("unreadable")):
            proposal = manager.create_proposal(
                session_id="sess-1",
                file_path="binary.dat",
                new_content="new",
            )

        assert proposal.old_content is None

    def test_get_proposal_found(self, manager, mock_backend):
        now = datetime.now(UTC)
        mock_backend.fetchone.return_value = {
            "id": "p1",
            "session_id": "sess-1",
            "file_path": "test.py",
            "old_content": "old",
            "new_content": "new",
            "description": "desc",
            "status": "pending",
            "created_at": now.isoformat(),
            "applied_at": None,
            "error_message": None,
        }

        proposal = manager.get_proposal("p1")
        assert proposal is not None
        assert proposal.id == "p1"
        assert proposal.status == ProposalStatus.PENDING

    def test_get_proposal_not_found(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        assert manager.get_proposal("missing") is None

    def test_get_session_proposals_no_filter(self, manager, mock_backend):
        now = datetime.now(UTC)
        mock_backend.fetchall.return_value = [
            {
                "id": "p1",
                "session_id": "sess-1",
                "file_path": "a.py",
                "old_content": None,
                "new_content": "new",
                "description": "",
                "status": "pending",
                "created_at": now.isoformat(),
                "applied_at": None,
                "error_message": None,
            }
        ]

        proposals = manager.get_session_proposals("sess-1")
        assert len(proposals) == 1
        assert proposals[0].id == "p1"

    def test_get_session_proposals_with_filter(self, manager, mock_backend):
        mock_backend.fetchall.return_value = []
        proposals = manager.get_session_proposals("sess-1", status=ProposalStatus.APPLIED)
        assert proposals == []
        # Should have passed status.value as parameter
        call_args = mock_backend.fetchall.call_args
        assert ProposalStatus.APPLIED.value in call_args[0][1]

    def test_approve_proposal(self, manager, mock_backend, tmp_path):
        now = datetime.now(UTC)
        mock_backend.fetchone.return_value = {
            "id": "p1",
            "session_id": "sess-1",
            "file_path": "test.py",
            "old_content": "old",
            "new_content": "new content",
            "description": "update",
            "status": "pending",
            "created_at": now.isoformat(),
            "applied_at": None,
            "error_message": None,
        }

        proposal = manager.approve_proposal("p1")
        assert proposal.status == ProposalStatus.APPLIED
        assert proposal.applied_at is not None
        # File should be written
        written_file = tmp_path / "test.py"
        assert written_file.read_text() == "new content"

    def test_approve_proposal_with_existing_file_creates_backup(
        self, manager, mock_backend, tmp_path
    ):
        test_file = tmp_path / "exists.py"
        test_file.write_text("original", encoding="utf-8")
        manager.validator.validate_file_for_write.side_effect = lambda p: test_file

        now = datetime.now(UTC)
        mock_backend.fetchone.return_value = {
            "id": "p2",
            "session_id": "sess-1",
            "file_path": "exists.py",
            "old_content": "original",
            "new_content": "updated",
            "description": "",
            "status": "pending",
            "created_at": now.isoformat(),
            "applied_at": None,
            "error_message": None,
        }

        proposal = manager.approve_proposal("p2")
        assert proposal.status == ProposalStatus.APPLIED
        assert test_file.read_text() == "updated"
        assert test_file.with_suffix(".py.bak").exists()

    def test_approve_proposal_not_found(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        with pytest.raises(ValueError, match="not found"):
            manager.approve_proposal("missing")

    def test_approve_proposal_not_pending(self, manager, mock_backend):
        now = datetime.now(UTC)
        mock_backend.fetchone.return_value = {
            "id": "p1",
            "session_id": "sess-1",
            "file_path": "test.py",
            "old_content": None,
            "new_content": "new",
            "description": "",
            "status": "applied",
            "created_at": now.isoformat(),
            "applied_at": now.isoformat(),
            "error_message": None,
        }
        with pytest.raises(ValueError, match="not pending"):
            manager.approve_proposal("p1")

    def test_approve_proposal_write_fails(self, manager, mock_backend, tmp_path):
        now = datetime.now(UTC)
        mock_backend.fetchone.return_value = {
            "id": "p1",
            "session_id": "sess-1",
            "file_path": "test.py",
            "old_content": None,
            "new_content": "new",
            "description": "",
            "status": "pending",
            "created_at": now.isoformat(),
            "applied_at": None,
            "error_message": None,
        }
        # Make validation fail on second call (inside approve)
        manager.validator.validate_file_for_write.side_effect = OSError("disk full")

        with pytest.raises(OSError):
            manager.approve_proposal("p1")

        # Should have marked as failed in DB
        last_execute = mock_backend.execute.call_args
        assert ProposalStatus.FAILED.value in last_execute[0][1]

    def test_reject_proposal(self, manager, mock_backend):
        now = datetime.now(UTC)
        mock_backend.fetchone.return_value = {
            "id": "p1",
            "session_id": "sess-1",
            "file_path": "test.py",
            "old_content": None,
            "new_content": "new",
            "description": "",
            "status": "pending",
            "created_at": now.isoformat(),
            "applied_at": None,
            "error_message": None,
        }

        proposal = manager.reject_proposal("p1")
        assert proposal.status == ProposalStatus.REJECTED

    def test_reject_proposal_not_found(self, manager, mock_backend):
        mock_backend.fetchone.return_value = None
        with pytest.raises(ValueError, match="not found"):
            manager.reject_proposal("missing")

    def test_reject_proposal_not_pending(self, manager, mock_backend):
        now = datetime.now(UTC)
        mock_backend.fetchone.return_value = {
            "id": "p1",
            "session_id": "sess-1",
            "file_path": "test.py",
            "old_content": None,
            "new_content": "new",
            "description": "",
            "status": "rejected",
            "created_at": now.isoformat(),
            "applied_at": None,
            "error_message": None,
        }
        with pytest.raises(ValueError, match="not pending"):
            manager.reject_proposal("p1")

    def test_row_to_proposal_with_applied_at(self, manager):
        now = datetime.now(UTC)
        row = {
            "id": "p1",
            "session_id": "sess-1",
            "file_path": "test.py",
            "old_content": None,
            "new_content": "new",
            "description": None,
            "status": "applied",
            "created_at": now.isoformat(),
            "applied_at": now.isoformat(),
            "error_message": "some error",
        }
        proposal = manager._row_to_proposal(row)
        assert proposal.description == ""
        assert proposal.applied_at is not None
        assert proposal.error_message == "some error"


class TestLogFileAccess:
    """Tests for the log_file_access function."""

    def test_log_success(self):
        backend = MagicMock()
        backend.adapt_query.side_effect = lambda q: q

        log_file_access(
            backend=backend,
            session_id="sess-1",
            tool="read_file",
            file_path="test.py",
            operation="read",
            success=True,
        )

        backend.execute.assert_called_once()
        args = backend.execute.call_args[0][1]
        assert args[1] == "sess-1"
        assert args[2] == "read_file"
        assert args[3] == "test.py"
        assert args[4] == "read"
        assert args[5]  # timestamp
        assert args[6] is True
        assert args[7] is None

    def test_log_failure(self):
        backend = MagicMock()
        backend.adapt_query.side_effect = lambda q: q

        log_file_access(
            backend=backend,
            session_id="sess-1",
            tool="write_file",
            file_path="test.py",
            operation="write",
            success=False,
            error_message="Permission denied",
        )

        args = backend.execute.call_args[0][1]
        assert args[6] is False
        assert args[7] == "Permission denied"
