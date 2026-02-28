"""Tests for SQLite-backed ApprovalGate persistence."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from animus_forge.self_improve.approval import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalStage,
    ApprovalStatus,
)
from animus_forge.state.backends import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_backend(tmp_path: Path) -> SQLiteBackend:
    """Create a SQLiteBackend in a temp directory."""
    return SQLiteBackend(str(tmp_path / "test.db"))


@pytest.fixture()
def gate(db_backend: SQLiteBackend) -> ApprovalGate:
    """Create a persistent ApprovalGate."""
    return ApprovalGate(backend=db_backend)


@pytest.fixture()
def memory_gate() -> ApprovalGate:
    """Create an in-memory ApprovalGate (no backend)."""
    return ApprovalGate()


# ===========================================================================
# In-Memory Mode (backward compatibility)
# ===========================================================================


class TestApprovalGateInMemory:
    """Tests for ApprovalGate without database backend."""

    def test_request_approval(self, memory_gate: ApprovalGate):
        """Request creates an approval in pending state."""
        req = memory_gate.request_approval(
            stage=ApprovalStage.PLAN,
            title="Test Plan",
            description="Test description",
            details={"key": "value"},
        )
        assert req.status == ApprovalStatus.PENDING
        assert req.title == "Test Plan"
        assert req.stage == ApprovalStage.PLAN
        assert req.details == {"key": "value"}

    def test_approve(self, memory_gate: ApprovalGate):
        """Approve moves request from pending to approved."""
        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        result = memory_gate.approve(req.id, approved_by="tester", reason="LGTM")
        assert result is not None
        assert result.status == ApprovalStatus.APPROVED
        assert result.decided_by == "tester"
        assert result.reason == "LGTM"
        assert result.decided_at is not None

    def test_reject(self, memory_gate: ApprovalGate):
        """Reject moves request from pending to rejected."""
        req = memory_gate.request_approval(stage=ApprovalStage.APPLY, title="T", description="D")
        result = memory_gate.reject(req.id, rejected_by="reviewer", reason="Too risky")
        assert result is not None
        assert result.status == ApprovalStatus.REJECTED
        assert result.decided_by == "reviewer"

    def test_approve_not_found(self, memory_gate: ApprovalGate):
        """Approve returns None for unknown ID."""
        assert memory_gate.approve("nonexistent") is None

    def test_reject_not_found(self, memory_gate: ApprovalGate):
        """Reject returns None for unknown ID."""
        assert memory_gate.reject("nonexistent") is None

    def test_is_approved(self, memory_gate: ApprovalGate):
        """is_approved checks history."""
        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        assert memory_gate.is_approved(req.id) is False
        memory_gate.approve(req.id)
        assert memory_gate.is_approved(req.id) is True

    def test_is_approved_rejected(self, memory_gate: ApprovalGate):
        """is_approved returns False for rejected requests."""
        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        memory_gate.reject(req.id)
        assert memory_gate.is_approved(req.id) is False

    def test_get_pending(self, memory_gate: ApprovalGate):
        """get_pending returns only pending requests."""
        r1 = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        r2 = memory_gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")
        memory_gate.approve(r1.id)

        pending = memory_gate.get_pending()
        assert len(pending) == 1
        assert pending[0].id == r2.id

    def test_get_pending_filter_stage(self, memory_gate: ApprovalGate):
        """get_pending filters by stage."""
        memory_gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        memory_gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")

        plan_pending = memory_gate.get_pending(stage=ApprovalStage.PLAN)
        assert len(plan_pending) == 1
        assert plan_pending[0].stage == ApprovalStage.PLAN

    def test_auto_approve_for_testing(self, memory_gate: ApprovalGate):
        """auto_approve_for_testing marks as approved."""
        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        memory_gate.auto_approve_for_testing(req.id)
        assert memory_gate.is_approved(req.id) is True

    def test_get_history(self, memory_gate: ApprovalGate):
        """get_history returns decided requests."""
        r1 = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        r2 = memory_gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")
        memory_gate.approve(r1.id)
        memory_gate.reject(r2.id)

        history = memory_gate.get_history()
        assert len(history) == 2

    def test_get_history_filter_stage(self, memory_gate: ApprovalGate):
        """get_history filters by stage."""
        r1 = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        r2 = memory_gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")
        memory_gate.approve(r1.id)
        memory_gate.approve(r2.id)

        plan_history = memory_gate.get_history(stage=ApprovalStage.PLAN)
        assert len(plan_history) == 1


# ===========================================================================
# Persistent Mode (SQLite backend)
# ===========================================================================


class TestApprovalGatePersistent:
    """Tests for ApprovalGate with SQLite persistence."""

    def test_request_persisted(self, gate: ApprovalGate, db_backend: SQLiteBackend):
        """Request is written to database."""
        req = gate.request_approval(
            stage=ApprovalStage.PLAN,
            title="Persist Test",
            description="Should be in DB",
            details={"files": ["a.py"]},
        )

        row = db_backend.fetchone("SELECT * FROM self_improve_approvals WHERE id = ?", (req.id,))
        assert row is not None
        assert row["title"] == "Persist Test"
        assert row["status"] == "pending"
        assert row["stage"] == "plan"

    def test_approve_persisted(self, gate: ApprovalGate, db_backend: SQLiteBackend):
        """Approval decision is persisted."""
        req = gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        gate.approve(req.id, approved_by="admin")

        row = db_backend.fetchone("SELECT * FROM self_improve_approvals WHERE id = ?", (req.id,))
        assert row["status"] == "approved"
        assert row["decided_by"] == "admin"
        assert row["decided_at"] is not None

    def test_reject_persisted(self, gate: ApprovalGate, db_backend: SQLiteBackend):
        """Rejection decision is persisted."""
        req = gate.request_approval(stage=ApprovalStage.APPLY, title="T", description="D")
        gate.reject(req.id, rejected_by="reviewer", reason="Nope")

        row = db_backend.fetchone("SELECT * FROM self_improve_approvals WHERE id = ?", (req.id,))
        assert row["status"] == "rejected"
        assert row["decided_by"] == "reviewer"
        assert row["reason"] == "Nope"

    def test_get_pending_from_db(self, gate: ApprovalGate):
        """get_pending queries database."""
        gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        r2 = gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")
        gate.approve(r2.id)

        pending = gate.get_pending()
        assert len(pending) == 1
        assert pending[0].stage == ApprovalStage.PLAN

    def test_get_pending_filter_by_stage(self, gate: ApprovalGate):
        """get_pending with stage filter queries correct rows."""
        gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")

        apply_pending = gate.get_pending(stage=ApprovalStage.APPLY)
        assert len(apply_pending) == 1

    def test_is_approved_from_db(self, gate: ApprovalGate, db_backend: SQLiteBackend):
        """is_approved checks database for unknown in-memory IDs."""
        req = gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        gate.approve(req.id)

        # Create new gate instance sharing same backend (simulates restart)
        new_gate = ApprovalGate(backend=db_backend)
        assert new_gate.is_approved(req.id) is True

    def test_approve_from_db_when_not_in_memory(
        self, gate: ApprovalGate, db_backend: SQLiteBackend
    ):
        """Can approve requests loaded from DB even if not in memory."""
        req = gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        req_id = req.id

        # Create new gate (simulates restart — in-memory dict is empty)
        new_gate = ApprovalGate(backend=db_backend)
        result = new_gate.approve(req_id)
        assert result is not None
        assert result.status == ApprovalStatus.APPROVED

    def test_reject_from_db_when_not_in_memory(self, gate: ApprovalGate, db_backend: SQLiteBackend):
        """Can reject requests loaded from DB even if not in memory."""
        req = gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        req_id = req.id

        new_gate = ApprovalGate(backend=db_backend)
        result = new_gate.reject(req_id, reason="No")
        assert result is not None
        assert result.status == ApprovalStatus.REJECTED

    def test_get_history_from_db(self, gate: ApprovalGate):
        """get_history reads from database."""
        r1 = gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        r2 = gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")
        gate.approve(r1.id)
        gate.reject(r2.id)

        history = gate.get_history()
        assert len(history) == 2

    def test_get_history_filter_stage_from_db(self, gate: ApprovalGate):
        """get_history with stage filter reads from database."""
        r1 = gate.request_approval(stage=ApprovalStage.PLAN, title="A", description="D")
        r2 = gate.request_approval(stage=ApprovalStage.APPLY, title="B", description="D")
        gate.approve(r1.id)
        gate.approve(r2.id)

        plan_history = gate.get_history(stage=ApprovalStage.PLAN)
        assert len(plan_history) == 1


# ===========================================================================
# Async wait_for_approval tests
# ===========================================================================


class TestWaitForApproval:
    """Tests for async wait_for_approval polling."""

    def test_wait_already_decided(self, memory_gate: ApprovalGate):
        """Returns immediately if request was already decided."""
        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")
        memory_gate.approve(req.id)

        status = asyncio.run(memory_gate.wait_for_approval(req, timeout=5.0, poll_interval=0.1))
        assert status == ApprovalStatus.APPROVED

    def test_wait_expires_on_timeout(self, memory_gate: ApprovalGate):
        """Returns EXPIRED when timeout is reached."""
        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")

        status = asyncio.run(memory_gate.wait_for_approval(req, timeout=0.2, poll_interval=0.1))
        assert status == ApprovalStatus.EXPIRED
        assert req.status == ApprovalStatus.EXPIRED

    def test_wait_picks_up_external_approval(self, memory_gate: ApprovalGate):
        """Detects approval set externally during polling."""

        async def approve_after_delay():
            await asyncio.sleep(0.1)
            memory_gate.approve(req.id, approved_by="external")

        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")

        async def run():
            # Start approval in background, wait in foreground
            task = asyncio.create_task(approve_after_delay())
            status = await memory_gate.wait_for_approval(req, timeout=5.0, poll_interval=0.05)
            task.result()  # Propagate errors
            return status

        status = asyncio.run(run())
        assert status == ApprovalStatus.APPROVED

    def test_wait_picks_up_external_rejection(self, memory_gate: ApprovalGate):
        """Detects rejection set externally during polling."""

        async def reject_after_delay():
            await asyncio.sleep(0.1)
            memory_gate.reject(req.id, rejected_by="reviewer")

        req = memory_gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")

        async def run():
            task = asyncio.create_task(reject_after_delay())
            status = await memory_gate.wait_for_approval(req, timeout=5.0, poll_interval=0.05)
            task.result()  # Propagate errors
            return status

        status = asyncio.run(run())
        assert status == ApprovalStatus.REJECTED

    def test_wait_db_external_decision(self, db_backend: SQLiteBackend):
        """Detects approval written directly to database."""
        gate = ApprovalGate(backend=db_backend)
        req = gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")

        async def approve_via_db():
            await asyncio.sleep(0.1)
            # Simulate external process updating the DB directly
            with db_backend.transaction():
                db_backend.execute(
                    "UPDATE self_improve_approvals SET status = 'approved', decided_by = 'dashboard', decided_at = datetime('now') WHERE id = ?",
                    (req.id,),
                )

        async def run():
            task = asyncio.create_task(approve_via_db())
            status = await gate.wait_for_approval(req, timeout=5.0, poll_interval=0.05)
            task.result()  # Propagate errors
            return status

        status = asyncio.run(run())
        assert status == ApprovalStatus.APPROVED

    def test_wait_expired_persisted(self, db_backend: SQLiteBackend):
        """Expired status is written to database."""
        gate = ApprovalGate(backend=db_backend)
        req = gate.request_approval(stage=ApprovalStage.PLAN, title="T", description="D")

        status = asyncio.run(gate.wait_for_approval(req, timeout=0.2, poll_interval=0.1))
        assert status == ApprovalStatus.EXPIRED

        row = db_backend.fetchone(
            "SELECT status FROM self_improve_approvals WHERE id = ?", (req.id,)
        )
        assert row["status"] == "expired"


# ===========================================================================
# Edge cases and dataclass tests
# ===========================================================================


class TestApprovalEdgeCases:
    """Edge case and dataclass tests."""

    def test_approval_status_enum(self):
        """ApprovalStatus enum values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.EXPIRED.value == "expired"

    def test_approval_stage_enum(self):
        """ApprovalStage enum values."""
        assert ApprovalStage.PLAN.value == "plan"
        assert ApprovalStage.APPLY.value == "apply"
        assert ApprovalStage.MERGE.value == "merge"

    def test_approval_request_defaults(self):
        """ApprovalRequest has correct defaults."""
        req = ApprovalRequest(
            id="test",
            stage=ApprovalStage.PLAN,
            title="Title",
            description="Desc",
            details={},
        )
        assert req.status == ApprovalStatus.PENDING
        assert req.decided_at is None
        assert req.decided_by is None
        assert req.reason is None

    def test_request_approval_empty_details(self, memory_gate: ApprovalGate):
        """details defaults to empty dict when None."""
        req = memory_gate.request_approval(
            stage=ApprovalStage.PLAN, title="T", description="D", details=None
        )
        assert req.details == {}
