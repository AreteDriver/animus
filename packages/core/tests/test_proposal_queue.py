"""Tests for ProposalQueue approval lifecycle."""

from pathlib import Path

import pytest

from animus.citizens.proposal import (
    EvidenceItem,
    ImprovementProposal,
    ProposalConfidence,
    ProposalStatus,
    RiskAssessment,
)
from animus.citizens.proposal_queue import ProposalQueue, QueuedProposal, Transition


@pytest.fixture
def sample_proposal():
    return ImprovementProposal(
        id="PROP-001",
        title="Fix flaky tests",
        problem="Tests fail intermittently on CI",
        evidence=[
            EvidenceItem(
                source="ci_log",
                description="test_auth.py::test_login failed 3 of last 5 runs",
            )
        ],
        root_cause="Race condition in async cleanup",
        recommendation="Use asyncio.gather with explicit timeout",
        confidence_score=0.85,
        confidence_label=ProposalConfidence.HIGH,
        estimated_effort_hours=4.0,
        affected_components=["tests/", "api/auth"],
        status=ProposalStatus.DRAFT,
    )


@pytest.fixture
def queue():
    return ProposalQueue()


class TestQueuedProposal:
    def test_queued_proposal_current_status(self, sample_proposal):
        qp = QueuedProposal(proposal=sample_proposal)
        assert qp.current_status == ProposalStatus.DRAFT
        assert qp.age_hours >= 0

    def test_queued_proposal_transition_updates_status(self, sample_proposal):
        qp = QueuedProposal(proposal=sample_proposal)
        qp.transitions.append(
            Transition(
                from_status=ProposalStatus.DRAFT,
                to_status=ProposalStatus.SUBMITTED,
                actor="citizen",
            )
        )
        assert qp.current_status == ProposalStatus.SUBMITTED

    def test_queued_proposal_to_dict_roundtrip(self, sample_proposal):
        qp = QueuedProposal(
            proposal=sample_proposal,
            priority=2,
            tags=["urgent", "test"],
        )
        qp.transitions.append(
            Transition(
                from_status=ProposalStatus.DRAFT,
                to_status=ProposalStatus.SUBMITTED,
                actor="citizen",
                reason="Auto-submitted",
            )
        )
        d = qp.to_dict()
        restored = QueuedProposal.from_dict(d)
        assert restored.proposal.id == qp.proposal.id
        assert restored.current_status == qp.current_status
        assert restored.priority == 2
        assert restored.tags == ["urgent", "test"]


class TestProposalQueueSubmit:
    def test_submit_adds_to_queue(self, queue, sample_proposal):
        qp = queue.submit(sample_proposal, priority=1, tags=["architect"])
        assert qp.current_status == ProposalStatus.SUBMITTED
        assert len(qp.transitions) == 1
        assert qp.transitions[0].from_status == ProposalStatus.DRAFT
        assert qp.transitions[0].to_status == ProposalStatus.SUBMITTED
        assert qp.priority == 1
        assert qp.tags == ["architect"]

    def test_submit_sets_proposal_status(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        assert sample_proposal.status == ProposalStatus.SUBMITTED

    def test_get_retrieves_submitted(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        retrieved = queue.get("PROP-001")
        assert retrieved is not None
        assert retrieved.proposal.id == "PROP-001"

    def test_get_returns_none_for_missing(self, queue):
        assert queue.get("MISSING") is None


class TestProposalQueueApprove:
    def test_approve_transitions_status(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        result = queue.approve("PROP-001", actor="human", reason="LGTM")
        assert result is not None
        assert result.current_status == ProposalStatus.APPROVED
        assert result.proposal.approved_by == "human"
        assert result.proposal.approved_at is not None
        assert len(result.transitions) == 2

    def test_approve_returns_none_for_missing(self, queue):
        assert queue.approve("MISSING", actor="human") is None

    def test_approve_noop_if_not_submitted(self, queue, sample_proposal):
        queue._proposals["PROP-001"] = QueuedProposal(
            proposal=sample_proposal,
            transitions=[
                Transition(
                    from_status=ProposalStatus.DRAFT,
                    to_status=ProposalStatus.REJECTED,
                    actor="human",
                )
            ],
        )
        result = queue.approve("PROP-001", actor="human")
        assert result.current_status == ProposalStatus.REJECTED


class TestProposalQueueReject:
    def test_reject_transitions_status(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        result = queue.reject("PROP-001", actor="human", reason="Not aligned with roadmap")
        assert result.current_status == ProposalStatus.REJECTED
        assert len(result.transitions) == 2

    def test_reject_idempotent_for_complete(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        queue.approve("PROP-001", actor="human")
        queue.commission("PROP-001", actor="forge")
        queue.complete("PROP-001", actor="forge")
        result = queue.reject("PROP-001", actor="human")
        assert result.current_status == ProposalStatus.COMPLETE


class TestProposalQueueCommission:
    def test_commission_from_approved(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        queue.approve("PROP-001", actor="human")
        result = queue.commission("PROP-001", actor="forge")
        assert result.current_status == ProposalStatus.COMMISSIONED
        assert len(result.transitions) == 3

    def test_commission_noop_if_not_approved(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        result = queue.commission("PROP-001", actor="forge")
        assert result.current_status == ProposalStatus.SUBMITTED


class TestProposalQueueComplete:
    def test_complete_commissioned(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        queue.approve("PROP-001", actor="human")
        queue.commission("PROP-001", actor="forge")
        result = queue.complete("PROP-001", actor="forge")
        assert result.current_status == ProposalStatus.COMPLETE


class TestProposalQueueQueries:
    def test_list_by_status(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        assert len(queue.list_by_status(ProposalStatus.SUBMITTED)) == 1
        assert len(queue.list_by_status(ProposalStatus.APPROVED)) == 0

    def test_list_pending(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        assert len(queue.list_pending()) == 1
        queue.approve("PROP-001", actor="human")
        assert len(queue.list_pending()) == 0

    def test_list_approved(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        assert len(queue.list_approved()) == 0
        queue.approve("PROP-001", actor="human")
        assert len(queue.list_approved()) == 1

    def test_list_commissioned(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        queue.approve("PROP-001", actor="human")
        queue.commission("PROP-001", actor="forge")
        assert len(queue.list_commissioned()) == 1

    def test_list_completed(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        queue.approve("PROP-001", actor="human")
        queue.commission("PROP-001", actor="forge")
        queue.complete("PROP-001", actor="forge")
        assert len(queue.list_completed()) == 1

    def test_list_rejected(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        queue.reject("PROP-001", actor="human")
        assert len(queue.list_rejected()) == 1


class TestProposalQueueBacklog:
    def test_backlog_excludes_complete_and_rejected(self, queue, sample_proposal):
        queue.submit(sample_proposal, priority=1)
        assert len(queue.get_backlog()) == 1

        queue.approve("PROP-001", actor="human")
        assert len(queue.get_backlog()) == 1

        queue.complete("PROP-001", actor="forge")
        assert len(queue.get_backlog()) == 0

    def test_backlog_sorted_by_priority(self, queue):
        p1 = ImprovementProposal(
            id="P-01", title="High", problem="x",
            status=ProposalStatus.DRAFT,
        )
        p2 = ImprovementProposal(
            id="P-02", title="Low", problem="y",
            status=ProposalStatus.DRAFT,
        )
        queue.submit(p1, priority=1)
        queue.submit(p2, priority=3)
        backlog = queue.get_backlog()
        assert backlog[0].proposal.id == "P-01"
        assert backlog[1].proposal.id == "P-02"


class TestProposalQueueStats:
    def test_stats(self, queue, sample_proposal):
        assert queue.stats() == {
            "total": 0, "pending": 0, "approved": 0,
            "commissioned": 0, "complete": 0, "rejected": 0,
        }
        queue.submit(sample_proposal)
        s = queue.stats()
        assert s["total"] == 1
        assert s["pending"] == 1

        queue.approve("PROP-001", actor="human")
        s = queue.stats()
        assert s["pending"] == 0
        assert s["approved"] == 1

        queue.commission("PROP-001", actor="forge")
        s = queue.stats()
        assert s["approved"] == 0
        assert s["commissioned"] == 1

    def test_repr(self, queue, sample_proposal):
        queue.submit(sample_proposal)
        r = repr(queue)
        assert "ProposalQueue" in r
        assert "total=1" in r


class TestProposalQueuePersistence:
    def test_load_from_memory_without_memory(self, queue, monkeypatch):
        import animus.citizens.proposal_queue as pq_module
        monkeypatch.setattr(
            pq_module, "_DEFAULT_SQLITE_PATH", Path("/nonexistent/animus/proposal_queue.db")
        )
        assert queue.load_from_memory() == 0

    def test_persist_without_memory_is_noop(self, queue, sample_proposal):
        # Should not raise even without memory layer
        queue.submit(sample_proposal)
        queue.approve("PROP-001", actor="human")

    def test_file_persistence_roundtrip(self, tmp_path, sample_proposal):
        path = tmp_path / "proposal_queue.json"
        queue = ProposalQueue(storage_path=str(path))
        queue.submit(sample_proposal, priority=1, tags=["test"])
        queue.approve("PROP-001", actor="human")

        # Verify file was written
        assert path.exists()

        # Load into a fresh queue
        queue2 = ProposalQueue(storage_path=str(path))
        loaded = queue2.load_from_memory()
        assert loaded == 1
        qp = queue2.get("PROP-001")
        assert qp is not None
        assert qp.current_status.value == "approved"
        assert qp.priority == 1
        assert qp.tags == ["test"]

    def test_file_persistence_over_memory(self, tmp_path, sample_proposal):
        """File takes precedence over memory fallback."""
        path = tmp_path / "proposal_queue.json"
        queue = ProposalQueue(storage_path=str(path))
        queue.submit(sample_proposal)
        queue.approve("PROP-001", actor="human")

        # Create a fresh queue pointing to the same file
        queue2 = ProposalQueue(storage_path=str(path))
        queue2.load_from_memory()
        assert queue2.get("PROP-001") is not None
        assert queue2.stats()["approved"] == 1
