"""Tests for convergent.triumvirate voting engine."""

from __future__ import annotations

import pytest
from convergent.coordination_config import CoordinationConfig
from convergent.protocol import (
    AgentIdentity,
    ConsensusRequest,
    DecisionOutcome,
    QuorumLevel,
    Vote,
    VoteChoice,
)
from convergent.score_store import ScoreStore
from convergent.scoring import PhiScorer
from convergent.triumvirate import Triumvirate


@pytest.fixture()
def scorer() -> PhiScorer:
    return PhiScorer(ScoreStore(":memory:"))


@pytest.fixture()
def config() -> CoordinationConfig:
    return CoordinationConfig()


@pytest.fixture()
def tri(scorer: PhiScorer, config: CoordinationConfig) -> Triumvirate:
    return Triumvirate(scorer, config)


def _agent(name: str, phi: float = 0.5) -> AgentIdentity:
    return AgentIdentity(name, "reviewer", "claude:sonnet", phi_score=phi)


def _seed_score(scorer: PhiScorer, agent_id: str, phi: float) -> None:
    """Seed a stored phi score so apply_vote_weight uses it."""
    scorer._store.save_score(agent_id, "reviewer", phi)


def _vote(
    agent: AgentIdentity,
    choice: VoteChoice,
    confidence: float = 0.8,
    reasoning: str = "test",
) -> Vote:
    return Vote(agent=agent, choice=choice, confidence=confidence, reasoning=reasoning)


# --- Request creation ---


class TestCreateRequest:
    def test_creates_request(self, tri: Triumvirate) -> None:
        req = tri.create_request("task-1", "Merge PR?", "context here")
        assert isinstance(req, ConsensusRequest)
        assert req.task_id == "task-1"
        assert req.question == "Merge PR?"
        assert req.quorum is QuorumLevel.MAJORITY  # config default

    def test_custom_quorum(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.UNANIMOUS)
        assert req.quorum is QuorumLevel.UNANIMOUS

    def test_with_artifacts(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", artifacts=["src/main.py"])
        assert req.artifacts == ["src/main.py"]

    def test_unique_request_ids(self, tri: Triumvirate) -> None:
        r1 = tri.create_request("t", "q", "c")
        r2 = tri.create_request("t", "q", "c")
        assert r1.request_id != r2.request_id


# --- Vote submission ---


class TestSubmitVote:
    def test_submit_vote(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        agent = _agent("a1", 0.8)
        vote = _vote(agent, VoteChoice.APPROVE)
        tri.submit_vote(req.request_id, vote)
        # No error means success

    def test_unknown_request(self, tri: Triumvirate) -> None:
        with pytest.raises(KeyError, match="Unknown request"):
            tri.submit_vote("nonexistent", _vote(_agent("a"), VoteChoice.APPROVE))

    def test_vote_after_decision(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.evaluate(req.request_id)
        with pytest.raises(ValueError, match="Decision already made"):
            tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.APPROVE))


# --- MAJORITY quorum ---


class TestMajorityQuorum:
    def test_majority_approve(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.MAJORITY)
        tri.submit_vote(req.request_id, _vote(_agent("a1", 0.8), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2", 0.7), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a3", 0.6), VoteChoice.REJECT))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED
        assert decision.total_weighted_approve > decision.total_weighted_reject

    def test_majority_reject(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.MAJORITY)
        tri.submit_vote(req.request_id, _vote(_agent("a1", 0.8), VoteChoice.REJECT))
        tri.submit_vote(req.request_id, _vote(_agent("a2", 0.7), VoteChoice.REJECT))
        tri.submit_vote(req.request_id, _vote(_agent("a3", 0.6), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.REJECTED

    def test_phi_weight_breaks_tie(self, tri: Triumvirate, scorer: PhiScorer) -> None:
        """High-trust approve vs low-trust reject: approve wins on weight."""
        _seed_score(scorer, "a1", 0.9)
        _seed_score(scorer, "a2", 0.3)
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.MAJORITY)
        # Same confidence, different phi scores
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a1", 0.9), VoteChoice.APPROVE, confidence=0.8),
        )
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a2", 0.3), VoteChoice.REJECT, confidence=0.8),
        )
        decision = tri.evaluate(req.request_id)
        # approve weight: 0.9*0.8=0.72, reject weight: 0.3*0.8=0.24
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_majority_tie_broken_by_highest_weight(
        self, tri: Triumvirate, scorer: PhiScorer
    ) -> None:
        """When weighted totals are exactly equal, highest individual vote wins."""
        _seed_score(scorer, "a1", 0.6)
        _seed_score(scorer, "a2", 0.3)
        _seed_score(scorer, "a3", 0.3)
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.MAJORITY)
        # Approve: 0.6*1.0 = 0.6, Reject: 0.3*1.0 + 0.3*1.0 = 0.6
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a1", 0.6), VoteChoice.APPROVE, confidence=1.0),
        )
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a2", 0.3), VoteChoice.REJECT, confidence=1.0),
        )
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a3", 0.3), VoteChoice.REJECT, confidence=1.0),
        )
        decision = tri.evaluate(req.request_id)
        # Tie: highest individual = a1 at 0.6 (approve)
        assert decision.outcome is DecisionOutcome.APPROVED


# --- ANY quorum ---


class TestAnyQuorum:
    def test_single_approve(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.ANY)
        tri.submit_vote(req.request_id, _vote(_agent("a1", 0.3), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2", 0.9), VoteChoice.REJECT))
        tri.submit_vote(req.request_id, _vote(_agent("a3", 0.9), VoteChoice.REJECT))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_no_approvals(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.ANY)
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.REJECT))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.ABSTAIN))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.REJECTED


# --- UNANIMOUS quorum ---


class TestUnanimousQuorum:
    def test_unanimous_approve(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.UNANIMOUS)
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a3"), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_unanimous_with_abstain(self, tri: Triumvirate) -> None:
        """Abstain doesn't count against — 2 approve + 1 abstain passes."""
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.UNANIMOUS)
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a3"), VoteChoice.ABSTAIN))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_unanimous_with_reject(self, tri: Triumvirate) -> None:
        """One reject blocks unanimous — 2 approve + 1 reject fails."""
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.UNANIMOUS)
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a3"), VoteChoice.REJECT))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.REJECTED

    def test_unanimous_all_abstain_deadlocks(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.UNANIMOUS)
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.ABSTAIN))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.ABSTAIN))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.DEADLOCK

    def test_unanimous_human_same_as_unanimous(self, tri: Triumvirate) -> None:
        """UNANIMOUS_HUMAN uses same quorum logic as UNANIMOUS."""
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.UNANIMOUS_HUMAN)
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED


# --- Escalation ---


class TestEscalation:
    def test_escalation_overrides(self, tri: Triumvirate) -> None:
        """Any escalate vote → ESCALATED regardless of other votes."""
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a3"), VoteChoice.ESCALATE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.ESCALATED

    def test_single_escalation(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.ESCALATE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.ESCALATED


# --- Timeout / Deadlock ---


class TestTimeout:
    def test_no_votes_deadlocks(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.DEADLOCK

    def test_timeout_deadlocks(self, scorer: PhiScorer) -> None:
        """Votes submitted but timeout exceeded → DEADLOCK."""
        config = CoordinationConfig(vote_timeout_seconds=0)
        tri = Triumvirate(scorer, config)
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        # Timeout is 0 seconds, so any elapsed time triggers it
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.DEADLOCK


# --- Decision retrieval ---


class TestDecisionRetrieval:
    def test_get_decision_before_evaluate(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        assert tri.get_decision(req.request_id) is None

    def test_get_decision_after_evaluate(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.evaluate(req.request_id)
        decision = tri.get_decision(req.request_id)
        assert decision is not None
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_get_vote_history(self, tri: Triumvirate) -> None:
        # Two requests for same task
        r1 = tri.create_request("task-1", "q1", "c")
        r2 = tri.create_request("task-1", "q2", "c")
        r3 = tri.create_request("task-2", "q3", "c")  # different task
        for r in (r1, r2, r3):
            tri.submit_vote(r.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
            tri.evaluate(r.request_id)
        history = tri.get_vote_history("task-1")
        assert len(history) == 2
        assert all(d.request.task_id == "task-1" for d in history)

    def test_evaluate_unknown_request(self, tri: Triumvirate) -> None:
        with pytest.raises(KeyError, match="Unknown request"):
            tri.evaluate("nonexistent")


# --- Reasoning summary ---


class TestReasoningSummary:
    def test_summary_includes_all_votes(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(
            req.request_id,
            _vote(_agent("alice"), VoteChoice.APPROVE, reasoning="Clean code"),
        )
        tri.submit_vote(
            req.request_id,
            _vote(_agent("bob"), VoteChoice.REJECT, reasoning="Missing tests"),
        )
        decision = tri.evaluate(req.request_id)
        assert "alice" in decision.reasoning_summary
        assert "bob" in decision.reasoning_summary
        assert "Clean code" in decision.reasoning_summary
        assert "Missing tests" in decision.reasoning_summary


# --- Decision fields ---


class TestDecisionFields:
    def test_weighted_totals(self, tri: Triumvirate, scorer: PhiScorer) -> None:
        _seed_score(scorer, "a1", 0.8)
        _seed_score(scorer, "a2", 0.6)
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a1", 0.8), VoteChoice.APPROVE, confidence=0.9),
        )
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a2", 0.6), VoteChoice.REJECT, confidence=0.7),
        )
        decision = tri.evaluate(req.request_id)
        assert decision.total_weighted_approve == pytest.approx(0.72)  # 0.8*0.9
        assert decision.total_weighted_reject == pytest.approx(0.42)  # 0.6*0.7

    def test_request_preserved(self, tri: Triumvirate) -> None:
        req = tri.create_request("task-x", "Question?", "ctx")
        tri.submit_vote(req.request_id, _vote(_agent("a"), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.request.task_id == "task-x"
        assert decision.request.question == "Question?"

    def test_votes_preserved(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.REJECT))
        decision = tri.evaluate(req.request_id)
        assert len(decision.votes) == 2


# --- Agent count flexibility ---


class TestAgentCount:
    def test_two_agents(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_five_agents(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        for i in range(3):
            tri.submit_vote(req.request_id, _vote(_agent(f"a{i}"), VoteChoice.APPROVE))
        for i in range(3, 5):
            tri.submit_vote(req.request_id, _vote(_agent(f"a{i}"), VoteChoice.REJECT))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED  # 3 > 2

    def test_single_agent(self, tri: Triumvirate) -> None:
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED


# --- Tie-breaking edge cases ---


class TestNaiveTimestamp:
    def test_naive_requested_at_treated_as_utc(self, scorer: PhiScorer) -> None:
        """ConsensusRequest with naive timestamp still evaluates correctly."""
        from datetime import datetime

        from convergent.protocol import ConsensusRequest

        tri = Triumvirate(scorer, CoordinationConfig())
        # Manually create a request with naive timestamp
        naive_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")  # noqa: DTZ005
        req = ConsensusRequest(
            request_id="r-naive",
            task_id="t",
            question="q",
            context="c",
            quorum=QuorumLevel.MAJORITY,
            requested_at=naive_ts,
        )
        tri._requests["r-naive"] = req
        tri._votes["r-naive"] = []
        # No votes + naive timestamp → deadlock (but won't crash)
        decision = tri.evaluate("r-naive")
        assert decision.outcome is DecisionOutcome.DEADLOCK


class TestTieBreaking:
    def test_majority_tie_all_abstain_deadlocks(self, tri: Triumvirate) -> None:
        """MAJORITY with only abstain votes → tie (0==0) → _break_tie → DEADLOCK."""
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.MAJORITY)
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.ABSTAIN))
        tri.submit_vote(req.request_id, _vote(_agent("a2"), VoteChoice.ABSTAIN))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.DEADLOCK

    def test_majority_tie_broken_by_reject(self, tri: Triumvirate, scorer: PhiScorer) -> None:
        """When tied totals, highest-weighted vote is a reject → REJECTED."""
        _seed_score(scorer, "a1", 0.3)
        _seed_score(scorer, "a2", 0.3)
        _seed_score(scorer, "a3", 0.6)
        req = tri.create_request("t", "q", "c", quorum=QuorumLevel.MAJORITY)
        # Approve: 0.3*1.0 + 0.3*1.0 = 0.6
        # Reject: 0.6*1.0 = 0.6
        # Tie → highest individual = reject at 0.6
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a1", 0.3), VoteChoice.APPROVE, confidence=1.0),
        )
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a2", 0.3), VoteChoice.APPROVE, confidence=1.0),
        )
        tri.submit_vote(
            req.request_id,
            _vote(_agent("a3", 0.6), VoteChoice.REJECT, confidence=1.0),
        )
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.REJECTED


# --- Decision persistence ---


class TestDecisionPersistence:
    def test_evaluate_persists_to_store(self, scorer: PhiScorer) -> None:
        store = ScoreStore(":memory:")
        tri = Triumvirate(scorer, CoordinationConfig(), store=store)
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        tri.evaluate(req.request_id)
        history = store.get_decision_history()
        assert len(history) == 1
        assert history[0]["outcome"] == "approved"

    def test_deadlock_persists_to_store(self, scorer: PhiScorer) -> None:
        store = ScoreStore(":memory:")
        tri = Triumvirate(scorer, CoordinationConfig(), store=store)
        req = tri.create_request("t", "q", "c")
        # No votes → deadlock
        tri.evaluate(req.request_id)
        history = store.get_decision_history()
        assert len(history) == 1
        assert history[0]["outcome"] == "deadlock"

    def test_escalation_persists_to_store(self, scorer: PhiScorer) -> None:
        store = ScoreStore(":memory:")
        tri = Triumvirate(scorer, CoordinationConfig(), store=store)
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.ESCALATE))
        tri.evaluate(req.request_id)
        history = store.get_decision_history()
        assert len(history) == 1
        assert history[0]["outcome"] == "escalated"

    def test_no_store_no_error(self, scorer: PhiScorer) -> None:
        """Triumvirate without store still works — graceful degradation."""
        tri = Triumvirate(scorer, CoordinationConfig())  # No store
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED

    def test_persist_failure_graceful(self, scorer: PhiScorer) -> None:
        """If store.record_decision raises, evaluate still returns the decision."""
        from unittest.mock import MagicMock

        bad_store = MagicMock()
        bad_store.record_decision.side_effect = RuntimeError("db error")
        tri = Triumvirate(scorer, CoordinationConfig(), store=bad_store)
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1"), VoteChoice.APPROVE))
        decision = tri.evaluate(req.request_id)
        assert decision.outcome is DecisionOutcome.APPROVED
        bad_store.record_decision.assert_called_once()

    def test_votes_persisted_with_decision(self, scorer: PhiScorer) -> None:
        store = ScoreStore(":memory:")
        tri = Triumvirate(scorer, CoordinationConfig(), store=store)
        req = tri.create_request("t", "q", "c")
        tri.submit_vote(req.request_id, _vote(_agent("a1", 0.8), VoteChoice.APPROVE))
        tri.submit_vote(req.request_id, _vote(_agent("a2", 0.6), VoteChoice.REJECT))
        tri.evaluate(req.request_id)
        records = store.get_vote_records(request_id=req.request_id)
        assert len(records) == 2
        choices = {r["choice"] for r in records}
        assert choices == {"approve", "reject"}


# --- Public API ---


class TestPublicAPI:
    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "Triumvirate")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "Triumvirate" in convergent.__all__
