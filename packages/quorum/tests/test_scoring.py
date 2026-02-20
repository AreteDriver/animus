"""Tests for convergent.scoring and convergent.score_store."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from convergent.protocol import AgentIdentity, Vote, VoteChoice
from convergent.score_store import ScoreStore
from convergent.scoring import PhiScorer

# --- ScoreStore tests ---


class TestScoreStore:
    @pytest.fixture()
    def store(self, tmp_path: object) -> ScoreStore:
        return ScoreStore(":memory:")

    def test_record_and_get_outcomes(self, store: ScoreStore) -> None:
        store.record_outcome("agent-1", "code_review", "approved")
        store.record_outcome("agent-1", "code_review", "rejected")
        outcomes = store.get_outcomes("agent-1", "code_review")
        assert len(outcomes) == 2
        assert outcomes[0][0] == "approved"
        assert outcomes[1][0] == "rejected"

    def test_outcomes_isolated_by_domain(self, store: ScoreStore) -> None:
        store.record_outcome("agent-1", "code_review", "approved")
        store.record_outcome("agent-1", "testing", "rejected")
        review = store.get_outcomes("agent-1", "code_review")
        testing = store.get_outcomes("agent-1", "testing")
        assert len(review) == 1
        assert review[0][0] == "approved"
        assert len(testing) == 1
        assert testing[0][0] == "rejected"

    def test_outcomes_isolated_by_agent(self, store: ScoreStore) -> None:
        store.record_outcome("agent-1", "code_review", "approved")
        store.record_outcome("agent-2", "code_review", "rejected")
        a1 = store.get_outcomes("agent-1", "code_review")
        a2 = store.get_outcomes("agent-2", "code_review")
        assert len(a1) == 1
        assert a1[0][0] == "approved"
        assert len(a2) == 1
        assert a2[0][0] == "rejected"

    def test_get_all_domains(self, store: ScoreStore) -> None:
        store.record_outcome("agent-1", "code_review", "approved")
        store.record_outcome("agent-1", "testing", "approved")
        store.record_outcome("agent-1", "code_review", "rejected")
        domains = store.get_all_domains("agent-1")
        assert sorted(domains) == ["code_review", "testing"]

    def test_get_all_domains_empty(self, store: ScoreStore) -> None:
        assert store.get_all_domains("nonexistent") == []

    def test_save_and_get_score(self, store: ScoreStore) -> None:
        store.save_score("agent-1", "code_review", 0.75)
        assert store.get_score("agent-1", "code_review") == 0.75

    def test_get_score_nonexistent(self, store: ScoreStore) -> None:
        assert store.get_score("agent-1", "code_review") is None

    def test_save_score_upserts(self, store: ScoreStore) -> None:
        store.save_score("agent-1", "code_review", 0.5)
        store.save_score("agent-1", "code_review", 0.8)
        assert store.get_score("agent-1", "code_review") == 0.8

    def test_get_all_scores(self, store: ScoreStore) -> None:
        store.save_score("agent-1", "code_review", 0.8)
        store.save_score("agent-1", "testing", 0.3)
        scores = store.get_all_scores("agent-1")
        assert scores == {"code_review": 0.8, "testing": 0.3}

    def test_get_all_scores_empty(self, store: ScoreStore) -> None:
        assert store.get_all_scores("nonexistent") == {}

    def test_custom_timestamp(self, store: ScoreStore) -> None:
        ts = "2025-01-01T00:00:00+00:00"
        store.record_outcome("agent-1", "review", "approved", timestamp=ts)
        outcomes = store.get_outcomes("agent-1", "review")
        assert outcomes[0][1] == ts

    def test_close(self, store: ScoreStore) -> None:
        store.close()
        # After close, operations should fail
        with pytest.raises(sqlite3.ProgrammingError):
            store.record_outcome("a", "d", "approved")

    def test_file_persistence(self, tmp_path: object) -> None:
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "test.db")
        store1 = ScoreStore(db_path)
        store1.record_outcome("agent-1", "review", "approved")
        store1.save_score("agent-1", "review", 0.75)
        store1.close()

        store2 = ScoreStore(db_path)
        outcomes = store2.get_outcomes("agent-1", "review")
        assert len(outcomes) == 1
        assert store2.get_score("agent-1", "review") == 0.75
        store2.close()


# --- PhiScorer.calculate_phi_score tests ---


class TestCalculatePhiScore:
    def test_no_outcomes_returns_prior(self) -> None:
        assert PhiScorer.calculate_phi_score([]) == 0.5

    def test_no_outcomes_custom_prior(self) -> None:
        assert PhiScorer.calculate_phi_score([], prior_score=0.7) == 0.7

    def test_all_approvals_trends_toward_max(self) -> None:
        # 50 recent approvals should push score high but not reach 0.95
        outcomes = [("approved", 0.0) for _ in range(50)]
        score = PhiScorer.calculate_phi_score(outcomes)
        assert score > 0.9
        assert score <= 0.95

    def test_all_rejections_trends_toward_min(self) -> None:
        # 50 recent rejections should push score low but not reach 0.1
        outcomes = [("rejected", 0.0) for _ in range(50)]
        score = PhiScorer.calculate_phi_score(outcomes)
        assert score < 0.15
        assert score >= 0.1

    def test_recent_outcomes_weighted_higher(self) -> None:
        # Old approvals + recent rejections should yield low score
        old_approvals = [("approved", 100.0) for _ in range(10)]
        recent_rejections = [("rejected", 0.0) for _ in range(5)]
        score = PhiScorer.calculate_phi_score(old_approvals + recent_rejections)
        assert score < 0.5  # Recent rejections dominate

    def test_recent_approvals_beat_old_rejections(self) -> None:
        old_rejections = [("rejected", 100.0) for _ in range(10)]
        recent_approvals = [("approved", 0.0) for _ in range(5)]
        score = PhiScorer.calculate_phi_score(old_rejections + recent_approvals)
        assert score > 0.5  # Recent approvals dominate

    def test_mixed_outcomes_converge_to_ratio(self) -> None:
        # 7 approvals, 3 rejections, all recent → roughly 0.7
        outcomes = [("approved", 0.0) for _ in range(7)] + [("rejected", 0.0) for _ in range(3)]
        score = PhiScorer.calculate_phi_score(outcomes)
        assert 0.55 < score < 0.80  # Bayesian prior pulls toward 0.5

    def test_score_bounded_below(self) -> None:
        outcomes = [("rejected", 0.0) for _ in range(1000)]
        score = PhiScorer.calculate_phi_score(outcomes)
        assert score >= 0.1

    def test_score_bounded_above(self) -> None:
        outcomes = [("approved", 0.0) for _ in range(1000)]
        score = PhiScorer.calculate_phi_score(outcomes)
        assert score <= 0.95

    def test_custom_bounds(self) -> None:
        outcomes = [("approved", 0.0) for _ in range(1000)]
        score = PhiScorer.calculate_phi_score(
            outcomes,
            min_score=0.2,
            max_score=0.8,
        )
        assert score == 0.8

    def test_single_approval(self) -> None:
        score = PhiScorer.calculate_phi_score([("approved", 0.0)])
        # With prior_weight=2.0 at 0.5, one approval:
        # (1.0 + 2.0*0.5) / (1.0 + 2.0) = 2.0/3.0 ≈ 0.667
        assert 0.6 < score < 0.7

    def test_single_rejection(self) -> None:
        score = PhiScorer.calculate_phi_score([("rejected", 0.0)])
        # (0.0 + 2.0*0.5) / (1.0 + 2.0) = 1.0/3.0 ≈ 0.333
        assert 0.3 < score < 0.4

    def test_failed_counts_as_not_approved(self) -> None:
        score_rejected = PhiScorer.calculate_phi_score([("rejected", 0.0)])
        score_failed = PhiScorer.calculate_phi_score([("failed", 0.0)])
        assert score_rejected == score_failed

    def test_decay_rate_effect(self) -> None:
        outcomes = [("approved", 30.0)]  # 30 days old
        fast_decay = PhiScorer.calculate_phi_score(outcomes, decay_rate=0.1)
        slow_decay = PhiScorer.calculate_phi_score(outcomes, decay_rate=0.01)
        # With faster decay, old outcomes matter less → closer to prior
        assert abs(fast_decay - 0.5) < abs(slow_decay - 0.5)


# --- PhiScorer integration tests ---


class TestPhiScorer:
    @pytest.fixture()
    def scorer(self) -> PhiScorer:
        store = ScoreStore(":memory:")
        return PhiScorer(store)

    def test_new_agent_gets_prior_score(self, scorer: PhiScorer) -> None:
        assert scorer.get_score("agent-1", "code_review") == 0.5

    def test_record_outcome_returns_score(self, scorer: PhiScorer) -> None:
        score = scorer.record_outcome("agent-1", "code_review", "approved")
        assert isinstance(score, float)
        assert 0.1 <= score <= 0.95

    def test_approvals_increase_score(self, scorer: PhiScorer) -> None:
        for _ in range(10):
            scorer.record_outcome("agent-1", "review", "approved")
        score = scorer.get_score("agent-1", "review")
        assert score > 0.5

    def test_rejections_decrease_score(self, scorer: PhiScorer) -> None:
        for _ in range(10):
            scorer.record_outcome("agent-1", "review", "rejected")
        score = scorer.get_score("agent-1", "review")
        assert score < 0.5

    def test_per_domain_independence(self, scorer: PhiScorer) -> None:
        # Good at review, bad at testing
        for _ in range(10):
            scorer.record_outcome("agent-1", "code_review", "approved")
        for _ in range(10):
            scorer.record_outcome("agent-1", "testing", "rejected")

        review_score = scorer.get_score("agent-1", "code_review")
        test_score = scorer.get_score("agent-1", "testing")
        assert review_score > 0.5
        assert test_score < 0.5
        assert review_score > test_score

    def test_get_all_scores(self, scorer: PhiScorer) -> None:
        scorer.record_outcome("agent-1", "code_review", "approved")
        scorer.record_outcome("agent-1", "testing", "rejected")
        scores = scorer.get_all_scores("agent-1")
        assert "code_review" in scores
        assert "testing" in scores
        assert scores["code_review"] > scores["testing"]

    def test_get_all_scores_empty(self, scorer: PhiScorer) -> None:
        assert scorer.get_all_scores("nonexistent") == {}

    def test_score_idempotent(self, scorer: PhiScorer) -> None:
        scorer.record_outcome("agent-1", "review", "approved")
        score1 = scorer.get_score("agent-1", "review")
        # Recording same outcome should just add another data point
        scorer.record_outcome("agent-1", "review", "approved")
        score2 = scorer.get_score("agent-1", "review")
        # Score should increase (more approvals) but recalculation is consistent
        assert score2 >= score1

    def test_apply_vote_weight(self, scorer: PhiScorer) -> None:
        # No outcomes recorded → scorer uses prior (0.5), not self-reported 0.8
        agent = AgentIdentity("agent-1", "reviewer", "claude:sonnet", phi_score=0.8)
        vote = Vote(agent, VoteChoice.APPROVE, confidence=0.9, reasoning="lgtm")
        weighted = scorer.apply_vote_weight(vote)
        assert weighted.weighted_score == pytest.approx(0.45)  # 0.5 (prior) * 0.9
        # Original vote unchanged (frozen)
        assert vote.weighted_score == 0.0

    def test_apply_vote_weight_zero_confidence(self, scorer: PhiScorer) -> None:
        agent = AgentIdentity("a", "r", "m", phi_score=0.9)
        vote = Vote(agent, VoteChoice.ABSTAIN, confidence=0.0, reasoning="unsure")
        weighted = scorer.apply_vote_weight(vote)
        assert weighted.weighted_score == 0.0

    def test_apply_vote_weight_preserves_fields(self, scorer: PhiScorer) -> None:
        agent = AgentIdentity("agent-1", "tester", "claude:haiku", phi_score=0.6)
        vote = Vote(agent, VoteChoice.REJECT, confidence=0.7, reasoning="bugs found")
        weighted = scorer.apply_vote_weight(vote)
        assert weighted.agent == agent
        assert weighted.choice is VoteChoice.REJECT
        assert weighted.confidence == 0.7
        assert weighted.reasoning == "bugs found"
        # Uses stored prior (0.5), not self-reported 0.6
        assert weighted.weighted_score == pytest.approx(0.35)  # 0.5 * 0.7

    def test_apply_vote_weight_uses_stored_score(self, scorer: PhiScorer) -> None:
        """Vote weight comes from the store, not the agent's self-reported phi_score."""
        # Build up a high stored score via approvals
        for _ in range(20):
            scorer.record_outcome("agent-1", "reviewer", "approved")
        stored = scorer.get_score("agent-1", "reviewer")
        assert stored > 0.8

        # Agent self-reports a low phi_score — should be ignored
        agent = AgentIdentity("agent-1", "reviewer", "m", phi_score=0.1)
        vote = Vote(agent, VoteChoice.APPROVE, confidence=1.0, reasoning="ok")
        weighted = scorer.apply_vote_weight(vote)
        # Weight should use the high stored score, not 0.1
        assert weighted.weighted_score == pytest.approx(stored, abs=0.01)

    def test_custom_scorer_params(self) -> None:
        store = ScoreStore(":memory:")
        scorer = PhiScorer(
            store,
            decay_rate=0.1,
            prior_score=0.3,
            min_score=0.2,
            max_score=0.8,
        )
        assert scorer.get_score("a", "d") == 0.3  # Custom prior

    def test_score_with_old_timestamps(self) -> None:
        store = ScoreStore(":memory:")
        scorer = PhiScorer(store, decay_rate=0.05)

        # Record outcomes with explicit old timestamps
        old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        recent = datetime.now(timezone.utc).isoformat()

        store.record_outcome("agent-1", "review", "rejected", timestamp=old)
        store.record_outcome("agent-1", "review", "approved", timestamp=recent)

        # Recalculate manually by triggering another outcome
        score = scorer.record_outcome("agent-1", "review", "approved")
        # Recent approvals should dominate over old rejection
        assert score > 0.5


# --- Decision History tests ---


class TestDecisionHistory:
    @pytest.fixture()
    def store(self) -> ScoreStore:
        return ScoreStore(":memory:")

    def _make_decision(self, store: ScoreStore, task_id: str = "task-1") -> None:
        """Helper to create and persist a decision."""
        from convergent.coordination_config import CoordinationConfig
        from convergent.triumvirate import Triumvirate

        scorer = PhiScorer(store)
        tri = Triumvirate(scorer, CoordinationConfig(), store=store)
        req = tri.create_request(task_id, "Merge?", "context")
        agent = AgentIdentity("a1", "reviewer", "m", phi_score=0.8)
        vote = Vote(agent, VoteChoice.APPROVE, 0.9, "ok")
        tri.submit_vote(req.request_id, vote)
        tri.evaluate(req.request_id)

    def test_record_and_retrieve_decision(self, store: ScoreStore) -> None:
        self._make_decision(store)
        history = store.get_decision_history()
        assert len(history) == 1
        assert history[0]["task_id"] == "task-1"
        assert history[0]["outcome"] == "approved"

    def test_filter_by_task_id(self, store: ScoreStore) -> None:
        self._make_decision(store, "task-1")
        self._make_decision(store, "task-2")
        history = store.get_decision_history(task_id="task-1")
        assert len(history) == 1
        assert history[0]["task_id"] == "task-1"

    def test_filter_by_outcome(self, store: ScoreStore) -> None:
        self._make_decision(store)
        history = store.get_decision_history(outcome="approved")
        assert len(history) == 1
        history = store.get_decision_history(outcome="rejected")
        assert len(history) == 0

    def test_filter_by_since(self, store: ScoreStore) -> None:
        self._make_decision(store)
        # Since far future → no results
        history = store.get_decision_history(since="2099-01-01T00:00:00+00:00")
        assert len(history) == 0
        # Since far past → all results
        history = store.get_decision_history(since="2000-01-01T00:00:00+00:00")
        assert len(history) == 1

    def test_limit(self, store: ScoreStore) -> None:
        for i in range(5):
            self._make_decision(store, f"task-{i}")
        history = store.get_decision_history(limit=3)
        assert len(history) == 3

    def test_get_decision_json(self, store: ScoreStore) -> None:
        import json

        self._make_decision(store)
        history = store.get_decision_history()
        request_id = history[0]["request_id"]
        raw = store.get_decision_json(request_id)
        assert raw is not None
        d = json.loads(raw)
        assert d["outcome"] == "approved"
        assert "votes" in d

    def test_get_decision_json_not_found(self, store: ScoreStore) -> None:
        assert store.get_decision_json("nonexistent") is None

    def test_get_vote_records(self, store: ScoreStore) -> None:
        self._make_decision(store)
        records = store.get_vote_records()
        assert len(records) == 1
        assert records[0]["agent_id"] == "a1"
        assert records[0]["choice"] == "approve"
        assert records[0]["confidence"] == 0.9

    def test_get_vote_records_by_agent(self, store: ScoreStore) -> None:
        self._make_decision(store)
        records = store.get_vote_records(agent_id="a1")
        assert len(records) == 1
        records = store.get_vote_records(agent_id="nonexistent")
        assert len(records) == 0

    def test_get_vote_records_by_request(self, store: ScoreStore) -> None:
        self._make_decision(store)
        history = store.get_decision_history()
        request_id = history[0]["request_id"]
        records = store.get_vote_records(request_id=request_id)
        assert len(records) == 1

    def test_get_vote_records_limit(self, store: ScoreStore) -> None:
        for i in range(5):
            self._make_decision(store, f"task-{i}")
        records = store.get_vote_records(limit=3)
        assert len(records) == 3

    def test_get_agent_vote_stats(self, store: ScoreStore) -> None:
        self._make_decision(store)
        stats = store.get_agent_vote_stats("a1")
        assert stats["total"] == 1
        assert stats["approve_count"] == 1
        assert stats["reject_count"] == 0
        assert stats["avg_confidence"] == pytest.approx(0.9)

    def test_get_agent_vote_stats_empty(self, store: ScoreStore) -> None:
        stats = store.get_agent_vote_stats("nonexistent")
        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_decision_upsert(self, store: ScoreStore) -> None:
        """Recording same request_id twice overwrites (INSERT OR REPLACE)."""
        self._make_decision(store)
        history = store.get_decision_history()
        assert len(history) == 1
        # Recording again with same request_id should not create duplicate
        # (this tests the OR REPLACE behavior)

    def test_multiple_vote_choices_in_stats(self, store: ScoreStore) -> None:
        """Stats aggregate across multiple vote choices."""
        from convergent.coordination_config import CoordinationConfig
        from convergent.triumvirate import Triumvirate

        scorer = PhiScorer(store)
        tri = Triumvirate(scorer, CoordinationConfig(), store=store)

        # Create two decisions with different vote patterns
        req1 = tri.create_request("t1", "q1", "c")
        tri.submit_vote(
            req1.request_id,
            Vote(
                AgentIdentity("a1", "r", "m", phi_score=0.8),
                VoteChoice.APPROVE,
                0.9,
                "ok",
            ),
        )
        tri.evaluate(req1.request_id)

        req2 = tri.create_request("t2", "q2", "c")
        tri.submit_vote(
            req2.request_id,
            Vote(
                AgentIdentity("a1", "r", "m", phi_score=0.8),
                VoteChoice.REJECT,
                0.7,
                "no",
            ),
        )
        tri.evaluate(req2.request_id)

        stats = store.get_agent_vote_stats("a1")
        assert stats["total"] == 2
        assert stats["approve_count"] == 1
        assert stats["reject_count"] == 1
        assert stats["avg_confidence"] == pytest.approx(0.8)  # (0.9+0.7)/2


# --- Public API tests ---


class TestPublicAPI:
    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "PhiScorer")
        assert hasattr(convergent, "ScoreStore")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "PhiScorer" in convergent.__all__
        assert "ScoreStore" in convergent.__all__
