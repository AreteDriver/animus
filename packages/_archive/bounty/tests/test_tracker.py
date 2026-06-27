"""Tests for animus_bounty.tracker."""

from __future__ import annotations

from animus_bounty.models import (
    BountyStatus,
)
from animus_bounty.tracker import BountyTracker


class TestBountyTracker:
    def test_record_bounties(self, tmp_data_dir, sample_bounty, hard_bounty):
        tracker = BountyTracker(tmp_data_dir)
        count = tracker.record_bounties([sample_bounty, hard_bounty])
        assert count == 2

    def test_record_bounties_dedup(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        assert tracker.record_bounties([sample_bounty]) == 1
        assert tracker.record_bounties([sample_bounty]) == 0

    def test_record_match(self, tmp_data_dir, sample_match):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_match(sample_match)
        loaded = tracker.match_store.load_all()
        assert len(loaded) == 1

    def test_claim_bounty(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty])
        claim = tracker.claim_bounty(sample_bounty.bounty_id, notes="Test claim")
        assert claim.status == BountyStatus.CLAIMED
        assert claim.notes == "Test claim"

    def test_submit_bounty(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty])
        tracker.claim_bounty(sample_bounty.bounty_id)
        result = tracker.submit_bounty(sample_bounty.bounty_id, notes="PR submitted")
        assert result is not None
        assert result.status == BountyStatus.SUBMITTED
        assert result.submitted_at is not None
        assert result.notes == "PR submitted"

    def test_submit_unclaimed(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        result = tracker.submit_bounty("nonexistent")
        assert result is None

    def test_mark_paid(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty])
        tracker.claim_bounty(sample_bounty.bounty_id)
        result = tracker.mark_paid(sample_bounty.bounty_id, actual_reward=25.0)
        assert result is not None
        assert result.status == BountyStatus.PAID
        assert result.paid_at is not None
        assert result.actual_reward == 25.0

    def test_mark_paid_unclaimed(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        result = tracker.mark_paid("nonexistent")
        assert result is None

    def test_mark_paid_with_notes(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.claim_bounty(sample_bounty.bounty_id)
        result = tracker.mark_paid(sample_bounty.bounty_id, actual_reward=50.0, notes="Bonus")
        assert result is not None
        assert result.notes == "Bonus"

    def test_mark_rejected(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.claim_bounty(sample_bounty.bounty_id)
        result = tracker.mark_rejected(sample_bounty.bounty_id, notes="Not accepted")
        assert result is not None
        assert result.status == BountyStatus.REJECTED
        assert result.notes == "Not accepted"

    def test_mark_rejected_unclaimed(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        result = tracker.mark_rejected("nonexistent")
        assert result is None

    def test_submit_without_notes(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.claim_bounty(sample_bounty.bounty_id)
        result = tracker.submit_bounty(sample_bounty.bounty_id)
        assert result is not None
        assert result.notes == ""

    def test_mark_rejected_without_notes(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.claim_bounty(sample_bounty.bounty_id, notes="original note")
        result = tracker.mark_rejected(sample_bounty.bounty_id)
        assert result is not None
        assert result.notes == "original note"  # preserved from claim

    def test_get_summary_empty(self, tmp_data_dir):
        tracker = BountyTracker(tmp_data_dir)
        summary = tracker.get_summary()
        assert summary["total_discovered"] == 0
        assert summary["total_claims"] == 0
        assert summary["total_earned_usd"] == 0.0
        assert summary["avg_match_score"] == 0.0

    def test_get_summary_with_data(self, tmp_data_dir, sample_bounty, hard_bounty, sample_match):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty, hard_bounty])
        tracker.record_match(sample_match)
        tracker.claim_bounty(sample_bounty.bounty_id)
        tracker.mark_paid(sample_bounty.bounty_id, actual_reward=25.0)

        summary = tracker.get_summary()
        assert summary["total_discovered"] == 2
        assert summary["total_claims"] == 1
        assert summary["total_matches"] == 1
        assert summary["total_earned_usd"] == 25.0
        assert summary["avg_match_score"] == 0.85
        assert "github" in summary["platform_breakdown"]

    def test_get_summary_pending(self, tmp_data_dir, sample_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty])
        tracker.claim_bounty(sample_bounty.bounty_id)

        summary = tracker.get_summary()
        assert summary["total_pending_usd"] == 25.0

    def test_get_summary_status_breakdown(self, tmp_data_dir, sample_bounty, hard_bounty):
        tracker = BountyTracker(tmp_data_dir)
        tracker.record_bounties([sample_bounty, hard_bounty])
        tracker.claim_bounty(sample_bounty.bounty_id)
        tracker.claim_bounty(hard_bounty.bounty_id)
        tracker.mark_paid(sample_bounty.bounty_id, actual_reward=25.0)

        summary = tracker.get_summary()
        assert summary["status_breakdown"]["paid"] == 1
        assert summary["status_breakdown"]["claimed"] == 1
