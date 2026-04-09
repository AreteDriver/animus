"""Tests for animus_bounty.store."""

from __future__ import annotations

from animus_bounty.models import (
    BountyStatus,
    ClaimResult,
)
from animus_bounty.store import BountyStore, ClaimStore, MatchStore


class TestBountyStore:
    def test_empty_store(self, tmp_data_dir):
        store = BountyStore(tmp_data_dir)
        assert store.total_count() == 0
        assert store.load_all() == []

    def test_record_new(self, tmp_data_dir, sample_bounty):
        store = BountyStore(tmp_data_dir)
        assert store.record(sample_bounty)
        assert store.total_count() == 1

    def test_record_duplicate(self, tmp_data_dir, sample_bounty):
        store = BountyStore(tmp_data_dir)
        assert store.record(sample_bounty)
        assert not store.record(sample_bounty)
        assert store.total_count() == 1

    def test_is_new(self, tmp_data_dir, sample_bounty):
        store = BountyStore(tmp_data_dir)
        assert store.is_new(sample_bounty)
        store.record(sample_bounty)
        assert not store.is_new(sample_bounty)

    def test_record_batch(self, tmp_data_dir, sample_bounty, hard_bounty):
        store = BountyStore(tmp_data_dir)
        count = store.record_batch([sample_bounty, hard_bounty, sample_bounty])
        assert count == 2
        assert store.total_count() == 2

    def test_load_all(self, tmp_data_dir, sample_bounty, hard_bounty):
        store = BountyStore(tmp_data_dir)
        store.record(sample_bounty)
        store.record(hard_bounty)
        loaded = store.load_all()
        assert len(loaded) == 2
        assert loaded[0].title == sample_bounty.title

    def test_by_status(self, tmp_data_dir, sample_bounty):
        store = BountyStore(tmp_data_dir)
        store.record(sample_bounty)
        assert len(store.by_status(BountyStatus.OPEN)) == 1
        assert len(store.by_status(BountyStatus.PAID)) == 0

    def test_by_platform(self, tmp_data_dir, sample_bounty, hard_bounty):
        store = BountyStore(tmp_data_dir)
        store.record(sample_bounty)
        store.record(hard_bounty)
        assert len(store.by_platform("github")) == 1
        assert len(store.by_platform("gitcoin")) == 1
        assert len(store.by_platform("generic")) == 0

    def test_get_by_id(self, tmp_data_dir, sample_bounty):
        store = BountyStore(tmp_data_dir)
        store.record(sample_bounty)
        found = store.get_by_id(sample_bounty.bounty_id)
        assert found is not None
        assert found.title == sample_bounty.title

    def test_get_by_id_not_found(self, tmp_data_dir):
        store = BountyStore(tmp_data_dir)
        assert store.get_by_id("nonexistent") is None

    def test_persistence(self, tmp_data_dir, sample_bounty):
        store1 = BountyStore(tmp_data_dir)
        store1.record(sample_bounty)

        # Create a new store instance — should load existing data
        store2 = BountyStore(tmp_data_dir)
        assert store2.total_count() == 1
        assert not store2.is_new(sample_bounty)

    def test_corrupted_line_skipped(self, tmp_data_dir):
        bounty_file = tmp_data_dir / "bounties.jsonl"
        bounty_file.write_text("not json\n")
        store = BountyStore(tmp_data_dir)
        assert store.total_count() == 0
        assert store.load_all() == []

    def test_creates_data_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "deep"
        store = BountyStore(new_dir)
        assert new_dir.exists()
        assert store.total_count() == 0

    def test_load_all_with_corrupted_line(self, tmp_data_dir, sample_bounty):
        store = BountyStore(tmp_data_dir)
        store.record(sample_bounty)
        # Append corrupted line
        with open(store._file, "a") as f:
            f.write("bad json\n")
        loaded = store.load_all()
        assert len(loaded) == 1

    def test_load_all_empty_lines(self, tmp_data_dir):
        bounty_file = tmp_data_dir / "bounties.jsonl"
        bounty_file.write_text("\n\n\n")
        store = BountyStore(tmp_data_dir)
        assert store.load_all() == []


class TestMatchStore:
    def test_empty(self, tmp_data_dir):
        store = MatchStore(tmp_data_dir)
        assert store.load_all() == []

    def test_record_and_load(self, tmp_data_dir, sample_match):
        store = MatchStore(tmp_data_dir)
        store.record(sample_match)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].match_score == 0.85

    def test_get_for_bounty(self, tmp_data_dir, sample_match):
        store = MatchStore(tmp_data_dir)
        store.record(sample_match)
        found = store.get_for_bounty(sample_match.bounty_id)
        assert found is not None
        assert found.match_score == 0.85

    def test_get_for_bounty_not_found(self, tmp_data_dir):
        store = MatchStore(tmp_data_dir)
        assert store.get_for_bounty("nonexistent") is None

    def test_corrupted_line(self, tmp_data_dir):
        store = MatchStore(tmp_data_dir)
        with open(store._file, "w") as f:
            f.write("corrupted\n")
        assert store.load_all() == []

    def test_creates_data_dir(self, tmp_path):
        new_dir = tmp_path / "match_store"
        MatchStore(new_dir)
        assert new_dir.exists()

    def test_empty_lines(self, tmp_data_dir):
        store = MatchStore(tmp_data_dir)
        with open(store._file, "w") as f:
            f.write("\n\n")
        assert store.load_all() == []


class TestClaimStore:
    def test_empty(self, tmp_data_dir):
        store = ClaimStore(tmp_data_dir)
        assert store.load_all() == []

    def test_record_and_load(self, tmp_data_dir, sample_claim):
        store = ClaimStore(tmp_data_dir)
        store.record(sample_claim)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].status == BountyStatus.CLAIMED

    def test_get_for_bounty(self, tmp_data_dir, sample_claim):
        store = ClaimStore(tmp_data_dir)
        store.record(sample_claim)
        found = store.get_for_bounty(sample_claim.bounty_id)
        assert found is not None

    def test_get_for_bounty_not_found(self, tmp_data_dir):
        store = ClaimStore(tmp_data_dir)
        assert store.get_for_bounty("nonexistent") is None

    def test_update_existing(self, tmp_data_dir, sample_claim):
        store = ClaimStore(tmp_data_dir)
        store.record(sample_claim)

        sample_claim.status = BountyStatus.PAID
        sample_claim.actual_reward = 25.0
        store.update(sample_claim)

        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].status == BountyStatus.PAID
        assert loaded[0].actual_reward == 25.0

    def test_update_nonexistent_appends(self, tmp_data_dir):
        store = ClaimStore(tmp_data_dir)
        claim = ClaimResult(bounty_id="new_id", status=BountyStatus.CLAIMED)
        store.update(claim)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].bounty_id == "new_id"

    def test_update_preserves_others(self, tmp_data_dir):
        store = ClaimStore(tmp_data_dir)
        c1 = ClaimResult(bounty_id="a", status=BountyStatus.CLAIMED)
        c2 = ClaimResult(bounty_id="b", status=BountyStatus.CLAIMED)
        store.record(c1)
        store.record(c2)

        c1.status = BountyStatus.PAID
        store.update(c1)

        loaded = store.load_all()
        assert len(loaded) == 2
        by_id = {c.bounty_id: c for c in loaded}
        assert by_id["a"].status == BountyStatus.PAID
        assert by_id["b"].status == BountyStatus.CLAIMED

    def test_corrupted_line(self, tmp_data_dir):
        store = ClaimStore(tmp_data_dir)
        with open(store._file, "w") as f:
            f.write("bad data\n")
        assert store.load_all() == []

    def test_creates_data_dir(self, tmp_path):
        new_dir = tmp_path / "claim_store"
        ClaimStore(new_dir)
        assert new_dir.exists()

    def test_empty_lines(self, tmp_data_dir):
        store = ClaimStore(tmp_data_dir)
        with open(store._file, "w") as f:
            f.write("\n\n")
        assert store.load_all() == []
