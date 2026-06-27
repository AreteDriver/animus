"""Tests for opportunity and evaluation stores."""

from animus_prospector.models import (
    Action,
    OpportunityStatus,
)
from animus_prospector.store import EvaluationStore, OpportunityStore


class TestOpportunityStore:
    def test_record_and_load(self, tmp_data_dir, sample_opportunity):
        store = OpportunityStore(tmp_data_dir)
        assert store.record(sample_opportunity) is True
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].title == sample_opportunity.title

    def test_dedup(self, tmp_data_dir, sample_opportunity):
        store = OpportunityStore(tmp_data_dir)
        assert store.record(sample_opportunity) is True
        assert store.record(sample_opportunity) is False  # duplicate
        assert store.total_count() == 1

    def test_is_new(self, tmp_data_dir, sample_opportunity):
        store = OpportunityStore(tmp_data_dir)
        assert store.is_new(sample_opportunity) is True
        store.record(sample_opportunity)
        assert store.is_new(sample_opportunity) is False

    def test_record_batch(self, tmp_data_dir, sample_opportunity, scam_opportunity):
        store = OpportunityStore(tmp_data_dir)
        count = store.record_batch([sample_opportunity, scam_opportunity])
        assert count == 2
        count2 = store.record_batch([sample_opportunity])  # duplicate
        assert count2 == 0

    def test_total_count(self, tmp_data_dir, sample_opportunity, scam_opportunity):
        store = OpportunityStore(tmp_data_dir)
        store.record(sample_opportunity)
        store.record(scam_opportunity)
        assert store.total_count() == 2

    def test_by_status(self, tmp_data_dir, sample_opportunity):
        store = OpportunityStore(tmp_data_dir)
        store.record(sample_opportunity)
        discovered = store.by_status(OpportunityStatus.DISCOVERED)
        assert len(discovered) == 1
        completed = store.by_status(OpportunityStatus.COMPLETED)
        assert len(completed) == 0

    def test_by_type(self, tmp_data_dir, sample_opportunity):
        store = OpportunityStore(tmp_data_dir)
        store.record(sample_opportunity)
        airdrops = store.by_type("airdrop")
        assert len(airdrops) == 1
        faucets = store.by_type("faucet")
        assert len(faucets) == 0

    def test_persistence(self, tmp_data_dir, sample_opportunity):
        store1 = OpportunityStore(tmp_data_dir)
        store1.record(sample_opportunity)

        store2 = OpportunityStore(tmp_data_dir)
        assert store2.is_new(sample_opportunity) is False
        assert store2.total_count() == 1

    def test_empty_store(self, tmp_data_dir):
        store = OpportunityStore(tmp_data_dir)
        assert store.load_all() == []
        assert store.total_count() == 0

    def test_creates_dir(self, tmp_path):
        data_dir = tmp_path / "new_dir"
        OpportunityStore(data_dir)
        assert data_dir.exists()


class TestEvaluationStore:
    def test_record_and_load(self, tmp_data_dir, sample_evaluation):
        store = EvaluationStore(tmp_data_dir)
        store.record(sample_evaluation)
        loaded = store.load_all()
        assert len(loaded) == 1
        assert loaded[0].opportunity_id == sample_evaluation.opportunity_id

    def test_get_for_opportunity(self, tmp_data_dir, sample_evaluation):
        store = EvaluationStore(tmp_data_dir)
        store.record(sample_evaluation)
        result = store.get_for_opportunity(sample_evaluation.opportunity_id)
        assert result is not None
        assert result.recommended_action == Action.AUTO_EXECUTE

    def test_get_for_opportunity_not_found(self, tmp_data_dir):
        store = EvaluationStore(tmp_data_dir)
        assert store.get_for_opportunity("nonexistent") is None

    def test_empty_store(self, tmp_data_dir):
        store = EvaluationStore(tmp_data_dir)
        assert store.load_all() == []

    def test_creates_dir(self, tmp_path):
        data_dir = tmp_path / "new_eval_dir"
        EvaluationStore(data_dir)
        assert data_dir.exists()
