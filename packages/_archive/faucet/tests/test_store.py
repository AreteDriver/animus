"""Tests for claim and health stores."""

from animus_faucet.models import (
    Chain,
    ClaimResult,
    ClaimStatus,
    FaucetHealth,
)
from animus_faucet.store import ClaimStore, HealthStore


class TestClaimStore:
    def test_record_and_load(self, tmp_data_dir, sample_claim_success):
        store = ClaimStore(tmp_data_dir)
        store.record(sample_claim_success)
        claims = store.load_all()
        assert len(claims) == 1
        assert claims[0].faucet_name == "test_faucet"
        assert claims[0].succeeded is True

    def test_multiple_records(self, tmp_data_dir, sample_claim_success, sample_claim_failure):
        store = ClaimStore(tmp_data_dir)
        store.record(sample_claim_success)
        store.record(sample_claim_failure)
        claims = store.load_all()
        assert len(claims) == 2

    def test_total_claims(self, tmp_data_dir, sample_claim_success):
        store = ClaimStore(tmp_data_dir)
        assert store.total_claims() == 0
        store.record(sample_claim_success)
        store.record(sample_claim_success)
        assert store.total_claims() == 2

    def test_total_earned_usd(self, tmp_data_dir, sample_claim_success, sample_claim_failure):
        store = ClaimStore(tmp_data_dir)
        store.record(sample_claim_success)
        store.record(sample_claim_success)
        store.record(sample_claim_failure)
        assert store.total_earned_usd() == 0.10

    def test_claims_for_faucet(self, tmp_data_dir, sample_claim_success):
        store = ClaimStore(tmp_data_dir)
        store.record(sample_claim_success)
        other = ClaimResult(
            faucet_name="other_faucet",
            chain=Chain.BTC,
            status=ClaimStatus.SUCCESS,
            amount_usd=0.01,
        )
        store.record(other)
        filtered = store.claims_for_faucet("test_faucet")
        assert len(filtered) == 1
        assert filtered[0].faucet_name == "test_faucet"

    def test_success_rate(self, tmp_data_dir, sample_claim_success, sample_claim_failure):
        store = ClaimStore(tmp_data_dir)
        assert store.success_rate() == 0.0
        store.record(sample_claim_success)
        assert store.success_rate() == 1.0
        store.record(sample_claim_failure)
        assert store.success_rate() == 0.5

    def test_empty_file(self, tmp_data_dir):
        store = ClaimStore(tmp_data_dir)
        assert store.load_all() == []
        assert store.total_claims() == 0

    def test_creates_dir(self, tmp_path):
        data_dir = tmp_path / "new_dir"
        ClaimStore(data_dir)
        assert data_dir.exists()


class TestHealthStore:
    def test_save_and_get(self, tmp_data_dir):
        store = HealthStore(tmp_data_dir)
        health = FaucetHealth(name="test", enabled=True, success_count=5)
        store.save(health)
        retrieved = store.get("test")
        assert retrieved is not None
        assert retrieved.name == "test"
        assert retrieved.success_count == 5

    def test_get_nonexistent(self, tmp_data_dir):
        store = HealthStore(tmp_data_dir)
        assert store.get("nonexistent") is None

    def test_get_all(self, tmp_data_dir):
        store = HealthStore(tmp_data_dir)
        store.save(FaucetHealth(name="a", enabled=True))
        store.save(FaucetHealth(name="b", enabled=False))
        all_health = store.get_all()
        assert len(all_health) == 2

    def test_remove(self, tmp_data_dir):
        store = HealthStore(tmp_data_dir)
        store.save(FaucetHealth(name="test", enabled=True))
        assert store.remove("test") is True
        assert store.get("test") is None

    def test_remove_nonexistent(self, tmp_data_dir):
        store = HealthStore(tmp_data_dir)
        assert store.remove("nonexistent") is False

    def test_persistence(self, tmp_data_dir):
        store1 = HealthStore(tmp_data_dir)
        store1.save(FaucetHealth(name="test", enabled=True, success_count=10))

        store2 = HealthStore(tmp_data_dir)
        retrieved = store2.get("test")
        assert retrieved is not None
        assert retrieved.success_count == 10

    def test_update_existing(self, tmp_data_dir):
        store = HealthStore(tmp_data_dir)
        store.save(FaucetHealth(name="test", enabled=True, success_count=1))
        store.save(FaucetHealth(name="test", enabled=True, success_count=5))
        assert store.get("test").success_count == 5
        assert len(store.get_all()) == 1
