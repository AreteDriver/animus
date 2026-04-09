"""Tests for MEV JSONL store."""

from __future__ import annotations

import json

import pytest

from animus_mev.models import (
    Chain,
    ExtractionResult,
    MempoolTx,
    MEVOpportunity,
    MEVType,
)
from animus_mev.store import MEVStore


class TestMEVStore:
    @pytest.fixture
    def store(self, tmp_data_dir):
        return MEVStore(tmp_data_dir)

    def test_create(self, store):
        assert store.total_opportunities() == 0

    def test_record_opportunity(self, store, sample_opportunity):
        assert store.record_opportunity(sample_opportunity) is True
        assert store.total_opportunities() == 1

    def test_record_duplicate(self, store, sample_opportunity):
        store.record_opportunity(sample_opportunity)
        assert store.record_opportunity(sample_opportunity) is False
        assert store.total_opportunities() == 1

    def test_is_new(self, store, sample_opportunity):
        assert store.is_new(sample_opportunity) is True
        store.record_opportunity(sample_opportunity)
        assert store.is_new(sample_opportunity) is False

    def test_load_opportunities(self, store, sample_opportunity):
        store.record_opportunity(sample_opportunity)
        loaded = store.load_opportunities()
        assert len(loaded) == 1
        assert loaded[0].mev_type == MEVType.BACKRUN

    def test_load_empty(self, store):
        assert store.load_opportunities() == []

    def test_record_result(self, store, sample_result):
        store.record_result(sample_result)
        results = store.load_results()
        assert len(results) == 1
        assert results[0].opportunity_id == "abc123"

    def test_load_results_empty(self, store):
        assert store.load_results() == []

    def test_recent_opportunities(self, store, sample_tx):
        for i in range(5):
            opp = MEVOpportunity(
                mev_type=MEVType.BACKRUN,
                chain=Chain.BASE,
                trigger_tx=MempoolTx(
                    chain=Chain.BASE,
                    tx_hash=f"0x{i:040x}",
                    from_addr="0x1",
                    to_addr="0x2626664c2603336e57b271c5c0b26f421741e481",
                    value=int(5e18),
                    gas_price=int(0.01e9),
                    input_data="0x04e45aaf" + "0" * 128,
                ),
                estimated_profit_usd=float(i),
                gas_cost_usd=0.1,
                confidence=0.8,
            )
            store.record_opportunity(opp)

        recent = store.recent_opportunities(limit=3)
        assert len(recent) == 3

    def test_recent_results(self, store):
        for i in range(5):
            store.record_result(
                ExtractionResult(
                    opportunity_id=f"opp_{i}",
                    executed=True,
                    actual_profit_usd=float(i),
                )
            )
        recent = store.recent_results(limit=3)
        assert len(recent) == 3

    def test_persistence_across_instances(self, tmp_data_dir, sample_opportunity):
        store1 = MEVStore(tmp_data_dir)
        store1.record_opportunity(sample_opportunity)
        assert store1.total_opportunities() == 1

        store2 = MEVStore(tmp_data_dir)
        assert store2.total_opportunities() == 1
        assert store2.is_new(sample_opportunity) is False

    def test_malformed_line_skipped(self, tmp_data_dir):
        MEVStore(tmp_data_dir)  # Initialize to create file structure
        # Write a malformed line
        opp_file = tmp_data_dir / "opportunities.jsonl"
        with open(opp_file, "w") as f:
            f.write("not json\n")
            f.write('{"opportunity_id": "abc"}\n')  # Missing required fields

        # Reload should not crash
        store2 = MEVStore(tmp_data_dir)
        assert store2.total_opportunities() == 1  # Only the parseable line's ID

    def test_malformed_results_skipped(self, tmp_data_dir):
        results_file = tmp_data_dir / "results.jsonl"
        with open(results_file, "w") as f:
            f.write("garbage\n")
            f.write(json.dumps({"opportunity_id": "x", "executed": True}) + "\n")

        store = MEVStore(tmp_data_dir)
        results = store.load_results()
        assert len(results) == 1

    def test_malformed_opportunities_load(self, tmp_data_dir):
        opp_file = tmp_data_dir / "opportunities.jsonl"
        with open(opp_file, "w") as f:
            f.write("not valid\n")

        store = MEVStore(tmp_data_dir)
        opps = store.load_opportunities()
        assert opps == []
