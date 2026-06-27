"""Tests for JSONL persistence store."""

from __future__ import annotations

from pathlib import Path

from animus_arbitrage.models import ExecutionMode, TradeResult
from animus_arbitrage.store import ArbitrageStore

from .conftest import make_opportunity, make_result


class TestArbitrageStore:
    def test_init_creates_dir(self, tmp_path: Path):
        ArbitrageStore(tmp_path / "new_dir")
        assert (tmp_path / "new_dir").exists()

    def test_record_opportunity(self, store: ArbitrageStore):
        opp = make_opportunity()
        assert store.record_opportunity(opp) is True
        assert store.opportunity_count() == 1

    def test_record_opportunity_dedup(self, store: ArbitrageStore):
        opp = make_opportunity()
        store.record_opportunity(opp)
        assert store.record_opportunity(opp) is False
        assert store.opportunity_count() == 1

    def test_record_opportunities_batch(self, store: ArbitrageStore):
        opp1 = make_opportunity()
        opp2 = make_opportunity()
        count = store.record_opportunities([opp1, opp2])
        assert count == 2
        assert store.opportunity_count() == 2

    def test_record_opportunities_batch_with_dupes(self, store: ArbitrageStore):
        opp1 = make_opportunity()
        store.record_opportunity(opp1)
        opp2 = make_opportunity()
        count = store.record_opportunities([opp1, opp2])
        assert count == 1  # opp1 is duplicate

    def test_load_opportunities(self, store: ArbitrageStore):
        opp = make_opportunity(spread_pct=1.5, profit_after_gas=10.0)
        store.record_opportunity(opp)
        loaded = store.load_opportunities()
        assert len(loaded) == 1
        assert loaded[0].opportunity_id == opp.opportunity_id
        assert loaded[0].spread_pct == 1.5

    def test_load_opportunities_empty(self, store: ArbitrageStore):
        loaded = store.load_opportunities()
        assert loaded == []

    def test_load_opportunities_corrupt_line(self, store: ArbitrageStore):
        """Corrupt lines should be skipped."""
        opp = make_opportunity()
        store.record_opportunity(opp)
        # Append corrupt line
        with open(store._opportunities_file, "a") as f:
            f.write("not json\n")
        loaded = store.load_opportunities()
        assert len(loaded) == 1

    def test_record_result(self, store: ArbitrageStore):
        result = make_result(profit=5.0)
        store.record_result(result)
        loaded = store.load_results()
        assert len(loaded) == 1
        assert loaded[0].actual_profit_usd == 5.0

    def test_load_results_empty(self, store: ArbitrageStore):
        loaded = store.load_results()
        assert loaded == []

    def test_load_results_corrupt_line(self, store: ArbitrageStore):
        result = make_result()
        store.record_result(result)
        with open(store._results_file, "a") as f:
            f.write("bad data\n")
        loaded = store.load_results()
        assert len(loaded) == 1

    def test_result_count(self, store: ArbitrageStore):
        store.record_result(make_result(opportunity_id="a"))
        store.record_result(make_result(opportunity_id="b"))
        assert store.result_count() == 2

    def test_persistence_across_instances(self, tmp_data_dir: Path):
        store1 = ArbitrageStore(tmp_data_dir)
        opp = make_opportunity()
        store1.record_opportunity(opp)

        store2 = ArbitrageStore(tmp_data_dir)
        assert store2.opportunity_count() == 1
        assert opp.opportunity_id in store2._seen_ids

    def test_empty_lines_skipped(self, store: ArbitrageStore):
        opp = make_opportunity()
        store.record_opportunity(opp)
        with open(store._opportunities_file, "a") as f:
            f.write("\n\n")
        loaded = store.load_opportunities()
        assert len(loaded) == 1

    def test_result_roundtrip(self, store: ArbitrageStore):
        result = TradeResult(
            opportunity_id="test123",
            executed=True,
            mode=ExecutionMode.LIVE,
            buy_tx="0xaaa",
            sell_tx="0xbbb",
            actual_profit_usd=5.0,
            slippage_pct=0.1,
        )
        store.record_result(result)
        loaded = store.load_results()
        assert len(loaded) == 1
        r = loaded[0]
        assert r.opportunity_id == "test123"
        assert r.executed is True
        assert r.mode == ExecutionMode.LIVE
        assert r.buy_tx == "0xaaa"
        assert r.sell_tx == "0xbbb"
        assert r.actual_profit_usd == 5.0
        assert r.slippage_pct == 0.1

    def test_opportunity_roundtrip_with_expiry(self, store: ArbitrageStore):
        """Full opportunity roundtrip with expiry set."""
        opp = make_opportunity(spread_pct=2.0, profit_after_gas=15.0, trade_size=200.0)
        store.record_opportunity(opp)
        loaded = store.load_opportunities()
        assert len(loaded) == 1
        assert loaded[0].trade_size_usd == 200.0
        assert loaded[0].expires_at is not None

    def test_opportunity_roundtrip_no_expiry(self, store: ArbitrageStore):
        """Opportunity with no expiry should roundtrip."""
        opp = make_opportunity()
        opp.expires_at = None
        store.record_opportunity(opp)
        loaded = store.load_opportunities()
        assert len(loaded) == 1
        assert loaded[0].expires_at is None

    def test_load_seen_ids_corrupt(self, tmp_data_dir: Path):
        """Corrupt lines in seen IDs loading should be skipped."""
        opp_file = tmp_data_dir / "opportunities.jsonl"
        with open(opp_file, "w") as f:
            f.write("not json at all\n")
            f.write("{}\n")  # Missing opportunity_id
        store = ArbitrageStore(tmp_data_dir)
        # Should have loaded the empty-id from {} and skipped bad line
        assert store.opportunity_count() == 1

    def test_result_with_error(self, store: ArbitrageStore):
        result = make_result(error="insufficient funds")
        store.record_result(result)
        loaded = store.load_results()
        assert loaded[0].error == "insufficient funds"
