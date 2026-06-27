"""Tests for MEV data models."""

from __future__ import annotations

from animus_mev.models import (
    DEX_ROUTERS,
    SWAP_SELECTORS,
    Chain,
    ExtractionResult,
    MempoolTx,
    MEVOpportunity,
    MEVType,
)


class TestMEVType:
    def test_values(self):
        assert MEVType.BACKRUN.value == "backrun"
        assert MEVType.ARBITRAGE.value == "arbitrage"
        assert MEVType.LIQUIDATION.value == "liquidation"
        assert MEVType.SANDWICH_DETECT.value == "sandwich_detect"

    def test_from_string(self):
        assert MEVType("backrun") == MEVType.BACKRUN
        assert MEVType("sandwich_detect") == MEVType.SANDWICH_DETECT


class TestChain:
    def test_values(self):
        assert Chain.BASE.value == "base"
        assert Chain.ARB.value == "arb"
        assert Chain.OP.value == "op"
        assert Chain.ETH.value == "eth"

    def test_from_string(self):
        assert Chain("base") == Chain.BASE


class TestDEXRouters:
    def test_base_routers_exist(self):
        assert Chain.BASE in DEX_ROUTERS
        assert len(DEX_ROUTERS[Chain.BASE]) >= 1

    def test_all_chains_have_routers(self):
        for chain in Chain:
            assert chain in DEX_ROUTERS

    def test_addresses_are_lowercase(self):
        for chain, routers in DEX_ROUTERS.items():
            for addr in routers:
                assert addr == addr.lower(), f"{addr} on {chain} is not lowercase"


class TestSwapSelectors:
    def test_known_selectors(self):
        assert "0x04e45aaf" in SWAP_SELECTORS
        assert SWAP_SELECTORS["0x04e45aaf"] == "exactInputSingle"

    def test_has_common_selectors(self):
        assert len(SWAP_SELECTORS) >= 4


class TestMempoolTx:
    def test_create(self, sample_tx):
        assert sample_tx.chain == Chain.BASE
        assert sample_tx.tx_hash == "0xabc123def456"
        assert sample_tx.value == int(5e18)

    def test_dex_detected_true(self, sample_tx):
        assert sample_tx.dex_detected is True

    def test_dex_detected_false(self, small_tx):
        assert small_tx.dex_detected is False

    def test_swap_selector_detected(self, sample_tx):
        assert sample_tx.swap_selector == "exactInputSingle"

    def test_swap_selector_none(self, small_tx):
        assert small_tx.swap_selector is None

    def test_swap_selector_short_input(self):
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0xshort",
            from_addr="0x1",
            to_addr="0x2",
            value=0,
            gas_price=0,
            input_data="0x04e4",
        )
        assert tx.swap_selector is None

    def test_swap_amount_usd(self, sample_tx):
        # 5 ETH * 3000 = 15000 USD
        assert sample_tx.swap_amount_usd == 15000.0

    def test_swap_amount_usd_zero(self):
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0x",
            from_addr="0x1",
            to_addr="0x2",
            value=0,
            gas_price=0,
            input_data="0x",
        )
        assert tx.swap_amount_usd == 0.0

    def test_to_dict(self, sample_tx):
        d = sample_tx.to_dict()
        assert d["chain"] == "base"
        assert d["tx_hash"] == "0xabc123def456"
        assert d["dex_detected"] is True
        assert d["swap_selector"] == "exactInputSingle"
        assert "timestamp" in d

    def test_to_dict_truncates_long_input(self):
        tx = MempoolTx(
            chain=Chain.BASE,
            tx_hash="0x",
            from_addr="0x1",
            to_addr="0x2",
            value=0,
            gas_price=0,
            input_data="0x" + "a" * 200,
        )
        d = tx.to_dict()
        assert len(d["input_data"]) == 66

    def test_from_dict(self, sample_tx):
        d = sample_tx.to_dict()
        # from_dict needs full input_data for roundtrip
        d["input_data"] = sample_tx.input_data
        restored = MempoolTx.from_dict(d)
        assert restored.chain == sample_tx.chain
        assert restored.tx_hash == sample_tx.tx_hash
        assert restored.value == sample_tx.value

    def test_from_dict_defaults(self):
        d = {
            "chain": "base",
            "tx_hash": "0x123",
            "from_addr": "0x1",
            "to_addr": "0x2",
            "value": 0,
            "gas_price": 0,
        }
        tx = MempoolTx.from_dict(d)
        assert tx.input_data == "0x"
        assert tx.block_number is None


class TestMEVOpportunity:
    def test_create(self, sample_opportunity):
        assert sample_opportunity.mev_type == MEVType.BACKRUN
        assert sample_opportunity.chain == Chain.BASE
        assert sample_opportunity.estimated_profit_usd == 5.0

    def test_opportunity_id_deterministic(self, sample_opportunity):
        id1 = sample_opportunity.opportunity_id
        id2 = sample_opportunity.opportunity_id
        assert id1 == id2
        assert len(id1) == 16

    def test_net_profit(self, sample_opportunity):
        assert sample_opportunity.net_profit_usd == 4.9

    def test_is_profitable(self, sample_opportunity):
        assert sample_opportunity.is_profitable is True

    def test_not_profitable(self, sample_tx):
        opp = MEVOpportunity(
            mev_type=MEVType.BACKRUN,
            chain=Chain.BASE,
            trigger_tx=sample_tx,
            estimated_profit_usd=0.05,
            gas_cost_usd=0.10,
            confidence=0.5,
        )
        assert opp.is_profitable is False

    def test_to_dict(self, sample_opportunity):
        d = sample_opportunity.to_dict()
        assert d["mev_type"] == "backrun"
        assert d["chain"] == "base"
        assert d["net_profit_usd"] == 4.9
        assert "trigger_tx" in d
        assert "opportunity_id" in d

    def test_from_dict(self, sample_opportunity):
        d = sample_opportunity.to_dict()
        # Reconstruct full input_data for trigger_tx
        d["trigger_tx"]["input_data"] = sample_opportunity.trigger_tx.input_data
        restored = MEVOpportunity.from_dict(d)
        assert restored.mev_type == sample_opportunity.mev_type
        assert restored.estimated_profit_usd == sample_opportunity.estimated_profit_usd

    def test_from_dict_defaults(self, sample_tx):
        d = {
            "mev_type": "backrun",
            "chain": "base",
            "trigger_tx": sample_tx.to_dict(),
            "estimated_profit_usd": 1.0,
            "gas_cost_usd": 0.1,
        }
        d["trigger_tx"]["input_data"] = sample_tx.input_data
        opp = MEVOpportunity.from_dict(d)
        assert opp.confidence == 0.0
        assert opp.expires_at_block is None


class TestExtractionResult:
    def test_create(self, sample_result):
        assert sample_result.executed is True
        assert sample_result.actual_profit_usd == 4.5

    def test_net_profit(self, sample_result):
        assert sample_result.net_profit_usd == 4.42

    def test_success(self, sample_result):
        assert sample_result.success is True

    def test_not_success_error(self):
        r = ExtractionResult(
            opportunity_id="abc",
            executed=True,
            error="reverted",
        )
        assert r.success is False

    def test_not_success_not_executed(self):
        r = ExtractionResult(
            opportunity_id="abc",
            executed=False,
        )
        assert r.success is False

    def test_not_success_negative_profit(self):
        r = ExtractionResult(
            opportunity_id="abc",
            executed=True,
            actual_profit_usd=0.01,
            gas_used_usd=0.10,
        )
        assert r.success is False

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["opportunity_id"] == "abc123"
        assert d["executed"] is True
        assert d["net_profit_usd"] == 4.42

    def test_from_dict(self, sample_result):
        d = sample_result.to_dict()
        restored = ExtractionResult.from_dict(d)
        assert restored.opportunity_id == sample_result.opportunity_id
        assert restored.actual_profit_usd == sample_result.actual_profit_usd

    def test_from_dict_defaults(self):
        d = {"opportunity_id": "xyz", "executed": False}
        r = ExtractionResult.from_dict(d)
        assert r.tx_hash is None
        assert r.actual_profit_usd == 0.0
        assert r.error is None
