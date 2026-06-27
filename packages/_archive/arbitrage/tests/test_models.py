"""Tests for arbitrage domain models."""

from __future__ import annotations

import pytest

from animus_arbitrage.models import (
    DEX,
    ArbitrageOpportunity,
    Chain,
    ExecutionMode,
    PriceQuote,
    TokenPair,
    TradeResult,
)

from .conftest import make_quote


class TestDEX:
    def test_all_values(self):
        assert DEX.UNISWAP_V3 == "uniswap_v3"
        assert DEX.SUSHISWAP == "sushiswap"
        assert DEX.CURVE == "curve"
        assert DEX.JUPITER == "jupiter"
        assert DEX.CETUS == "cetus"
        assert DEX.TURBOS == "turbos"
        assert DEX.RAYDIUM == "raydium"

    def test_from_string(self):
        assert DEX("uniswap_v3") == DEX.UNISWAP_V3

    def test_invalid_dex(self):
        with pytest.raises(ValueError):
            DEX("not_a_dex")


class TestChain:
    def test_all_values(self):
        assert Chain.ETH == "eth"
        assert Chain.BASE == "base"
        assert Chain.ARB == "arb"
        assert Chain.OP == "op"
        assert Chain.SOL == "sol"
        assert Chain.SUI == "sui"

    def test_from_string(self):
        assert Chain("sui") == Chain.SUI


class TestExecutionMode:
    def test_dry_run(self):
        assert ExecutionMode.DRY_RUN == "dry_run"

    def test_live(self):
        assert ExecutionMode.LIVE == "live"


class TestTokenPair:
    def test_basic_creation(self, eth_usdc_pair: TokenPair):
        assert eth_usdc_pair.base_token == "ETH"
        assert eth_usdc_pair.quote_token == "USDC"
        assert eth_usdc_pair.chain == Chain.ETH

    def test_symbol(self, eth_usdc_pair: TokenPair):
        assert eth_usdc_pair.symbol == "ETH/USDC"

    def test_default_addresses(self, eth_usdc_pair: TokenPair):
        assert eth_usdc_pair.base_address == ""
        assert eth_usdc_pair.quote_address == ""

    def test_with_addresses(self):
        pair = TokenPair(
            base_token="ETH",
            quote_token="USDC",
            chain=Chain.ETH,
            base_address="0xabc",
            quote_address="0xdef",
        )
        assert pair.base_address == "0xabc"
        assert pair.quote_address == "0xdef"

    def test_hash(self, eth_usdc_pair: TokenPair):
        pair2 = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        assert hash(eth_usdc_pair) == hash(pair2)

    def test_equality(self, eth_usdc_pair: TokenPair):
        pair2 = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        assert eth_usdc_pair == pair2

    def test_inequality_different_chain(self, eth_usdc_pair: TokenPair):
        pair2 = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.BASE)
        assert eth_usdc_pair != pair2

    def test_inequality_different_token(self, eth_usdc_pair: TokenPair):
        pair2 = TokenPair(base_token="WBTC", quote_token="USDC", chain=Chain.ETH)
        assert eth_usdc_pair != pair2

    def test_inequality_not_token_pair(self, eth_usdc_pair: TokenPair):
        assert eth_usdc_pair.__eq__("not a pair") is NotImplemented

    def test_hashable_in_set(self, eth_usdc_pair: TokenPair):
        pair2 = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        s = {eth_usdc_pair, pair2}
        assert len(s) == 1


class TestPriceQuote:
    def test_basic_creation(self, eth_usdc_pair: TokenPair):
        quote = PriceQuote(
            dex=DEX.UNISWAP_V3,
            pair=eth_usdc_pair,
            price=2000.0,
            liquidity_usd=100_000.0,
        )
        assert quote.dex == DEX.UNISWAP_V3
        assert quote.price == 2000.0

    def test_defaults(self, eth_usdc_pair: TokenPair):
        quote = PriceQuote(
            dex=DEX.UNISWAP_V3,
            pair=eth_usdc_pair,
            price=2000.0,
            liquidity_usd=100_000.0,
        )
        assert quote.gas_estimate_usd == 0.0
        assert quote.block_number is None
        assert quote.timestamp is not None

    def test_invalid_price(self, eth_usdc_pair: TokenPair):
        with pytest.raises(ValueError):
            PriceQuote(
                dex=DEX.UNISWAP_V3,
                pair=eth_usdc_pair,
                price=0.0,
                liquidity_usd=100_000.0,
            )

    def test_negative_price(self, eth_usdc_pair: TokenPair):
        with pytest.raises(ValueError):
            PriceQuote(
                dex=DEX.UNISWAP_V3,
                pair=eth_usdc_pair,
                price=-1.0,
                liquidity_usd=100_000.0,
            )

    def test_negative_liquidity(self, eth_usdc_pair: TokenPair):
        with pytest.raises(ValueError):
            PriceQuote(
                dex=DEX.UNISWAP_V3,
                pair=eth_usdc_pair,
                price=2000.0,
                liquidity_usd=-1.0,
            )

    def test_to_dict(self, eth_usdc_pair: TokenPair):
        quote = PriceQuote(
            dex=DEX.UNISWAP_V3,
            pair=eth_usdc_pair,
            price=2000.0,
            liquidity_usd=100_000.0,
            gas_estimate_usd=5.0,
            block_number=12345,
        )
        d = quote.to_dict()
        assert d["dex"] == "uniswap_v3"
        assert d["price"] == 2000.0
        assert d["pair"]["chain"] == "eth"
        assert d["gas_estimate_usd"] == 5.0
        assert d["block_number"] == 12345

    def test_to_dict_no_block(self, eth_usdc_pair: TokenPair):
        quote = PriceQuote(
            dex=DEX.UNISWAP_V3,
            pair=eth_usdc_pair,
            price=2000.0,
            liquidity_usd=100_000.0,
        )
        d = quote.to_dict()
        assert d["block_number"] is None


class TestArbitrageOpportunity:
    def test_basic_creation(self, eth_usdc_pair: TokenPair):
        buy = make_quote(dex=DEX.UNISWAP_V3, pair=eth_usdc_pair, price=2000.0)
        sell = make_quote(dex=DEX.SUSHISWAP, pair=eth_usdc_pair, price=2020.0)

        opp = ArbitrageOpportunity(
            buy_quote=buy,
            sell_quote=sell,
            spread_pct=1.0,
            profit_after_gas_usd=5.0,
            confidence=0.8,
        )
        assert opp.spread_pct == 1.0
        assert opp.profit_after_gas_usd == 5.0
        assert opp.opportunity_id is not None

    def test_auto_id(self):
        from .conftest import make_opportunity

        opp1 = make_opportunity()
        opp2 = make_opportunity()
        assert opp1.opportunity_id != opp2.opportunity_id

    def test_is_cross_chain_false(self, eth_usdc_pair: TokenPair):
        buy = make_quote(dex=DEX.UNISWAP_V3, pair=eth_usdc_pair, price=2000.0)
        sell = make_quote(dex=DEX.SUSHISWAP, pair=eth_usdc_pair, price=2020.0)
        opp = ArbitrageOpportunity(
            buy_quote=buy,
            sell_quote=sell,
            spread_pct=1.0,
            profit_after_gas_usd=5.0,
            confidence=0.8,
        )
        assert opp.is_cross_chain is False

    def test_is_cross_chain_true(self):
        pair_eth = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        pair_arb = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ARB)
        buy = make_quote(dex=DEX.UNISWAP_V3, pair=pair_eth, price=2000.0)
        sell = make_quote(dex=DEX.SUSHISWAP, pair=pair_arb, price=2020.0, chain=Chain.ARB)
        opp = ArbitrageOpportunity(
            buy_quote=buy,
            sell_quote=sell,
            spread_pct=1.0,
            profit_after_gas_usd=5.0,
            confidence=0.8,
        )
        assert opp.is_cross_chain is True

    def test_to_dict(self):
        from .conftest import make_opportunity

        opp = make_opportunity(spread_pct=1.5, profit_after_gas=10.0)
        d = opp.to_dict()
        assert d["spread_pct"] == 1.5
        assert d["profit_after_gas_usd"] == 10.0
        assert "opportunity_id" in d
        assert "detected_at" in d
        assert "is_cross_chain" in d
        assert "pair" in d

    def test_to_dict_no_expiry(self, eth_usdc_pair: TokenPair):
        buy = make_quote(pair=eth_usdc_pair, price=2000.0)
        sell = make_quote(dex=DEX.SUSHISWAP, pair=eth_usdc_pair, price=2020.0)
        opp = ArbitrageOpportunity(
            buy_quote=buy,
            sell_quote=sell,
            spread_pct=1.0,
            profit_after_gas_usd=5.0,
            confidence=0.8,
            expires_at=None,
        )
        d = opp.to_dict()
        assert d["expires_at"] is None

    def test_confidence_bounds(self, eth_usdc_pair: TokenPair):
        buy = make_quote(pair=eth_usdc_pair, price=2000.0)
        sell = make_quote(dex=DEX.SUSHISWAP, pair=eth_usdc_pair, price=2020.0)
        with pytest.raises(ValueError):
            ArbitrageOpportunity(
                buy_quote=buy,
                sell_quote=sell,
                spread_pct=1.0,
                profit_after_gas_usd=5.0,
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            ArbitrageOpportunity(
                buy_quote=buy,
                sell_quote=sell,
                spread_pct=1.0,
                profit_after_gas_usd=5.0,
                confidence=-0.1,
            )


class TestTradeResult:
    def test_basic_creation(self):
        result = TradeResult(
            opportunity_id="abc123",
            executed=True,
            mode=ExecutionMode.LIVE,
            buy_tx="0xaaa",
            sell_tx="0xbbb",
            actual_profit_usd=5.0,
        )
        assert result.executed is True
        assert result.success is True

    def test_dry_run_not_executed(self):
        result = TradeResult(
            opportunity_id="abc123",
            executed=False,
            mode=ExecutionMode.DRY_RUN,
        )
        assert result.executed is False
        assert result.success is False

    def test_failed_trade(self):
        result = TradeResult(
            opportunity_id="abc123",
            executed=True,
            mode=ExecutionMode.LIVE,
            error="insufficient funds",
        )
        assert result.executed is True
        assert result.success is False

    def test_to_dict(self):
        result = TradeResult(
            opportunity_id="abc123",
            executed=True,
            mode=ExecutionMode.LIVE,
            buy_tx="0xaaa",
            actual_profit_usd=5.0,
            slippage_pct=0.1,
        )
        d = result.to_dict()
        assert d["opportunity_id"] == "abc123"
        assert d["executed"] is True
        assert d["mode"] == "live"
        assert d["buy_tx"] == "0xaaa"
        assert d["actual_profit_usd"] == 5.0
        assert d["slippage_pct"] == 0.1
        assert d["success"] is True
        assert "timestamp" in d
        assert "result_id" in d

    def test_defaults(self):
        result = TradeResult(
            opportunity_id="abc123",
            executed=False,
        )
        assert result.mode == ExecutionMode.DRY_RUN
        assert result.buy_tx is None
        assert result.sell_tx is None
        assert result.actual_profit_usd == 0.0
        assert result.slippage_pct == 0.0
        assert result.error is None
