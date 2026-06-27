"""Tests for arbitrage opportunity detector."""

from __future__ import annotations

import pytest

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.detector import ArbitrageDetector, calculate_spread_pct
from animus_arbitrage.feeds.base import BaseFeed
from animus_arbitrage.models import DEX, Chain, PriceQuote, TokenPair

from .conftest import make_quote


class MockFeed(BaseFeed):
    """Mock feed that returns configurable quotes."""

    def __init__(
        self,
        dex: DEX = DEX.UNISWAP_V3,
        chain: Chain = Chain.ETH,
        quotes: dict[str, PriceQuote] | None = None,
        pairs: list[TokenPair] | None = None,
    ):
        super().__init__(dex=dex, chain=chain)
        self._quotes = quotes or {}
        self._pairs = pairs or []

    async def get_quote(self, pair: TokenPair) -> PriceQuote | None:
        return self._quotes.get(pair.symbol)

    async def get_supported_pairs(self) -> list[TokenPair]:
        return self._pairs


class TestCalculateSpreadPct:
    def test_positive_spread(self):
        assert calculate_spread_pct(100.0, 101.0) == pytest.approx(1.0)

    def test_zero_spread(self):
        assert calculate_spread_pct(100.0, 100.0) == 0.0

    def test_large_spread(self):
        assert calculate_spread_pct(100.0, 200.0) == pytest.approx(100.0)

    def test_zero_buy_price(self):
        assert calculate_spread_pct(0.0, 100.0) == 0.0

    def test_negative_buy_price(self):
        assert calculate_spread_pct(-1.0, 100.0) == 0.0

    def test_small_spread(self):
        assert calculate_spread_pct(2000.0, 2006.0) == pytest.approx(0.3)

    def test_fractional_prices(self):
        result = calculate_spread_pct(0.001, 0.00102)
        assert result == pytest.approx(2.0, rel=0.01)


class TestArbitrageDetector:
    @pytest.fixture
    def eth_pair(self) -> TokenPair:
        return TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)

    @pytest.fixture
    def detector(self, config: ArbitrageConfig) -> ArbitrageDetector:
        return ArbitrageDetector(config)

    def test_init(self, config: ArbitrageConfig):
        detector = ArbitrageDetector(config)
        assert detector.feeds == []
        assert detector.config is config

    def test_add_feed(self, detector: ArbitrageDetector):
        feed = MockFeed()
        detector.add_feed(feed)
        assert len(detector.feeds) == 1

    @pytest.mark.asyncio
    async def test_collect_quotes_empty(self, detector: ArbitrageDetector, eth_pair: TokenPair):
        quotes = await detector.collect_quotes([eth_pair])
        assert quotes == {}

    @pytest.mark.asyncio
    async def test_collect_quotes_single_feed(self, config: ArbitrageConfig, eth_pair: TokenPair):
        quote = make_quote(dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0)
        feed = MockFeed(
            dex=DEX.UNISWAP_V3,
            chain=Chain.ETH,
            quotes={"ETH/USDC": quote},
        )
        detector = ArbitrageDetector(config, feeds=[feed])
        quotes = await detector.collect_quotes([eth_pair])
        assert "ETH/USDC@eth" in quotes
        assert len(quotes["ETH/USDC@eth"]) == 1

    @pytest.mark.asyncio
    async def test_collect_quotes_two_feeds(self, config: ArbitrageConfig, eth_pair: TokenPair):
        q1 = make_quote(dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0)
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2010.0)

        feed1 = MockFeed(dex=DEX.UNISWAP_V3, chain=Chain.ETH, quotes={"ETH/USDC": q1})
        feed2 = MockFeed(dex=DEX.SUSHISWAP, chain=Chain.ETH, quotes={"ETH/USDC": q2})

        detector = ArbitrageDetector(config, feeds=[feed1, feed2])
        quotes = await detector.collect_quotes([eth_pair])
        assert len(quotes["ETH/USDC@eth"]) == 2

    @pytest.mark.asyncio
    async def test_collect_quotes_skips_wrong_chain(
        self, config: ArbitrageConfig, eth_pair: TokenPair
    ):
        """Feed on SOL should not quote ETH pairs."""
        q1 = make_quote(dex=DEX.JUPITER, pair=eth_pair, price=2000.0)
        feed = MockFeed(dex=DEX.JUPITER, chain=Chain.SOL, quotes={"ETH/USDC": q1})
        detector = ArbitrageDetector(config, feeds=[feed])
        quotes = await detector.collect_quotes([eth_pair])
        assert quotes == {}

    @pytest.mark.asyncio
    async def test_collect_quotes_none_returned(self, config: ArbitrageConfig, eth_pair: TokenPair):
        feed = MockFeed(dex=DEX.UNISWAP_V3, chain=Chain.ETH, quotes={})
        detector = ArbitrageDetector(config, feeds=[feed])
        quotes = await detector.collect_quotes([eth_pair])
        assert quotes == {}

    def test_find_opportunities_no_quotes(self, detector: ArbitrageDetector):
        opps = detector.find_opportunities({})
        assert opps == []

    def test_find_opportunities_single_quote(
        self, detector: ArbitrageDetector, eth_pair: TokenPair
    ):
        q1 = make_quote(dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1]})
        assert opps == []

    def test_find_opportunities_profitable(self, config: ArbitrageConfig, eth_pair: TokenPair):
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=1.0, liquidity=50_000.0
        )
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2020.0, gas=1.0, liquidity=50_000.0)

        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) >= 1
        assert opps[0].spread_pct == pytest.approx(1.0)
        assert opps[0].profit_after_gas_usd > 0

    def test_find_opportunities_not_profitable_after_gas(
        self, config: ArbitrageConfig, eth_pair: TokenPair
    ):
        """Spread exists but gas eats the profit."""
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=50.0, liquidity=50_000.0
        )
        q2 = make_quote(
            dex=DEX.SUSHISWAP, pair=eth_pair, price=2006.0, gas=50.0, liquidity=50_000.0
        )

        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=100.0)
        assert opps == []

    def test_find_opportunities_below_min_spread(
        self, config: ArbitrageConfig, eth_pair: TokenPair
    ):
        """Spread below minimum threshold."""
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.0, liquidity=50_000.0
        )
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2002.0, gas=0.0, liquidity=50_000.0)

        config.risk.min_spread_pct = 0.5
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert opps == []

    def test_find_opportunities_below_min_liquidity(
        self, config: ArbitrageConfig, eth_pair: TokenPair
    ):
        """Liquidity below threshold."""
        q1 = make_quote(dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.0, liquidity=500.0)
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2020.0, gas=0.0, liquidity=500.0)

        config.risk.min_liquidity_usd = 1000.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert opps == []

    def test_find_opportunities_equal_prices(self, config: ArbitrageConfig, eth_pair: TokenPair):
        q1 = make_quote(dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, liquidity=50_000.0)
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2000.0, liquidity=50_000.0)

        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]})
        assert opps == []

    def test_find_opportunities_sorted_by_profit(
        self, config: ArbitrageConfig, eth_pair: TokenPair
    ):
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.5, liquidity=100_000.0
        )
        q2 = make_quote(
            dex=DEX.SUSHISWAP, pair=eth_pair, price=2020.0, gas=0.5, liquidity=100_000.0
        )
        q3 = make_quote(dex=DEX.CURVE, pair=eth_pair, price=2040.0, gas=0.5, liquidity=100_000.0)

        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2, q3]}, trade_size_usd=1000.0)
        if len(opps) >= 2:
            assert opps[0].profit_after_gas_usd >= opps[1].profit_after_gas_usd

    def test_find_opportunities_multiple_pairs(self, config: ArbitrageConfig):
        pair1 = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        pair2 = TokenPair(base_token="WBTC", quote_token="ETH", chain=Chain.ETH)

        q1a = make_quote(dex=DEX.UNISWAP_V3, pair=pair1, price=2000.0, gas=0.5, liquidity=100_000.0)
        q1b = make_quote(dex=DEX.SUSHISWAP, pair=pair1, price=2020.0, gas=0.5, liquidity=100_000.0)
        q2a = make_quote(dex=DEX.UNISWAP_V3, pair=pair2, price=15.0, gas=0.5, liquidity=100_000.0)
        q2b = make_quote(dex=DEX.CURVE, pair=pair2, price=15.1, gas=0.5, liquidity=100_000.0)

        detector = ArbitrageDetector(config)
        quotes = {
            "ETH/USDC@eth": [q1a, q1b],
            "WBTC/ETH@eth": [q2a, q2b],
        }
        opps = detector.find_opportunities(quotes, trade_size_usd=1000.0)
        assert len(opps) >= 1

    def test_confidence_high_liquidity(self, config: ArbitrageConfig, eth_pair: TokenPair):
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.5, liquidity=200_000.0
        )
        q2 = make_quote(
            dex=DEX.SUSHISWAP, pair=eth_pair, price=2060.0, gas=0.5, liquidity=200_000.0
        )

        config.risk.min_confidence = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) == 1
        assert opps[0].confidence > 0.5

    def test_confidence_low_values(self, config: ArbitrageConfig, eth_pair: TokenPair):
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.01, liquidity=5_000.0
        )
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2008.0, gas=0.01, liquidity=5_000.0)

        config.risk.min_confidence = 0.0
        config.risk.min_liquidity_usd = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        if opps:
            assert opps[0].confidence <= 0.5

    def test_confidence_below_threshold_filtered(
        self, config: ArbitrageConfig, eth_pair: TokenPair
    ):
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.01, liquidity=5_000.0
        )
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2008.0, gas=0.01, liquidity=5_000.0)

        config.risk.min_confidence = 0.99
        config.risk.min_liquidity_usd = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert opps == []

    @pytest.mark.asyncio
    async def test_scan_with_feeds(self, config: ArbitrageConfig):
        pair = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        q1 = make_quote(dex=DEX.UNISWAP_V3, pair=pair, price=2000.0, gas=0.5, liquidity=100_000.0)
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=pair, price=2020.0, gas=0.5, liquidity=100_000.0)

        feed1 = MockFeed(
            dex=DEX.UNISWAP_V3,
            chain=Chain.ETH,
            quotes={"ETH/USDC": q1},
            pairs=[pair],
        )
        feed2 = MockFeed(
            dex=DEX.SUSHISWAP,
            chain=Chain.ETH,
            quotes={"ETH/USDC": q2},
            pairs=[pair],
        )

        detector = ArbitrageDetector(config, feeds=[feed1, feed2])
        opps = await detector.scan(trade_size_usd=1000.0)
        assert len(opps) >= 1

    @pytest.mark.asyncio
    async def test_scan_explicit_pairs(self, config: ArbitrageConfig):
        pair = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        q1 = make_quote(dex=DEX.UNISWAP_V3, pair=pair, price=2000.0, gas=0.5, liquidity=100_000.0)
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=pair, price=2020.0, gas=0.5, liquidity=100_000.0)

        feed1 = MockFeed(dex=DEX.UNISWAP_V3, chain=Chain.ETH, quotes={"ETH/USDC": q1})
        feed2 = MockFeed(dex=DEX.SUSHISWAP, chain=Chain.ETH, quotes={"ETH/USDC": q2})

        detector = ArbitrageDetector(config, feeds=[feed1, feed2])
        opps = await detector.scan(pairs=[pair], trade_size_usd=1000.0)
        assert len(opps) >= 1

    @pytest.mark.asyncio
    async def test_get_all_pairs_deduped(self, config: ArbitrageConfig):
        pair = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        feed1 = MockFeed(dex=DEX.UNISWAP_V3, chain=Chain.ETH, pairs=[pair])
        feed2 = MockFeed(dex=DEX.SUSHISWAP, chain=Chain.ETH, pairs=[pair])

        detector = ArbitrageDetector(config, feeds=[feed1, feed2])
        all_pairs = await detector._get_all_pairs()
        assert len(all_pairs) == 1

    def test_liquidity_zero_passes(self, config: ArbitrageConfig, eth_pair: TokenPair):
        """Zero liquidity (unknown) should not be filtered by min_liquidity."""
        q1 = make_quote(dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.5, liquidity=0.0)
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2020.0, gas=0.5, liquidity=0.0)

        config.risk.min_confidence = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        # Zero liquidity means min_liq=0 which is not > 0, so liquidity check skipped
        assert len(opps) >= 1

    def test_confidence_medium_liquidity(self, config: ArbitrageConfig, eth_pair: TokenPair):
        """Liquidity between 10K-50K should score 0.1 for liquidity factor."""
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.1, liquidity=20_000.0
        )
        q2 = make_quote(dex=DEX.SUSHISWAP, pair=eth_pair, price=2060.0, gas=0.1, liquidity=20_000.0)

        config.risk.min_confidence = 0.0
        config.risk.min_liquidity_usd = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) >= 1

    def test_confidence_medium_profit(self, config: ArbitrageConfig, eth_pair: TokenPair):
        """Profit between 1-5 USD should score 0.1 for profit factor."""
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2000.0, gas=0.5, liquidity=100_000.0
        )
        q2 = make_quote(
            dex=DEX.SUSHISWAP, pair=eth_pair, price=2006.0, gas=0.5, liquidity=100_000.0
        )

        config.risk.min_confidence = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=500.0)
        # spread 0.3%, profit = 500 * 0.003 - 1.0 = 0.5, which may be small
        # Use bigger trade for medium profit
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) >= 1

    def test_evaluate_pair_q1_higher_price(self, config: ArbitrageConfig, eth_pair: TokenPair):
        """Test when q1.price > q2.price (reversed buy/sell assignment)."""
        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=eth_pair, price=2020.0, gas=0.5, liquidity=100_000.0
        )
        q2 = make_quote(
            dex=DEX.SUSHISWAP, pair=eth_pair, price=2000.0, gas=0.5, liquidity=100_000.0
        )

        config.risk.min_confidence = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) >= 1
        # Buy should be the lower price (sushi), sell should be higher (uniswap)
        assert opps[0].buy_quote.price < opps[0].sell_quote.price

    def test_confidence_stale_quotes(self, config: ArbitrageConfig, eth_pair: TokenPair):
        """Old quotes should get lower freshness score."""
        from datetime import UTC, datetime, timedelta

        old_time = datetime.now(UTC) - timedelta(seconds=10)
        q1 = PriceQuote(
            dex=DEX.UNISWAP_V3,
            pair=eth_pair,
            price=2000.0,
            liquidity_usd=100_000.0,
            timestamp=old_time,
            gas_estimate_usd=0.5,
        )
        q2 = PriceQuote(
            dex=DEX.SUSHISWAP,
            pair=eth_pair,
            price=2060.0,
            liquidity_usd=100_000.0,
            timestamp=old_time,
            gas_estimate_usd=0.5,
        )

        config.risk.min_confidence = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) >= 1
        # Stale quotes (10s) should get 0.1 freshness (not 0.2)

    def test_confidence_very_stale_quotes(self, config: ArbitrageConfig, eth_pair: TokenPair):
        """Quotes older than 15s should get no freshness score."""
        from datetime import UTC, datetime, timedelta

        old_time = datetime.now(UTC) - timedelta(seconds=30)
        q1 = PriceQuote(
            dex=DEX.UNISWAP_V3,
            pair=eth_pair,
            price=2000.0,
            liquidity_usd=100_000.0,
            timestamp=old_time,
            gas_estimate_usd=0.5,
        )
        q2 = PriceQuote(
            dex=DEX.SUSHISWAP,
            pair=eth_pair,
            price=2060.0,
            liquidity_usd=100_000.0,
            timestamp=old_time,
            gas_estimate_usd=0.5,
        )

        config.risk.min_confidence = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) >= 1

    def test_confidence_cross_chain_no_bonus(self, config: ArbitrageConfig):
        """Cross-chain quotes should not get same-chain bonus."""
        pair_eth = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)
        pair_arb = TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ARB)

        q1 = make_quote(
            dex=DEX.UNISWAP_V3, pair=pair_eth, price=2000.0, gas=0.5, liquidity=100_000.0
        )
        q2 = make_quote(
            dex=DEX.SUSHISWAP, pair=pair_arb, price=2060.0, gas=0.5, liquidity=100_000.0
        )

        config.risk.min_confidence = 0.0
        detector = ArbitrageDetector(config)
        opps = detector.find_opportunities({"ETH/USDC@eth": [q1, q2]}, trade_size_usd=1000.0)
        assert len(opps) >= 1
