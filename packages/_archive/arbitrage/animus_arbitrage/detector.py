"""Cross-DEX arbitrage opportunity detector."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from itertools import combinations

from animus_arbitrage.config import ArbitrageConfig
from animus_arbitrage.feeds.base import BaseFeed
from animus_arbitrage.models import ArbitrageOpportunity, PriceQuote, TokenPair

logger = logging.getLogger("animus_arbitrage.detector")

# Default opportunity TTL
DEFAULT_EXPIRY_SECONDS = 30


class ArbitrageDetector:
    """Detects arbitrage opportunities across multiple DEX feeds.

    Compares prices for the same token pair across different DEXes,
    calculates gas-adjusted profitability, and filters by confidence
    and risk parameters.
    """

    def __init__(self, config: ArbitrageConfig, feeds: list[BaseFeed] | None = None):
        self.config = config
        self.feeds: list[BaseFeed] = feeds or []
        self._last_quotes: dict[str, list[PriceQuote]] = {}

    def add_feed(self, feed: BaseFeed) -> None:
        """Register a price feed."""
        self.feeds.append(feed)

    async def collect_quotes(self, pairs: list[TokenPair]) -> dict[str, list[PriceQuote]]:
        """Collect price quotes from all feeds for the given pairs.

        Returns a mapping of pair symbol -> list of quotes from different DEXes.
        """
        quotes: dict[str, list[PriceQuote]] = {}

        for pair in pairs:
            pair_key = f"{pair.symbol}@{pair.chain.value}"
            pair_quotes = []

            for feed in self.feeds:
                if feed.chain != pair.chain:
                    continue
                quote = await feed.get_quote(pair)
                if quote is not None:
                    pair_quotes.append(quote)

            if pair_quotes:
                quotes[pair_key] = pair_quotes

        self._last_quotes = quotes
        return quotes

    def find_opportunities(
        self,
        quotes: dict[str, list[PriceQuote]],
        trade_size_usd: float = 0.0,
    ) -> list[ArbitrageOpportunity]:
        """Analyze collected quotes to find arbitrage opportunities.

        Compares all quote pairs for the same token pair, calculates
        spread and gas-adjusted profit.
        """
        opportunities: list[ArbitrageOpportunity] = []
        effective_trade_size = trade_size_usd or self.config.execution.max_trade_size_usd

        for _pair_key, pair_quotes in quotes.items():
            if len(pair_quotes) < 2:
                continue

            for q1, q2 in combinations(pair_quotes, 2):
                opp = self._evaluate_pair(q1, q2, effective_trade_size)
                if opp is not None:
                    opportunities.append(opp)

        opportunities.sort(key=lambda o: o.profit_after_gas_usd, reverse=True)
        return opportunities

    def _evaluate_pair(
        self,
        q1: PriceQuote,
        q2: PriceQuote,
        trade_size_usd: float,
    ) -> ArbitrageOpportunity | None:
        """Evaluate a potential arbitrage between two quotes.

        Returns an ArbitrageOpportunity if profitable after gas, else None.
        """
        if q1.price == q2.price:
            return None

        # Determine buy/sell sides (buy low, sell high)
        if q1.price < q2.price:
            buy_quote, sell_quote = q1, q2
        else:
            buy_quote, sell_quote = q2, q1

        spread_pct = calculate_spread_pct(buy_quote.price, sell_quote.price)

        if spread_pct < self.config.risk.min_spread_pct:
            return None

        # Calculate gas-adjusted profit
        total_gas = buy_quote.gas_estimate_usd + sell_quote.gas_estimate_usd
        gross_profit = trade_size_usd * (spread_pct / 100.0)
        profit_after_gas = gross_profit - total_gas

        if profit_after_gas <= 0:
            return None

        # Check minimum liquidity
        min_liq = min(buy_quote.liquidity_usd, sell_quote.liquidity_usd)
        if min_liq > 0 and min_liq < self.config.risk.min_liquidity_usd:
            return None

        confidence = self._calculate_confidence(
            spread_pct, profit_after_gas, min_liq, buy_quote, sell_quote
        )

        if confidence < self.config.risk.min_confidence:
            return None

        now = datetime.now(UTC)
        return ArbitrageOpportunity(
            buy_quote=buy_quote,
            sell_quote=sell_quote,
            spread_pct=spread_pct,
            profit_after_gas_usd=profit_after_gas,
            confidence=confidence,
            detected_at=now,
            expires_at=now + timedelta(seconds=DEFAULT_EXPIRY_SECONDS),
            trade_size_usd=trade_size_usd,
        )

    def _calculate_confidence(
        self,
        spread_pct: float,
        profit_after_gas: float,
        min_liquidity: float,
        buy_quote: PriceQuote,
        sell_quote: PriceQuote,
    ) -> float:
        """Calculate confidence score for an opportunity.

        Factors: spread stability, liquidity depth, quote freshness.
        Returns 0.0-1.0.
        """
        score = 0.0

        # Spread factor: higher spread = more confidence (up to a point)
        if spread_pct >= 2.0:
            score += 0.3
        elif spread_pct >= 1.0:
            score += 0.2
        elif spread_pct >= 0.5:
            score += 0.1

        # Liquidity factor
        if min_liquidity >= 100_000:
            score += 0.3
        elif min_liquidity >= 50_000:
            score += 0.2
        elif min_liquidity >= 10_000:
            score += 0.1

        # Profit factor
        if profit_after_gas >= 10.0:
            score += 0.2
        elif profit_after_gas >= 5.0:
            score += 0.15
        elif profit_after_gas >= 1.0:
            score += 0.1

        # Freshness factor: quotes should be recent
        now = datetime.now(UTC)
        max_age = max(
            (now - buy_quote.timestamp).total_seconds(),
            (now - sell_quote.timestamp).total_seconds(),
        )
        if max_age < 5:
            score += 0.2
        elif max_age < 15:
            score += 0.1

        # Same chain bonus (no bridging risk)
        if buy_quote.pair.chain == sell_quote.pair.chain:
            score += 0.1

        return min(score, 1.0)

    async def scan(
        self,
        pairs: list[TokenPair] | None = None,
        trade_size_usd: float = 0.0,
    ) -> list[ArbitrageOpportunity]:
        """Full scan: collect quotes and find opportunities."""
        if pairs is None:
            pairs = await self._get_all_pairs()

        quotes = await self.collect_quotes(pairs)
        return self.find_opportunities(quotes, trade_size_usd)

    async def _get_all_pairs(self) -> list[TokenPair]:
        """Get union of all supported pairs across feeds."""
        all_pairs: list[TokenPair] = []
        seen: set[tuple[str, str, str]] = set()

        for feed in self.feeds:
            for pair in await feed.get_supported_pairs():
                key = (pair.base_token, pair.quote_token, pair.chain.value)
                if key not in seen:
                    seen.add(key)
                    all_pairs.append(pair)

        return all_pairs


def calculate_spread_pct(buy_price: float, sell_price: float) -> float:
    """Calculate the percentage spread between buy and sell prices.

    Args:
        buy_price: Lower price (buy side).
        sell_price: Higher price (sell side).

    Returns:
        Spread as a percentage. Always >= 0.
    """
    if buy_price <= 0:
        return 0.0
    return ((sell_price - buy_price) / buy_price) * 100.0
