"""Jupiter (Solana) aggregator price feed."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from animus_arbitrage.feeds.base import BaseFeed
from animus_arbitrage.models import DEX, Chain, PriceQuote, TokenPair

logger = logging.getLogger("animus_arbitrage.feeds.jupiter")

JUPITER_PRICE_API = "https://price.jup.ag/v6/price"

# Well-known Solana token mints
SOL_MINTS: dict[str, str] = {
    "SOL": "So11111111111111111111111111111111111111112",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
}

SOL_GAS_ESTIMATE_USD = 0.001


class JupiterFeed(BaseFeed):
    """Jupiter aggregator price feed for Solana."""

    def __init__(
        self,
        rpc_url: str = "",
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(dex=DEX.JUPITER, chain=Chain.SOL, rpc_url=rpc_url)
        self._client = client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def get_quote(self, pair: TokenPair) -> PriceQuote | None:
        """Fetch price from Jupiter price API."""
        try:
            mint = SOL_MINTS.get(pair.base_token)
            if not mint:
                mint = pair.base_address or pair.base_token

            client = await self._get_client()
            response = await client.get(
                JUPITER_PRICE_API,
                params={"ids": mint, "vsToken": pair.quote_token},
            )
            response.raise_for_status()
            data = response.json()

            price_data = data.get("data", {}).get(mint)
            if not price_data:
                return None

            price = float(price_data["price"])
            if price <= 0:
                return None

            return PriceQuote(
                dex=self.dex,
                pair=pair,
                price=price,
                liquidity_usd=0.0,  # Jupiter doesn't expose TVL in price API
                timestamp=datetime.now(UTC),
                gas_estimate_usd=SOL_GAS_ESTIMATE_USD,
            )
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Jupiter quote failed for %s: %s", pair.symbol, e)
            self._healthy = False
            return None

    async def get_supported_pairs(self) -> list[TokenPair]:
        """Return common Solana trading pairs."""
        common = [
            ("SOL", "USDC"),
            ("SOL", "USDT"),
            ("RAY", "USDC"),
            ("BONK", "USDC"),
        ]
        return [
            TokenPair(base_token=base, quote_token=quote, chain=Chain.SOL) for base, quote in common
        ]
