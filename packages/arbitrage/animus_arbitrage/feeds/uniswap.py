"""Uniswap V3 price feed adapter."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from animus_arbitrage.feeds.base import BaseFeed
from animus_arbitrage.models import DEX, Chain, PriceQuote, TokenPair

logger = logging.getLogger("animus_arbitrage.feeds.uniswap")

# Uniswap V3 subgraph URLs by chain
SUBGRAPH_URLS: dict[Chain, str] = {
    Chain.ETH: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
    Chain.BASE: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-base",
    Chain.ARB: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-arbitrum",
    Chain.OP: "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-optimism",
}

# Default gas estimates in USD by chain
GAS_ESTIMATES: dict[Chain, float] = {
    Chain.ETH: 5.0,
    Chain.BASE: 0.05,
    Chain.ARB: 0.15,
    Chain.OP: 0.10,
}


class UniswapV3Feed(BaseFeed):
    """Uniswap V3 price feed via subgraph API."""

    def __init__(
        self,
        chain: Chain = Chain.ETH,
        rpc_url: str = "",
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(dex=DEX.UNISWAP_V3, chain=chain, rpc_url=rpc_url)
        self._client = client
        self._subgraph_url = SUBGRAPH_URLS.get(chain, SUBGRAPH_URLS[Chain.ETH])

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def get_quote(self, pair: TokenPair) -> PriceQuote | None:
        """Fetch price from Uniswap V3 subgraph."""
        try:
            client = await self._get_client()
            query = {
                "query": (
                    "{ pools("
                    f'where: {{ token0_: {{symbol: "{pair.base_token}"}}, '
                    f'token1_: {{symbol: "{pair.quote_token}"}} }}, '
                    "first: 1, orderBy: totalValueLockedUSD, orderDirection: desc"
                    ") { token0Price token1Price totalValueLockedUSD } }"
                )
            }
            response = await client.post(self._subgraph_url, json=query)
            response.raise_for_status()
            data = response.json()

            pools = data.get("data", {}).get("pools", [])
            if not pools:
                return None

            pool = pools[0]
            price = float(pool["token1Price"])
            liquidity = float(pool["totalValueLockedUSD"])

            if price <= 0:
                return None

            return PriceQuote(
                dex=self.dex,
                pair=pair,
                price=price,
                liquidity_usd=liquidity,
                timestamp=datetime.now(UTC),
                gas_estimate_usd=GAS_ESTIMATES.get(self.chain, 5.0),
            )
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Uniswap quote failed for %s: %s", pair.symbol, e)
            self._healthy = False
            return None

    async def get_supported_pairs(self) -> list[TokenPair]:
        """Return common Uniswap V3 pairs for this chain."""
        common = [
            ("ETH", "USDC"),
            ("ETH", "USDT"),
            ("WBTC", "ETH"),
            ("ETH", "DAI"),
        ]
        return [
            TokenPair(base_token=base, quote_token=quote, chain=self.chain)
            for base, quote in common
        ]
