"""Cetus (Sui) DEX price feed."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from animus_arbitrage.feeds.base import BaseFeed
from animus_arbitrage.models import DEX, Chain, PriceQuote, TokenPair

logger = logging.getLogger("animus_arbitrage.feeds.cetus")

CETUS_API = "https://api-sui.cetus.zone/v2"
SUI_GAS_ESTIMATE_USD = 0.005


class CetusFeed(BaseFeed):
    """Cetus DEX price feed for Sui network."""

    def __init__(
        self,
        rpc_url: str = "",
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(dex=DEX.CETUS, chain=Chain.SUI, rpc_url=rpc_url)
        self._client = client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def get_quote(self, pair: TokenPair) -> PriceQuote | None:
        """Fetch price from Cetus API."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{CETUS_API}/pools",
                params={
                    "coin_type_a": pair.base_address or pair.base_token,
                    "coin_type_b": pair.quote_address or pair.quote_token,
                },
            )
            response.raise_for_status()
            data = response.json()

            pools = data.get("data", {}).get("lp_list", [])
            if not pools:
                return None

            pool = pools[0]
            price = float(pool.get("current_price", 0))
            liquidity = float(pool.get("tvl_in_usd", 0))

            if price <= 0:
                return None

            return PriceQuote(
                dex=self.dex,
                pair=pair,
                price=price,
                liquidity_usd=liquidity,
                timestamp=datetime.now(UTC),
                gas_estimate_usd=SUI_GAS_ESTIMATE_USD,
            )
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Cetus quote failed for %s: %s", pair.symbol, e)
            self._healthy = False
            return None

    async def get_supported_pairs(self) -> list[TokenPair]:
        """Return common Sui trading pairs."""
        common = [
            ("SUI", "USDC"),
            ("SUI", "USDT"),
            ("CETUS", "SUI"),
        ]
        return [
            TokenPair(base_token=base, quote_token=quote, chain=Chain.SUI) for base, quote in common
        ]
