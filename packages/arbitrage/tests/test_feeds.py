"""Tests for price feed adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from animus_arbitrage.feeds.base import BaseFeed
from animus_arbitrage.feeds.cetus import CetusFeed
from animus_arbitrage.feeds.jupiter import JupiterFeed
from animus_arbitrage.feeds.uniswap import UniswapV3Feed
from animus_arbitrage.models import DEX, Chain, TokenPair


class TestBaseFeed:
    def test_name(self):
        class ConcreteFeed(BaseFeed):
            async def get_quote(self, pair):
                return None

            async def get_supported_pairs(self):
                return []

        feed = ConcreteFeed(dex=DEX.UNISWAP_V3, chain=Chain.ETH)
        assert feed.name == "uniswap_v3@eth"

    @pytest.mark.asyncio
    async def test_health_check_default(self):
        class ConcreteFeed(BaseFeed):
            async def get_quote(self, pair):
                return None

            async def get_supported_pairs(self):
                return []

        feed = ConcreteFeed(dex=DEX.UNISWAP_V3, chain=Chain.ETH)
        assert await feed.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        class ConcreteFeed(BaseFeed):
            async def get_quote(self, pair):
                return None

            async def get_supported_pairs(self):
                return []

        feed = ConcreteFeed(dex=DEX.UNISWAP_V3, chain=Chain.ETH)
        feed._healthy = False
        assert await feed.health_check() is False


class TestUniswapV3Feed:
    @pytest.fixture
    def pair(self) -> TokenPair:
        return TokenPair(base_token="ETH", quote_token="USDC", chain=Chain.ETH)

    def test_init(self):
        feed = UniswapV3Feed(chain=Chain.ETH)
        assert feed.dex == DEX.UNISWAP_V3
        assert feed.chain == Chain.ETH

    def test_init_with_chain(self):
        feed = UniswapV3Feed(chain=Chain.BASE)
        assert feed.chain == Chain.BASE

    @pytest.mark.asyncio
    async def test_get_quote_success(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "pools": [
                    {
                        "token1Price": "2000.5",
                        "totalValueLockedUSD": "150000000",
                    }
                ]
            }
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        feed = UniswapV3Feed(chain=Chain.ETH, client=mock_client)
        quote = await feed.get_quote(pair)

        assert quote is not None
        assert quote.price == 2000.5
        assert quote.liquidity_usd == 150_000_000
        assert quote.dex == DEX.UNISWAP_V3

    @pytest.mark.asyncio
    async def test_get_quote_no_pools(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"pools": []}}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        feed = UniswapV3Feed(chain=Chain.ETH, client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None

    @pytest.mark.asyncio
    async def test_get_quote_zero_price(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {"pools": [{"token1Price": "0", "totalValueLockedUSD": "100000"}]}
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        feed = UniswapV3Feed(chain=Chain.ETH, client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None

    @pytest.mark.asyncio
    async def test_get_quote_http_error(self, pair: TokenPair):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("timeout"))

        feed = UniswapV3Feed(chain=Chain.ETH, client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None
        assert feed._healthy is False

    @pytest.mark.asyncio
    async def test_get_supported_pairs(self):
        feed = UniswapV3Feed(chain=Chain.ETH)
        pairs = await feed.get_supported_pairs()
        assert len(pairs) > 0
        assert all(p.chain == Chain.ETH for p in pairs)

    @pytest.mark.asyncio
    async def test_get_supported_pairs_base(self):
        feed = UniswapV3Feed(chain=Chain.BASE)
        pairs = await feed.get_supported_pairs()
        assert all(p.chain == Chain.BASE for p in pairs)

    @pytest.mark.asyncio
    async def test_get_quote_creates_client(self, pair: TokenPair):
        """_get_client should create a client if none provided."""
        feed = UniswapV3Feed(chain=Chain.ETH)
        client = await feed._get_client()
        assert client is not None
        assert feed._client is client
        await client.aclose()


class TestJupiterFeed:
    @pytest.fixture
    def pair(self) -> TokenPair:
        return TokenPair(base_token="SOL", quote_token="USDC", chain=Chain.SOL)

    def test_init(self):
        feed = JupiterFeed()
        assert feed.dex == DEX.JUPITER
        assert feed.chain == Chain.SOL

    @pytest.mark.asyncio
    async def test_get_quote_success(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "So11111111111111111111111111111111111111112": {
                    "price": 150.5,
                }
            }
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        feed = JupiterFeed(client=mock_client)
        quote = await feed.get_quote(pair)

        assert quote is not None
        assert quote.price == 150.5
        assert quote.dex == DEX.JUPITER

    @pytest.mark.asyncio
    async def test_get_quote_unknown_token(self, pair: TokenPair):
        """Unknown token uses address or token name as key."""
        unknown_pair = TokenPair(
            base_token="UNKNOWN",
            quote_token="USDC",
            chain=Chain.SOL,
            base_address="0xunknown",
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"0xunknown": {"price": 1.5}}}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        feed = JupiterFeed(client=mock_client)
        quote = await feed.get_quote(unknown_pair)
        assert quote is not None
        assert quote.price == 1.5

    @pytest.mark.asyncio
    async def test_get_quote_no_data(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {}}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        feed = JupiterFeed(client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None

    @pytest.mark.asyncio
    async def test_get_quote_zero_price(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {"So11111111111111111111111111111111111111112": {"price": 0}}
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        feed = JupiterFeed(client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None

    @pytest.mark.asyncio
    async def test_get_quote_http_error(self, pair: TokenPair):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("timeout"))

        feed = JupiterFeed(client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None
        assert feed._healthy is False

    @pytest.mark.asyncio
    async def test_get_supported_pairs(self):
        feed = JupiterFeed()
        pairs = await feed.get_supported_pairs()
        assert len(pairs) > 0
        assert all(p.chain == Chain.SOL for p in pairs)

    @pytest.mark.asyncio
    async def test_get_quote_creates_client(self, pair: TokenPair):
        feed = JupiterFeed()
        client = await feed._get_client()
        assert client is not None
        await client.aclose()


class TestCetusFeed:
    @pytest.fixture
    def pair(self) -> TokenPair:
        return TokenPair(base_token="SUI", quote_token="USDC", chain=Chain.SUI)

    def test_init(self):
        feed = CetusFeed()
        assert feed.dex == DEX.CETUS
        assert feed.chain == Chain.SUI

    @pytest.mark.asyncio
    async def test_get_quote_success(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {"lp_list": [{"current_price": "1.23", "tvl_in_usd": "5000000"}]}
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        feed = CetusFeed(client=mock_client)
        quote = await feed.get_quote(pair)

        assert quote is not None
        assert quote.price == 1.23
        assert quote.liquidity_usd == 5_000_000
        assert quote.dex == DEX.CETUS

    @pytest.mark.asyncio
    async def test_get_quote_no_pools(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": {"lp_list": []}}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        feed = CetusFeed(client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None

    @pytest.mark.asyncio
    async def test_get_quote_zero_price(self, pair: TokenPair):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": {"lp_list": [{"current_price": "0", "tvl_in_usd": "100"}]}
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        feed = CetusFeed(client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None

    @pytest.mark.asyncio
    async def test_get_quote_http_error(self, pair: TokenPair):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("timeout"))

        feed = CetusFeed(client=mock_client)
        quote = await feed.get_quote(pair)
        assert quote is None
        assert feed._healthy is False

    @pytest.mark.asyncio
    async def test_get_supported_pairs(self):
        feed = CetusFeed()
        pairs = await feed.get_supported_pairs()
        assert len(pairs) > 0
        assert all(p.chain == Chain.SUI for p in pairs)

    @pytest.mark.asyncio
    async def test_get_quote_creates_client(self, pair: TokenPair):
        feed = CetusFeed()
        client = await feed._get_client()
        assert client is not None
        await client.aclose()
