"""Tests for hunters."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_prospector.hunters.aggregator_hunter import (
    AggregatorHunter,
    _extract_links,
    _guess_chain,
    _is_crypto_opportunity,
)
from animus_prospector.hunters.base import BaseHunter
from animus_prospector.hunters.social_hunter import (
    SocialHunter,
    _classify_reddit_post,
    _matches_opportunity,
)
from animus_prospector.hunters.web_hunter import WebHunter, _classify_opportunity
from animus_prospector.models import Chain, HunterResult, OpportunityType


class TestBaseHunter:
    @pytest.mark.asyncio
    async def test_scan_wraps_with_timing(self, sample_config):
        class TestHunter(BaseHunter):
            name = "test"

            async def _scan(self):
                return HunterResult(hunter_name="test")

        hunter = TestHunter(sample_config)
        result = await hunter.scan()
        assert result.hunter_name == "test"
        assert result.scan_duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_scan_handles_exception(self, sample_config):
        class FailHunter(BaseHunter):
            name = "fail"

            async def _scan(self):
                raise RuntimeError("boom")

        hunter = FailHunter(sample_config)
        result = await hunter.scan()
        assert len(result.errors) == 1
        assert "boom" in result.errors[0]

    @pytest.mark.asyncio
    async def test_default_health_check(self, sample_config):
        class TestHunter(BaseHunter):
            name = "test"

            async def _scan(self):
                return HunterResult(hunter_name="test")

        hunter = TestHunter(sample_config)
        assert await hunter.health_check() is True


class TestExtractLinks:
    def test_basic(self):
        html = '<a href="https://example.com">Click here</a>'
        links = _extract_links(html, "https://base.com")
        assert len(links) == 1
        assert links[0] == ("https://example.com", "Click here")

    def test_relative_url(self):
        html = '<a href="/airdrops/new">New airdrop</a>'
        links = _extract_links(html, "https://airdrops.io/page")
        assert len(links) == 1
        assert links[0][0] == "https://airdrops.io/airdrops/new"

    def test_skips_empty_text(self):
        html = '<a href="https://x.com"></a>'
        links = _extract_links(html, "https://base.com")
        assert len(links) == 0

    def test_skips_short_text(self):
        html = '<a href="https://x.com">ab</a>'
        links = _extract_links(html, "https://base.com")
        assert len(links) == 0

    def test_strips_html_tags(self):
        html = '<a href="https://x.com"><b>Bold text</b></a>'
        links = _extract_links(html, "https://base.com")
        assert links[0][1] == "Bold text"

    def test_skips_non_http(self):
        html = '<a href="javascript:void(0)">Click</a>'
        links = _extract_links(html, "https://base.com")
        assert len(links) == 0


class TestIsCryptoOpportunity:
    def test_matches_airdrop(self):
        assert _is_crypto_opportunity("New airdrop", "https://x.com") is True

    def test_matches_faucet(self):
        assert _is_crypto_opportunity("Free faucet", "https://x.com") is True

    def test_matches_in_url(self):
        assert _is_crypto_opportunity("Click", "https://x.com/earn-tokens") is True

    def test_no_match(self):
        assert _is_crypto_opportunity("Weather forecast", "https://weather.com") is False

    def test_matches_quest(self):
        assert _is_crypto_opportunity("Complete quest for reward", "https://x.com") is True

    def test_matches_giveaway(self):
        assert _is_crypto_opportunity("BTC giveaway", "https://x.com") is True


class TestGuessChain:
    def test_bitcoin(self):
        assert _guess_chain("Free bitcoin faucet") == Chain.BTC

    def test_ethereum(self):
        assert _guess_chain("Ethereum airdrop") == Chain.ETH

    def test_sui(self):
        assert _guess_chain("Sui testnet rewards") == Chain.SUI

    def test_solana(self):
        assert _guess_chain("Solana NFT drop") == Chain.SOL

    def test_base(self):
        assert _guess_chain("Base chain quest") == Chain.BASE

    def test_arbitrum(self):
        assert _guess_chain("Arbitrum airdrop") == Chain.ARB

    def test_polygon(self):
        assert _guess_chain("Polygon staking rewards") == Chain.MATIC

    def test_optimism(self):
        assert _guess_chain("Optimism grant") == Chain.OP

    def test_unknown(self):
        assert _guess_chain("Some random text") == Chain.UNKNOWN

    def test_btc_short(self):
        assert _guess_chain("Free BTC") == Chain.BTC

    def test_sol_short(self):
        assert _guess_chain("Earn SOL") == Chain.SOL

    def test_matic_short(self):
        assert _guess_chain("MATIC rewards") == Chain.MATIC


class TestClassifyOpportunity:
    def test_faucet(self):
        assert _classify_opportunity("New faucet", "claim btc") == OpportunityType.FAUCET

    def test_airdrop(self):
        assert _classify_opportunity("Token airdrop", "free") == OpportunityType.AIRDROP

    def test_giveaway(self):
        assert _classify_opportunity("Crypto giveaway", "win") == OpportunityType.GIVEAWAY

    def test_quest(self):
        assert _classify_opportunity("Galxe quest", "earn") == OpportunityType.QUEST

    def test_bounty(self):
        assert _classify_opportunity("Bug bounty", "find") == OpportunityType.BOUNTY

    def test_grant(self):
        assert _classify_opportunity("Developer grant", "apply") == OpportunityType.GRANT

    def test_testnet(self):
        assert _classify_opportunity("Testnet launch", "deploy") == OpportunityType.TESTNET

    def test_defi(self):
        assert _classify_opportunity("High APY yield", "stake") == OpportunityType.DEFI_YIELD

    def test_default(self):
        assert _classify_opportunity("Something", "else") == OpportunityType.AIRDROP

    def test_layer3(self):
        assert _classify_opportunity("Layer3 campaign", "do") == OpportunityType.QUEST

    def test_zealy(self):
        assert _classify_opportunity("Zealy community", "join") == OpportunityType.QUEST

    def test_token_drop(self):
        assert _classify_opportunity("token drop live", "claim") == OpportunityType.AIRDROP

    def test_staking(self):
        assert _classify_opportunity("staking rewards", "earn") == OpportunityType.DEFI_YIELD


class TestMatchesOpportunity:
    def test_giveaway_with_amount(self):
        assert _matches_opportunity("crypto giveaway $500 today") is True

    def test_free_btc(self):
        assert _matches_opportunity("Get free btc now") is True

    def test_airdrop_live(self):
        assert _matches_opportunity("Airdrop is now live") is True

    def test_claim_free(self):
        assert _matches_opportunity("Claim free token rewards") is True

    def test_no_match(self):
        assert _matches_opportunity("The weather is nice today") is False

    def test_free_eth(self):
        assert _matches_opportunity("free eth for everyone") is True

    def test_airdrop_today(self):
        assert _matches_opportunity("new airdrop today!") is True


class TestClassifyRedditPost:
    def test_giveaway(self):
        assert _classify_reddit_post("BTC giveaway", "") == OpportunityType.GIVEAWAY

    def test_faucet(self):
        assert _classify_reddit_post("New faucet", "") == OpportunityType.FAUCET

    def test_airdrop(self):
        assert _classify_reddit_post("Token airdrop", "") == OpportunityType.AIRDROP

    def test_testnet(self):
        assert _classify_reddit_post("Testnet launch", "") == OpportunityType.TESTNET

    def test_bounty(self):
        assert _classify_reddit_post("Bug bounty", "program") == OpportunityType.BOUNTY

    def test_quest(self):
        assert _classify_reddit_post("New quest", "rewards") == OpportunityType.QUEST

    def test_default(self):
        assert _classify_reddit_post("Something else", "") == OpportunityType.AIRDROP


class TestAggregatorHunter:
    @pytest.mark.asyncio
    async def test_scan_with_mock(self, sample_config):
        hunter = AggregatorHunter(sample_config)
        html = '<html><a href="https://example.com/airdrop">Free token airdrop</a></html>'

        with patch("animus_prospector.hunters.aggregator_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.text = html
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await hunter._scan()

        assert result.hunter_name == "aggregator"
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_scan_handles_errors(self, sample_config):
        hunter = AggregatorHunter(sample_config)

        with patch("animus_prospector.hunters.aggregator_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await hunter._scan()

        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_health_check_success(self, sample_config):
        hunter = AggregatorHunter(sample_config)
        with patch("animus_prospector.hunters.aggregator_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.head = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client
            assert await hunter.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, sample_config):
        hunter = AggregatorHunter(sample_config)
        import httpx

        with patch("animus_prospector.hunters.aggregator_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(side_effect=httpx.ConnectError("down"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client
            assert await hunter.health_check() is False


class TestWebHunter:
    @pytest.mark.asyncio
    async def test_no_endpoint(self, sample_config):
        hunter = WebHunter(sample_config)
        result = await hunter._scan()
        assert len(result.errors) == 1
        assert "No search endpoint" in result.errors[0]

    @pytest.mark.asyncio
    async def test_scan_with_endpoint(self, sample_config):
        sample_config.hunters.sources["web"] = {"search_endpoint": "https://search.api/"}
        hunter = WebHunter(sample_config)

        with patch("animus_prospector.hunters.web_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "results": [
                    {
                        "title": "New crypto faucet 2026",
                        "url": "https://newfaucet.com",
                        "snippet": "Claim free tokens",
                    }
                ]
            }
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await hunter._scan()

        assert result.hunter_name == "web"


class TestSocialHunter:
    @pytest.mark.asyncio
    async def test_scan_reddit(self, sample_config):
        hunter = SocialHunter(sample_config)
        reddit_response = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Free BTC giveaway $100",
                            "selftext": "Claim free tokens",
                            "url": "https://giveaway.com",
                            "permalink": "/r/CryptoAirdrops/123",
                            "score": 50,
                            "num_comments": 10,
                        }
                    },
                    {
                        "data": {
                            "title": "Weather is nice",
                            "selftext": "Sunny day",
                            "url": "/r/weather",
                            "permalink": "/r/CryptoAirdrops/456",
                            "score": 5,
                            "num_comments": 1,
                        }
                    },
                ]
            }
        }

        with patch("animus_prospector.hunters.social_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = reddit_response
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await hunter._scan()

        assert result.hunter_name == "social"
        # Only the giveaway post matches
        matching = [o for o in result.opportunities if "giveaway" in o.title.lower()]
        assert len(matching) >= 1

    @pytest.mark.asyncio
    async def test_reddit_rate_limited(self, sample_config):
        hunter = SocialHunter(sample_config)

        with patch("animus_prospector.hunters.social_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await hunter._scan()

        assert result.hunter_name == "social"

    @pytest.mark.asyncio
    async def test_reddit_post_with_external_url(self, sample_config):
        hunter = SocialHunter(sample_config)
        reddit_response = {
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Airdrop live now $50",
                            "selftext": "",
                            "url": "https://external-airdrop.com/claim",
                            "permalink": "/r/CryptoAirdrops/789",
                            "score": 100,
                            "num_comments": 20,
                        }
                    }
                ]
            }
        }

        with patch("animus_prospector.hunters.social_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = reddit_response
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = await hunter._scan()

        if result.opportunities:
            assert "external-airdrop.com" in result.opportunities[0].url

    @pytest.mark.asyncio
    async def test_health_check(self, sample_config):
        hunter = SocialHunter(sample_config)
        with patch("animus_prospector.hunters.social_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.head = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client
            assert await hunter.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_down(self, sample_config):
        hunter = SocialHunter(sample_config)
        import httpx

        with patch("animus_prospector.hunters.social_hunter.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.head = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client
            assert await hunter.health_check() is False
