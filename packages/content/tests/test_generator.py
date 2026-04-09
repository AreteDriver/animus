"""Tests for content generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_content.config import ContentConfig, LLMConfig
from animus_content.generator import ContentGenerator
from animus_content.models import ContentPlatform, ContentStatus, ContentTopic


@pytest.fixture
def generator(config: ContentConfig) -> ContentGenerator:
    return ContentGenerator(config)


class TestContentGenerator:
    @pytest.mark.asyncio
    async def test_generate_with_ollama(
        self, generator: ContentGenerator, sample_topic: ContentTopic
    ) -> None:
        mock_response = {
            "response": "Great DeFi Article\n\nThis is the body about DeFi yield strategies."
        }
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            article = await generator.generate(sample_topic)

            assert article.title == "Great DeFi Article"
            assert "body about DeFi" in article.body
            assert article.status == ContentStatus.GENERATED
            assert article.platform == ContentPlatform.MIRROR
            assert article.word_count > 0

    @pytest.mark.asyncio
    async def test_generate_with_anthropic(
        self, config: ContentConfig, sample_topic: ContentTopic
    ) -> None:
        config.llm = LLMConfig(provider="anthropic", model="claude-3-haiku", api_key="sk-test")
        gen = ContentGenerator(config)

        mock_response = {"content": [{"text": "Anthropic Article\n\nBody from Anthropic."}]}
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            article = await gen.generate(sample_topic)
            assert article.title == "Anthropic Article"

    @pytest.mark.asyncio
    async def test_generate_llm_failure_returns_untitled(
        self, generator: ContentGenerator, sample_topic: ContentTopic
    ) -> None:
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

            article = await generator.generate(sample_topic)
            assert article.title == "Untitled"
            assert article.body == ""

    @pytest.mark.asyncio
    async def test_generate_with_extra_tags(
        self, generator: ContentGenerator, sample_topic: ContentTopic
    ) -> None:
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "Title\n\nBody text here."}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            article = await generator.generate(sample_topic, extra_tags=["nft", "gaming"])
            assert "nft" in article.tags or "gaming" in article.tags

    @pytest.mark.asyncio
    async def test_generate_with_word_count(
        self, generator: ContentGenerator, sample_topic: ContentTopic
    ) -> None:
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "Title\n\nBody text here."}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            article = await generator.generate(sample_topic, word_count=1500)
            assert article is not None

    @pytest.mark.asyncio
    async def test_suggest_topics(self, generator: ContentGenerator) -> None:
        raw = (
            "TOPIC: Bitcoin ETF Impact | KEYWORDS: btc, etf, institutional | ENGAGEMENT: 0.8\n"
            "TOPIC: Layer 2 Scaling | KEYWORDS: l2, rollups, zk | ENGAGEMENT: 0.7\n"
        )
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": raw}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            topics = await generator.suggest_topics(count=2)
            assert len(topics) == 2
            assert topics[0].topic == "Bitcoin ETF Impact"
            assert topics[0].estimated_engagement == 0.8
            assert "btc" in topics[0].keywords

    @pytest.mark.asyncio
    async def test_suggest_topics_empty_response(self, generator: ContentGenerator) -> None:
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": ""}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            topics = await generator.suggest_topics()
            assert topics == []

    @pytest.mark.asyncio
    async def test_suggest_topics_malformed_lines(self, generator: ContentGenerator) -> None:
        raw = "Some random text\nTOPIC: Valid | KEYWORDS: a,b | ENGAGEMENT: not_a_float\n"
        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": raw}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            topics = await generator.suggest_topics()
            # malformed engagement should be skipped
            assert len(topics) == 0

    @pytest.mark.asyncio
    async def test_anthropic_empty_content_blocks(
        self, config: ContentConfig, sample_topic: ContentTopic
    ) -> None:
        config.llm = LLMConfig(provider="anthropic", model="test", api_key="sk-test")
        gen = ContentGenerator(config)

        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"content": []}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            article = await gen.generate(sample_topic)
            assert article.title == "Untitled"

    @pytest.mark.asyncio
    async def test_anthropic_http_error(
        self, config: ContentConfig, sample_topic: ContentTopic
    ) -> None:
        config.llm = LLMConfig(provider="anthropic", model="test", api_key="sk-test")
        gen = ContentGenerator(config)

        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            import httpx

            mock_client.post = AsyncMock(side_effect=httpx.HTTPError("API error"))

            article = await gen.generate(sample_topic)
            assert article.title == "Untitled"

    @pytest.mark.asyncio
    async def test_unknown_provider_falls_back_to_ollama(
        self, config: ContentConfig, sample_topic: ContentTopic
    ) -> None:
        config.llm = LLMConfig(provider="unknown_provider", model="test")
        gen = ContentGenerator(config)

        with patch("animus_content.generator.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "Fallback Title\n\nFallback body."}
            mock_resp.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_resp)

            article = await gen.generate(sample_topic)
            assert article.title == "Fallback Title"


class TestParseResponse:
    def test_normal_response(self) -> None:
        title, body = ContentGenerator._parse_response("My Title\n\nMy body text")
        assert title == "My Title"
        assert body == "My body text"

    def test_response_with_hash_prefix(self) -> None:
        title, body = ContentGenerator._parse_response("# My Title\n\nBody")
        assert title == "My Title"

    def test_empty_response(self) -> None:
        title, body = ContentGenerator._parse_response("")
        assert title == "Untitled"
        assert body == ""

    def test_whitespace_only(self) -> None:
        title, body = ContentGenerator._parse_response("   \n  \n  ")
        assert title == "Untitled"

    def test_title_only_no_body(self) -> None:
        title, body = ContentGenerator._parse_response("Just a Title")
        assert title == "Just a Title"
        assert body == ""

    def test_multiple_blank_lines(self) -> None:
        title, body = ContentGenerator._parse_response("Title\n\n\n\nBody starts here")
        assert title == "Title"
        assert body == "Body starts here"


class TestParseTopics:
    def test_valid_topics(self) -> None:
        raw = "TOPIC: Bitcoin | KEYWORDS: btc, crypto | ENGAGEMENT: 0.9"
        topics = ContentGenerator._parse_topics(raw, ContentPlatform.MIRROR)
        assert len(topics) == 1
        assert topics[0].topic == "Bitcoin"
        assert topics[0].estimated_engagement == 0.9

    def test_empty_input(self) -> None:
        topics = ContentGenerator._parse_topics("", ContentPlatform.MIRROR)
        assert topics == []

    def test_no_topic_marker(self) -> None:
        topics = ContentGenerator._parse_topics("random text\nmore random", ContentPlatform.MIRROR)
        assert topics == []

    def test_multiple_topics(self) -> None:
        raw = (
            "TOPIC: A | KEYWORDS: x | ENGAGEMENT: 0.5\nTOPIC: B | KEYWORDS: y,z | ENGAGEMENT: 0.8\n"
        )
        topics = ContentGenerator._parse_topics(raw, ContentPlatform.PARAGRAPH)
        assert len(topics) == 2
        assert topics[1].target_platform == ContentPlatform.PARAGRAPH
