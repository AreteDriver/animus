"""Tests for MCP tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_content.config import ContentConfig
from animus_content.models import Article, ContentPlatform, ContentStatus, ContentTopic
from animus_content.tools.content_tools import (
    CONTENT_TOOLS,
    content_earnings,
    content_generate,
    content_publish,
    content_topics,
)


@pytest.fixture(autouse=True)
def mock_config(tmp_path: Path) -> ContentConfig:
    """Override config for all tool tests."""
    config = ContentConfig(data_dir=tmp_path / "content")
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with patch("animus_content.tools.content_tools._get_config", return_value=config):
        yield config


class TestContentTools:
    def test_tool_definitions(self) -> None:
        assert len(CONTENT_TOOLS) == 4
        names = {t["name"] for t in CONTENT_TOOLS}
        assert names == {
            "content_generate",
            "content_publish",
            "content_earnings",
            "content_topics",
        }

    def test_all_tools_have_category(self) -> None:
        for tool in CONTENT_TOOLS:
            assert tool["category"] == "content"

    def test_all_tools_have_handler(self) -> None:
        for tool in CONTENT_TOOLS:
            assert callable(tool["handler"])


class TestContentGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self, mock_config: ContentConfig) -> None:
        mock_article = Article(
            id="gen1",
            title="Generated Title",
            body="Body text here",
            platform=ContentPlatform.MIRROR,
            status=ContentStatus.GENERATED,
            word_count=3,
        )
        with patch("animus_content.tools.content_tools.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_article)
            mock_gen_cls.return_value = mock_gen

            result = await content_generate(topic="DeFi", platform="mirror")
            assert result["id"] == "gen1"
            assert result["title"] == "Generated Title"
            assert result["status"] == "generated"

    @pytest.mark.asyncio
    async def test_generate_with_keywords(self, mock_config: ContentConfig) -> None:
        mock_article = Article(
            id="gen2", title="KW Article", word_count=10, status=ContentStatus.GENERATED
        )
        with patch("animus_content.tools.content_tools.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate = AsyncMock(return_value=mock_article)
            mock_gen_cls.return_value = mock_gen

            result = await content_generate(topic="NFTs", keywords=["nft", "art"])
            assert result["id"] == "gen2"


class TestContentPublish:
    @pytest.mark.asyncio
    async def test_publish_article_not_found(self, mock_config: ContentConfig) -> None:
        result = await content_publish(article_id="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_publish_success(self, mock_config: ContentConfig) -> None:
        from animus_content.store import ContentStore

        store = ContentStore(mock_config.data_dir)
        article = Article(
            id="pub1", title="Test", status=ContentStatus.GENERATED, platform=ContentPlatform.MIRROR
        )
        store.save_article(article)

        with patch("animus_content.tools.content_tools.Publisher") as mock_pub_cls:
            mock_pub = MagicMock()
            from animus_content.models import PublishResult

            mock_pub.publish = AsyncMock(
                return_value=PublishResult(
                    article_id="pub1",
                    platform=ContentPlatform.MIRROR,
                    success=True,
                    published_url="https://mirror.xyz/test",
                )
            )
            mock_pub_cls.return_value = mock_pub

            result = await content_publish(article_id="pub1")
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_publish_with_platform_override(self, mock_config: ContentConfig) -> None:
        from animus_content.store import ContentStore

        store = ContentStore(mock_config.data_dir)
        article = Article(id="pub2", title="Test", status=ContentStatus.GENERATED)
        store.save_article(article)

        with patch("animus_content.tools.content_tools.Publisher") as mock_pub_cls:
            mock_pub = MagicMock()
            from animus_content.models import PublishResult

            mock_pub.publish = AsyncMock(
                return_value=PublishResult(
                    article_id="pub2",
                    platform=ContentPlatform.MEDIUM,
                    success=True,
                    published_url="https://medium.com/test",
                )
            )
            mock_pub_cls.return_value = mock_pub

            result = await content_publish(article_id="pub2", platform="medium")
            assert result["platform"] == "medium"


class TestContentEarnings:
    @pytest.mark.asyncio
    async def test_earnings_empty(self, mock_config: ContentConfig) -> None:
        result = await content_earnings()
        assert result["total_usd"] == 0.0
        assert result["by_platform"] == {}
        assert result["article_count"] == 0

    @pytest.mark.asyncio
    async def test_earnings_with_data(self, mock_config: ContentConfig) -> None:
        from animus_content.models import EarningsRecord
        from animus_content.store import ContentStore

        store = ContentStore(mock_config.data_dir)
        store.save_article(Article(id="a1", status=ContentStatus.PUBLISHED))
        store.save_article(Article(id="a2", status=ContentStatus.EARNING))
        store.save_earnings(
            EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=100.0)
        )

        result = await content_earnings()
        assert result["total_usd"] == 100.0
        assert result["article_count"] == 2
        assert result["published_count"] == 2


class TestContentTopics:
    @pytest.mark.asyncio
    async def test_topics_success(self, mock_config: ContentConfig) -> None:
        mock_topics = [
            ContentTopic(topic="Bitcoin ETFs", keywords=["btc"], estimated_engagement=0.9),
        ]
        with patch("animus_content.tools.content_tools.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.suggest_topics = AsyncMock(return_value=mock_topics)
            mock_gen_cls.return_value = mock_gen

            result = await content_topics(platform="mirror", count=3)
            assert len(result["topics"]) == 1
            assert result["topics"][0]["topic"] == "Bitcoin ETFs"

    @pytest.mark.asyncio
    async def test_topics_empty(self, mock_config: ContentConfig) -> None:
        with patch("animus_content.tools.content_tools.ContentGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.suggest_topics = AsyncMock(return_value=[])
            mock_gen_cls.return_value = mock_gen

            result = await content_topics()
            assert result["topics"] == []
