"""Tests for the publisher orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_content.models import Article, ContentPlatform, ContentStatus, PublishResult
from animus_content.platforms.base import BasePlatform
from animus_content.platforms.mirror import MirrorPlatform
from animus_content.publisher import Publisher
from animus_content.store import ContentStore


def _make_mock_platform(
    platform: ContentPlatform, configured: bool = True, success: bool = True
) -> BasePlatform:
    """Create a mock platform adapter."""
    mock = MagicMock(spec=BasePlatform)
    mock.platform = platform
    mock.is_configured.return_value = configured
    mock.publish = AsyncMock(
        return_value=PublishResult(
            article_id="test",
            platform=platform,
            success=success,
            published_url=f"https://{platform.value}.test/art" if success else "",
            error="" if success else "Mock error",
        )
    )
    return mock


class TestPublisher:
    def test_register_platform(self) -> None:
        pub = Publisher()
        mirror = MirrorPlatform(api_key="k")
        pub.register_platform(mirror)
        assert ContentPlatform.MIRROR in pub.platforms

    def test_get_platform(self) -> None:
        mirror = MirrorPlatform(api_key="k")
        pub = Publisher(platforms={ContentPlatform.MIRROR: mirror})
        assert pub.get_platform(ContentPlatform.MIRROR) is mirror
        assert pub.get_platform(ContentPlatform.MEDIUM) is None

    def test_available_platforms(self) -> None:
        configured = _make_mock_platform(ContentPlatform.MIRROR, configured=True)
        unconfigured = _make_mock_platform(ContentPlatform.MEDIUM, configured=False)
        pub = Publisher(
            platforms={ContentPlatform.MIRROR: configured, ContentPlatform.MEDIUM: unconfigured}
        )
        avail = pub.available_platforms()
        assert ContentPlatform.MIRROR in avail
        assert ContentPlatform.MEDIUM not in avail

    @pytest.mark.asyncio
    async def test_publish_success(self, sample_article: Article) -> None:
        mock = _make_mock_platform(ContentPlatform.MIRROR)
        pub = Publisher(platforms={ContentPlatform.MIRROR: mock})

        result = await pub.publish(sample_article)
        assert result.success is True
        assert sample_article.status == ContentStatus.PUBLISHED
        assert sample_article.published_url != ""

    @pytest.mark.asyncio
    async def test_publish_platform_not_registered(self, sample_article: Article) -> None:
        pub = Publisher()
        result = await pub.publish(sample_article)
        assert result.success is False
        assert "not registered" in result.error

    @pytest.mark.asyncio
    async def test_publish_platform_not_configured(self, sample_article: Article) -> None:
        mock = _make_mock_platform(ContentPlatform.MIRROR, configured=False)
        pub = Publisher(platforms={ContentPlatform.MIRROR: mock})

        result = await pub.publish(sample_article)
        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_publish_with_override_platform(self, sample_article: Article) -> None:
        mock = _make_mock_platform(ContentPlatform.MEDIUM)
        pub = Publisher(platforms={ContentPlatform.MEDIUM: mock})

        result = await pub.publish(sample_article, platform=ContentPlatform.MEDIUM)
        assert result.success is True
        assert sample_article.platform == ContentPlatform.MEDIUM

    @pytest.mark.asyncio
    async def test_publish_failure(self, sample_article: Article) -> None:
        mock = _make_mock_platform(ContentPlatform.MIRROR, success=False)
        pub = Publisher(platforms={ContentPlatform.MIRROR: mock})

        result = await pub.publish(sample_article)
        assert result.success is False
        assert sample_article.status == ContentStatus.GENERATED  # unchanged

    @pytest.mark.asyncio
    async def test_publish_saves_to_store(
        self, sample_article: Article, store: ContentStore
    ) -> None:
        mock = _make_mock_platform(ContentPlatform.MIRROR)
        pub = Publisher(platforms={ContentPlatform.MIRROR: mock}, store=store)

        await pub.publish(sample_article)
        saved = store.get_article(sample_article.id)
        assert saved is not None
        assert saved.status == ContentStatus.PUBLISHED

    @pytest.mark.asyncio
    async def test_publish_multi(self, sample_article: Article) -> None:
        mock_mirror = _make_mock_platform(ContentPlatform.MIRROR)
        mock_medium = _make_mock_platform(ContentPlatform.MEDIUM)
        pub = Publisher(
            platforms={
                ContentPlatform.MIRROR: mock_mirror,
                ContentPlatform.MEDIUM: mock_medium,
            }
        )

        results = await pub.publish_multi(
            sample_article, [ContentPlatform.MIRROR, ContentPlatform.MEDIUM]
        )
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_publish_multi_partial_failure(self, sample_article: Article) -> None:
        mock_ok = _make_mock_platform(ContentPlatform.MIRROR, success=True)
        mock_fail = _make_mock_platform(ContentPlatform.MEDIUM, success=False)
        pub = Publisher(
            platforms={
                ContentPlatform.MIRROR: mock_ok,
                ContentPlatform.MEDIUM: mock_fail,
            }
        )

        results = await pub.publish_multi(
            sample_article, [ContentPlatform.MIRROR, ContentPlatform.MEDIUM]
        )
        assert results[0].success is True
        assert results[1].success is False
