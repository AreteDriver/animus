"""Multi-platform publishing orchestrator."""

from __future__ import annotations

import logging
from datetime import datetime

from .models import Article, ContentPlatform, ContentStatus, PublishResult
from .platforms.base import BasePlatform
from .platforms.medium import MediumPlatform
from .platforms.mirror import MirrorPlatform
from .platforms.paragraph import ParagraphPlatform
from .store import ContentStore

logger = logging.getLogger(__name__)

PLATFORM_CLASSES: dict[ContentPlatform, type[BasePlatform]] = {
    ContentPlatform.MIRROR: MirrorPlatform,
    ContentPlatform.PARAGRAPH: ParagraphPlatform,
    ContentPlatform.MEDIUM: MediumPlatform,
}


class Publisher:
    """Orchestrates publishing articles to multiple platforms."""

    def __init__(
        self,
        platforms: dict[ContentPlatform, BasePlatform] | None = None,
        store: ContentStore | None = None,
    ) -> None:
        self.platforms: dict[ContentPlatform, BasePlatform] = platforms or {}
        self.store = store

    def register_platform(self, platform: BasePlatform) -> None:
        """Register a platform adapter."""
        self.platforms[platform.platform] = platform

    def get_platform(self, platform: ContentPlatform) -> BasePlatform | None:
        """Get a registered platform adapter."""
        return self.platforms.get(platform)

    async def publish(
        self, article: Article, platform: ContentPlatform | None = None
    ) -> PublishResult:
        """Publish an article to its target platform (or override)."""
        target = platform or article.platform
        adapter = self.platforms.get(target)

        if adapter is None:
            return PublishResult(
                article_id=article.id,
                platform=target,
                success=False,
                error=f"Platform {target.value} not registered",
            )

        if not adapter.is_configured():
            return PublishResult(
                article_id=article.id,
                platform=target,
                success=False,
                error=f"Platform {target.value} not configured (missing API key)",
            )

        result = await adapter.publish(article)

        if result.success:
            article.status = ContentStatus.PUBLISHED
            article.published_url = result.published_url
            article.published_at = datetime.now()
            article.platform = target
            if self.store:
                self.store.save_article(article)

        return result

    async def publish_multi(
        self, article: Article, platforms: list[ContentPlatform]
    ) -> list[PublishResult]:
        """Publish an article to multiple platforms."""
        results: list[PublishResult] = []
        for plat in platforms:
            result = await self.publish(article, platform=plat)
            results.append(result)
        return results

    def available_platforms(self) -> list[ContentPlatform]:
        """List platforms that are registered and configured."""
        return [p for p, adapter in self.platforms.items() if adapter.is_configured()]
