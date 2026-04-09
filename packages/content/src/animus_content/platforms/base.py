"""Abstract base class for platform adapters."""

from __future__ import annotations

import abc

from animus_content.models import Article, ContentPlatform, EarningsRecord, PublishResult


class BasePlatform(abc.ABC):
    """Base class for all platform adapters."""

    platform: ContentPlatform

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs

    @abc.abstractmethod
    async def publish(self, article: Article) -> PublishResult:
        """Publish an article to the platform."""

    @abc.abstractmethod
    async def fetch_earnings(self, article_id: str) -> list[EarningsRecord]:
        """Fetch earnings for a published article."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if the platform API is reachable."""

    def is_configured(self) -> bool:
        """Check if required credentials are present."""
        return bool(self.api_key)
