"""Track published articles and earnings."""

from __future__ import annotations

from datetime import datetime

from .models import Article, ContentPlatform, ContentStatus, EarningsRecord
from .platforms.base import BasePlatform
from .store import ContentStore


class EarningsTracker:
    """Tracks and aggregates earnings across platforms."""

    def __init__(
        self,
        store: ContentStore,
        platforms: dict[ContentPlatform, BasePlatform] | None = None,
    ) -> None:
        self.store = store
        self.platforms: dict[ContentPlatform, BasePlatform] = platforms or {}

    async def refresh_earnings(self, article: Article) -> list[EarningsRecord]:
        """Fetch and store latest earnings for an article."""
        adapter = self.platforms.get(article.platform)
        if adapter is None:
            return []
        records = await adapter.fetch_earnings(article.id)
        for record in records:
            self.store.save_earnings(record)
        if records and article.status == ContentStatus.PUBLISHED:
            article.status = ContentStatus.EARNING
            self.store.save_article(article)
        return records

    async def refresh_all(self) -> list[EarningsRecord]:
        """Refresh earnings for all published articles."""
        all_records: list[EarningsRecord] = []
        articles = self.store.list_articles()
        for article in articles:
            if article.status in (ContentStatus.PUBLISHED, ContentStatus.EARNING):
                records = await self.refresh_earnings(article)
                all_records.extend(records)
        return all_records

    def total_earnings_usd(self) -> float:
        """Get total earnings in USD."""
        return sum(r.amount_usd for r in self.store.list_earnings())

    def earnings_by_platform(self) -> dict[ContentPlatform, float]:
        """Get total earnings grouped by platform."""
        result: dict[ContentPlatform, float] = {}
        for record in self.store.list_earnings():
            result[record.platform] = result.get(record.platform, 0.0) + record.amount_usd
        return result

    def earnings_by_article(self) -> dict[str, float]:
        """Get total earnings grouped by article ID."""
        result: dict[str, float] = {}
        for record in self.store.list_earnings():
            result[record.article_id] = result.get(record.article_id, 0.0) + record.amount_usd
        return result

    def earnings_since(self, since: datetime) -> float:
        """Get total earnings in USD since a given datetime."""
        return sum(r.amount_usd for r in self.store.list_earnings() if r.earned_at >= since)

    def top_articles(self, limit: int = 10) -> list[tuple[Article, float]]:
        """Get top-earning articles."""
        by_article = self.earnings_by_article()
        articles = {a.id: a for a in self.store.list_articles()}
        ranked = sorted(by_article.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [(articles[aid], earnings) for aid, earnings in ranked if aid in articles]
