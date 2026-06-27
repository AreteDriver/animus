"""Tests for earnings tracker."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from animus_content.models import (
    Article,
    ContentPlatform,
    ContentStatus,
    EarningsRecord,
)
from animus_content.platforms.base import BasePlatform
from animus_content.store import ContentStore
from animus_content.tracker import EarningsTracker


class TestEarningsTracker:
    @pytest.fixture
    def tracker(self, store: ContentStore) -> EarningsTracker:
        return EarningsTracker(store=store)

    def test_total_earnings_empty(self, tracker: EarningsTracker) -> None:
        assert tracker.total_earnings_usd() == 0.0

    def test_total_earnings(self, tracker: EarningsTracker, store: ContentStore) -> None:
        store.save_earnings(
            EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=100.0)
        )
        store.save_earnings(
            EarningsRecord(article_id="a2", platform=ContentPlatform.MEDIUM, amount_usd=50.0)
        )
        assert tracker.total_earnings_usd() == 150.0

    def test_earnings_by_platform(self, tracker: EarningsTracker, store: ContentStore) -> None:
        store.save_earnings(
            EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=100.0)
        )
        store.save_earnings(
            EarningsRecord(article_id="a2", platform=ContentPlatform.MIRROR, amount_usd=50.0)
        )
        store.save_earnings(
            EarningsRecord(article_id="a3", platform=ContentPlatform.MEDIUM, amount_usd=25.0)
        )
        by_plat = tracker.earnings_by_platform()
        assert by_plat[ContentPlatform.MIRROR] == 150.0
        assert by_plat[ContentPlatform.MEDIUM] == 25.0

    def test_earnings_by_article(self, tracker: EarningsTracker, store: ContentStore) -> None:
        store.save_earnings(
            EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=10.0)
        )
        store.save_earnings(
            EarningsRecord(article_id="a1", platform=ContentPlatform.MEDIUM, amount_usd=5.0)
        )
        store.save_earnings(
            EarningsRecord(article_id="a2", platform=ContentPlatform.MIRROR, amount_usd=20.0)
        )
        by_art = tracker.earnings_by_article()
        assert by_art["a1"] == 15.0
        assert by_art["a2"] == 20.0

    def test_earnings_since(self, tracker: EarningsTracker, store: ContentStore) -> None:
        old = datetime.now() - timedelta(days=30)
        recent = datetime.now() - timedelta(hours=1)
        store.save_earnings(
            EarningsRecord(
                article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=100.0, earned_at=old
            )
        )
        store.save_earnings(
            EarningsRecord(
                article_id="a2", platform=ContentPlatform.MIRROR, amount_usd=50.0, earned_at=recent
            )
        )
        cutoff = datetime.now() - timedelta(days=1)
        assert tracker.earnings_since(cutoff) == 50.0

    def test_top_articles(self, tracker: EarningsTracker, store: ContentStore) -> None:
        a1 = Article(id="a1", title="Top Article", platform=ContentPlatform.MIRROR)
        a2 = Article(id="a2", title="Second", platform=ContentPlatform.MIRROR)
        store.save_article(a1)
        store.save_article(a2)
        store.save_earnings(
            EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=200.0)
        )
        store.save_earnings(
            EarningsRecord(article_id="a2", platform=ContentPlatform.MIRROR, amount_usd=50.0)
        )

        top = tracker.top_articles(limit=2)
        assert len(top) == 2
        assert top[0][0].id == "a1"
        assert top[0][1] == 200.0

    def test_top_articles_limit(self, tracker: EarningsTracker, store: ContentStore) -> None:
        for i in range(5):
            store.save_article(Article(id=f"a{i}", platform=ContentPlatform.MIRROR))
            store.save_earnings(
                EarningsRecord(
                    article_id=f"a{i}",
                    platform=ContentPlatform.MIRROR,
                    amount_usd=float(i * 10),
                )
            )
        top = tracker.top_articles(limit=3)
        assert len(top) == 3

    def test_top_articles_missing_article(
        self, tracker: EarningsTracker, store: ContentStore
    ) -> None:
        # Earnings exist but article deleted
        store.save_earnings(
            EarningsRecord(article_id="missing", platform=ContentPlatform.MIRROR, amount_usd=100.0)
        )
        top = tracker.top_articles()
        assert len(top) == 0

    @pytest.mark.asyncio
    async def test_refresh_earnings(self, store: ContentStore) -> None:
        mock_platform = MagicMock(spec=BasePlatform)
        mock_platform.fetch_earnings = AsyncMock(
            return_value=[
                EarningsRecord(article_id="pub1", platform=ContentPlatform.MIRROR, amount_usd=50.0)
            ]
        )

        article = Article(
            id="pub1", platform=ContentPlatform.MIRROR, status=ContentStatus.PUBLISHED
        )
        store.save_article(article)

        tracker = EarningsTracker(store=store, platforms={ContentPlatform.MIRROR: mock_platform})
        records = await tracker.refresh_earnings(article)
        assert len(records) == 1
        assert store.list_earnings()[0].amount_usd == 50.0
        # Article should be updated to EARNING status
        updated = store.get_article("pub1")
        assert updated is not None
        assert updated.status == ContentStatus.EARNING

    @pytest.mark.asyncio
    async def test_refresh_earnings_no_adapter(self, store: ContentStore) -> None:
        tracker = EarningsTracker(store=store)
        article = Article(id="a1", platform=ContentPlatform.MIRROR)
        records = await tracker.refresh_earnings(article)
        assert records == []

    @pytest.mark.asyncio
    async def test_refresh_all(self, store: ContentStore) -> None:
        mock_platform = MagicMock(spec=BasePlatform)
        mock_platform.fetch_earnings = AsyncMock(
            return_value=[
                EarningsRecord(article_id="pub1", platform=ContentPlatform.MIRROR, amount_usd=10.0)
            ]
        )

        a1 = Article(id="pub1", platform=ContentPlatform.MIRROR, status=ContentStatus.PUBLISHED)
        a2 = Article(id="draft1", platform=ContentPlatform.MIRROR, status=ContentStatus.DRAFT)
        store.save_article(a1)
        store.save_article(a2)

        tracker = EarningsTracker(store=store, platforms={ContentPlatform.MIRROR: mock_platform})
        records = await tracker.refresh_all()
        assert len(records) == 1  # only published articles refreshed

    @pytest.mark.asyncio
    async def test_refresh_earnings_already_earning(self, store: ContentStore) -> None:
        """Status stays EARNING if already EARNING."""
        mock_platform = MagicMock(spec=BasePlatform)
        mock_platform.fetch_earnings = AsyncMock(
            return_value=[
                EarningsRecord(article_id="e1", platform=ContentPlatform.MIRROR, amount_usd=5.0)
            ]
        )

        article = Article(id="e1", platform=ContentPlatform.MIRROR, status=ContentStatus.EARNING)
        store.save_article(article)

        tracker = EarningsTracker(store=store, platforms={ContentPlatform.MIRROR: mock_platform})
        records = await tracker.refresh_earnings(article)
        assert len(records) == 1
        # Status unchanged (still EARNING)
        updated = store.get_article("e1")
        assert updated is not None
        assert updated.status == ContentStatus.EARNING

    @pytest.mark.asyncio
    async def test_refresh_earnings_empty_result(self, store: ContentStore) -> None:
        """No earnings returned means no status change."""
        mock_platform = MagicMock(spec=BasePlatform)
        mock_platform.fetch_earnings = AsyncMock(return_value=[])

        article = Article(id="n1", platform=ContentPlatform.MIRROR, status=ContentStatus.PUBLISHED)
        store.save_article(article)

        tracker = EarningsTracker(store=store, platforms={ContentPlatform.MIRROR: mock_platform})
        records = await tracker.refresh_earnings(article)
        assert records == []
        updated = store.get_article("n1")
        assert updated is not None
        assert updated.status == ContentStatus.PUBLISHED
