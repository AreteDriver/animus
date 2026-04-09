"""Tests for JSONL persistence."""

from __future__ import annotations

from pathlib import Path

from animus_content.models import (
    Article,
    ContentPlatform,
    ContentStatus,
    EarningsRecord,
)
from animus_content.store import ContentStore


class TestContentStore:
    def test_init_creates_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "new_dir"
        ContentStore(d)
        assert d.exists()

    def test_save_and_get_article(self, store: ContentStore, sample_article: Article) -> None:
        store.save_article(sample_article)
        loaded = store.get_article(sample_article.id)
        assert loaded is not None
        assert loaded.title == sample_article.title
        assert loaded.id == sample_article.id

    def test_get_nonexistent_article(self, store: ContentStore) -> None:
        assert store.get_article("nonexistent") is None

    def test_list_articles_empty(self, store: ContentStore) -> None:
        assert store.list_articles() == []

    def test_list_articles(self, store: ContentStore) -> None:
        a1 = Article(id="a1", title="First", platform=ContentPlatform.MIRROR)
        a2 = Article(id="a2", title="Second", platform=ContentPlatform.MEDIUM)
        store.save_article(a1)
        store.save_article(a2)
        articles = store.list_articles()
        assert len(articles) == 2

    def test_list_articles_filter_platform(self, store: ContentStore) -> None:
        a1 = Article(id="a1", platform=ContentPlatform.MIRROR)
        a2 = Article(id="a2", platform=ContentPlatform.MEDIUM)
        store.save_article(a1)
        store.save_article(a2)
        mirror = store.list_articles(platform=ContentPlatform.MIRROR)
        assert len(mirror) == 1
        assert mirror[0].id == "a1"

    def test_list_articles_filter_status(self, store: ContentStore) -> None:
        a1 = Article(id="a1", status=ContentStatus.DRAFT)
        a2 = Article(id="a2", status=ContentStatus.PUBLISHED)
        store.save_article(a1)
        store.save_article(a2)
        published = store.list_articles(status=ContentStatus.PUBLISHED)
        assert len(published) == 1
        assert published[0].id == "a2"

    def test_update_article(self, store: ContentStore, sample_article: Article) -> None:
        store.save_article(sample_article)
        sample_article.title = "Updated Title"
        store.save_article(sample_article)
        loaded = store.get_article(sample_article.id)
        assert loaded is not None
        assert loaded.title == "Updated Title"
        assert store.article_count() == 1

    def test_delete_article(self, store: ContentStore, sample_article: Article) -> None:
        store.save_article(sample_article)
        assert store.delete_article(sample_article.id) is True
        assert store.get_article(sample_article.id) is None

    def test_delete_nonexistent(self, store: ContentStore) -> None:
        assert store.delete_article("nope") is False

    def test_article_count(self, store: ContentStore) -> None:
        assert store.article_count() == 0
        store.save_article(Article(id="a1"))
        assert store.article_count() == 1
        store.save_article(Article(id="a2"))
        assert store.article_count() == 2

    def test_save_and_list_earnings(
        self, store: ContentStore, sample_earnings: EarningsRecord
    ) -> None:
        store.save_earnings(sample_earnings)
        records = store.list_earnings()
        assert len(records) == 1
        assert records[0].article_id == sample_earnings.article_id

    def test_list_earnings_empty(self, store: ContentStore) -> None:
        assert store.list_earnings() == []

    def test_list_earnings_filter_article(self, store: ContentStore) -> None:
        e1 = EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=10.0)
        e2 = EarningsRecord(article_id="a2", platform=ContentPlatform.MIRROR, amount_usd=20.0)
        store.save_earnings(e1)
        store.save_earnings(e2)
        filtered = store.list_earnings(article_id="a1")
        assert len(filtered) == 1
        assert filtered[0].article_id == "a1"

    def test_list_earnings_filter_platform(self, store: ContentStore) -> None:
        e1 = EarningsRecord(article_id="a1", platform=ContentPlatform.MIRROR, amount_usd=10.0)
        e2 = EarningsRecord(article_id="a1", platform=ContentPlatform.MEDIUM, amount_usd=5.0)
        store.save_earnings(e1)
        store.save_earnings(e2)
        filtered = store.list_earnings(platform=ContentPlatform.MEDIUM)
        assert len(filtered) == 1
        assert filtered[0].platform == ContentPlatform.MEDIUM

    def test_malformed_article_line_skipped(self, store: ContentStore) -> None:
        store._articles_path.write_text("not valid json\n")
        articles = store.list_articles()
        assert articles == []

    def test_malformed_earnings_line_skipped(self, store: ContentStore) -> None:
        store._earnings_path.write_text("not valid json\n")
        records = store.list_earnings()
        assert records == []

    def test_multiple_earnings_append(self, store: ContentStore) -> None:
        for i in range(5):
            store.save_earnings(
                EarningsRecord(
                    article_id="a1",
                    platform=ContentPlatform.MIRROR,
                    amount_usd=float(i),
                )
            )
        records = store.list_earnings()
        assert len(records) == 5
