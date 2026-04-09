"""Tests for data models."""

from __future__ import annotations

from datetime import datetime, timedelta

from animus_content.models import (
    Article,
    ContentPlatform,
    ContentStatus,
    ContentTopic,
    EarningsRecord,
    PublishResult,
)


class TestContentPlatform:
    def test_values(self) -> None:
        assert ContentPlatform.MIRROR == "mirror"
        assert ContentPlatform.PARAGRAPH == "paragraph"
        assert ContentPlatform.MEDIUM == "medium"
        assert ContentPlatform.HASHNODE == "hashnode"
        assert ContentPlatform.DEVTO == "devto"
        assert ContentPlatform.SUBSTACK == "substack"

    def test_from_string(self) -> None:
        assert ContentPlatform("mirror") == ContentPlatform.MIRROR

    def test_all_values(self) -> None:
        assert len(ContentPlatform) == 6


class TestContentStatus:
    def test_values(self) -> None:
        assert ContentStatus.DRAFT == "draft"
        assert ContentStatus.GENERATED == "generated"
        assert ContentStatus.PUBLISHED == "published"
        assert ContentStatus.EARNING == "earning"
        assert ContentStatus.ARCHIVED == "archived"

    def test_all_values(self) -> None:
        assert len(ContentStatus) == 5


class TestContentTopic:
    def test_defaults(self) -> None:
        t = ContentTopic(topic="test")
        assert t.keywords == []
        assert t.target_platform == ContentPlatform.MIRROR
        assert t.estimated_engagement == 0.0
        assert t.last_published is None

    def test_is_stale_never_published(self) -> None:
        t = ContentTopic(topic="test")
        assert t.is_stale() is True

    def test_is_stale_recently_published(self) -> None:
        t = ContentTopic(topic="test", last_published=datetime.now())
        assert t.is_stale(cooldown_days=7) is False

    def test_is_stale_old_published(self) -> None:
        old = datetime.now() - timedelta(days=10)
        t = ContentTopic(topic="test", last_published=old)
        assert t.is_stale(cooldown_days=7) is True

    def test_is_stale_exact_boundary(self) -> None:
        exact = datetime.now() - timedelta(days=7)
        t = ContentTopic(topic="test", last_published=exact)
        assert t.is_stale(cooldown_days=7) is True

    def test_custom_cooldown(self) -> None:
        recent = datetime.now() - timedelta(days=2)
        t = ContentTopic(topic="test", last_published=recent)
        assert t.is_stale(cooldown_days=1) is True
        assert t.is_stale(cooldown_days=3) is False


class TestArticle:
    def test_defaults(self) -> None:
        a = Article()
        assert a.title == ""
        assert a.body == ""
        assert a.status == ContentStatus.DRAFT
        assert a.platform == ContentPlatform.MIRROR
        assert a.word_count == 0
        assert a.tags == []
        assert a.metadata == {}
        assert a.published_url == ""
        assert a.published_at is None
        assert len(a.id) == 12

    def test_compute_word_count(self) -> None:
        a = Article(body="one two three four five")
        result = a.compute_word_count()
        assert result == 5
        assert a.word_count == 5

    def test_compute_word_count_empty(self) -> None:
        a = Article(body="")
        result = a.compute_word_count()
        assert result == 0

    def test_compute_word_count_single_word(self) -> None:
        a = Article(body="blockchain")
        assert a.compute_word_count() == 1

    def test_serialization(self) -> None:
        a = Article(title="Test", body="Body text")
        data = a.model_dump()
        assert data["title"] == "Test"
        restored = Article.model_validate(data)
        assert restored.title == "Test"

    def test_json_round_trip(self) -> None:
        a = Article(title="Test", body="Body")
        json_str = a.model_dump_json()
        restored = Article.model_validate_json(json_str)
        assert restored.title == a.title
        assert restored.id == a.id

    def test_unique_ids(self) -> None:
        ids = {Article().id for _ in range(100)}
        assert len(ids) == 100


class TestPublishResult:
    def test_success(self) -> None:
        r = PublishResult(
            article_id="abc",
            platform=ContentPlatform.MIRROR,
            success=True,
            published_url="https://mirror.xyz/test",
        )
        assert r.success is True
        assert r.error == ""

    def test_failure(self) -> None:
        r = PublishResult(
            article_id="abc",
            platform=ContentPlatform.MEDIUM,
            success=False,
            error="API error",
        )
        assert r.success is False
        assert r.error == "API error"


class TestEarningsRecord:
    def test_defaults(self) -> None:
        e = EarningsRecord(article_id="abc", platform=ContentPlatform.MIRROR)
        assert e.amount == 0.0
        assert e.token == "USD"
        assert e.amount_usd == 0.0
        assert len(e.id) == 12

    def test_with_values(self) -> None:
        e = EarningsRecord(
            article_id="abc",
            platform=ContentPlatform.PARAGRAPH,
            amount=1.5,
            token="MATIC",
            amount_usd=0.75,
        )
        assert e.amount == 1.5
        assert e.token == "MATIC"
