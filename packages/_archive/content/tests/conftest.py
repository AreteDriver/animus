"""Shared fixtures for content module tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from animus_content.config import ContentConfig, LLMConfig
from animus_content.models import (
    Article,
    ContentPlatform,
    ContentStatus,
    ContentTopic,
    EarningsRecord,
)
from animus_content.store import ContentStore


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary data directory."""
    d = tmp_path / "content_data"
    d.mkdir()
    return d


@pytest.fixture
def config(tmp_data_dir: Path) -> ContentConfig:
    """Provide a test config."""
    return ContentConfig(
        data_dir=tmp_data_dir,
        llm=LLMConfig(provider="ollama", model="test-model", base_url="http://localhost:11434"),
        default_tags=["crypto", "web3"],
        min_word_count=500,
        max_word_count=2500,
    )


@pytest.fixture
def store(tmp_data_dir: Path) -> ContentStore:
    """Provide a fresh content store."""
    return ContentStore(tmp_data_dir)


@pytest.fixture
def sample_article() -> Article:
    """A sample article for testing."""
    return Article(
        id="test123",
        title="Test Article Title",
        body="This is the body of the test article with enough words to count.",
        platform=ContentPlatform.MIRROR,
        topic="test topic",
        status=ContentStatus.GENERATED,
        tags=["crypto", "test"],
        word_count=13,
    )


@pytest.fixture
def sample_topic() -> ContentTopic:
    """A sample topic for testing."""
    return ContentTopic(
        topic="DeFi yield strategies",
        keywords=["defi", "yield", "farming"],
        target_platform=ContentPlatform.MIRROR,
        estimated_engagement=0.7,
    )


@pytest.fixture
def sample_earnings() -> EarningsRecord:
    """A sample earnings record."""
    return EarningsRecord(
        id="earn123",
        article_id="test123",
        platform=ContentPlatform.MIRROR,
        amount=0.05,
        token="ETH",
        amount_usd=150.0,
        earned_at=datetime(2026, 4, 1, 12, 0),
    )


@pytest.fixture
def published_article() -> Article:
    """An article in published state."""
    return Article(
        id="pub456",
        title="Published Article",
        body="Published body content with several words.",
        platform=ContentPlatform.MIRROR,
        topic="published topic",
        status=ContentStatus.PUBLISHED,
        published_url="https://mirror.xyz/test/pub456",
        published_at=datetime(2026, 4, 1),
        tags=["crypto"],
        word_count=7,
    )
