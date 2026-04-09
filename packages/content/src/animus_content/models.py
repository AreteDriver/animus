"""Data models for the content module."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ContentPlatform(StrEnum):
    """Supported write-to-earn platforms."""

    MIRROR = "mirror"
    PARAGRAPH = "paragraph"
    MEDIUM = "medium"
    HASHNODE = "hashnode"
    DEVTO = "devto"
    SUBSTACK = "substack"


class ContentStatus(StrEnum):
    """Lifecycle status of a content piece."""

    DRAFT = "draft"
    GENERATED = "generated"
    PUBLISHED = "published"
    EARNING = "earning"
    ARCHIVED = "archived"


class ContentTopic(BaseModel):
    """A topic for content generation."""

    topic: str
    keywords: list[str] = Field(default_factory=list)
    target_platform: ContentPlatform = ContentPlatform.MIRROR
    estimated_engagement: float = 0.0
    last_published: datetime | None = None

    def is_stale(self, cooldown_days: int = 7) -> bool:
        """Check if topic has not been published recently."""
        if self.last_published is None:
            return True
        delta = datetime.now() - self.last_published
        return delta.days >= cooldown_days


class Article(BaseModel):
    """A generated or published article."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    body: str = ""
    platform: ContentPlatform = ContentPlatform.MIRROR
    topic: str = ""
    status: ContentStatus = ContentStatus.DRAFT
    published_url: str = ""
    published_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    word_count: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_word_count(self) -> int:
        """Recompute word count from body."""
        self.word_count = len(self.body.split())
        return self.word_count


class PublishResult(BaseModel):
    """Result of a publish attempt."""

    article_id: str
    platform: ContentPlatform
    success: bool
    published_url: str = ""
    error: str = ""


class EarningsRecord(BaseModel):
    """A single earnings event for an article."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    article_id: str
    platform: ContentPlatform
    amount: float = 0.0
    token: str = "USD"
    amount_usd: float = 0.0
    earned_at: datetime = Field(default_factory=datetime.now)
