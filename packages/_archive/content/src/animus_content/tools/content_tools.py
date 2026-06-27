"""Animus MCP tool definitions for content operations."""

from __future__ import annotations

from typing import Any

from animus_content.config import ContentConfig
from animus_content.generator import ContentGenerator
from animus_content.models import ContentPlatform, ContentStatus, ContentTopic
from animus_content.publisher import Publisher
from animus_content.store import ContentStore
from animus_content.tracker import EarningsTracker


def _get_config() -> ContentConfig:
    """Load or create default config."""
    config_path = ContentConfig().data_dir / "config.yaml"
    return ContentConfig.load(config_path)


def _get_store(config: ContentConfig) -> ContentStore:
    return ContentStore(config.data_dir)


async def content_generate(topic: str, platform: str = "mirror", **kwargs: Any) -> dict[str, Any]:
    """Generate an article on a given topic.

    Args:
        topic: The topic to write about.
        platform: Target platform (mirror, paragraph, medium, hashnode, devto, substack).

    Returns:
        Dict with article id, title, word_count, status.
    """
    config = _get_config()
    store = _get_store(config)
    generator = ContentGenerator(config)

    content_topic = ContentTopic(
        topic=topic,
        keywords=kwargs.get("keywords", []),
        target_platform=ContentPlatform(platform),
    )

    article = await generator.generate(content_topic)
    store.save_article(article)

    return {
        "id": article.id,
        "title": article.title,
        "word_count": article.word_count,
        "status": article.status.value,
        "platform": article.platform.value,
    }


async def content_publish(
    article_id: str, platform: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Publish a generated article to a platform.

    Args:
        article_id: The article ID to publish.
        platform: Override target platform (optional).

    Returns:
        Dict with success, published_url, error.
    """
    config = _get_config()
    store = _get_store(config)
    publisher = Publisher(store=store)

    article = store.get_article(article_id)
    if article is None:
        return {"success": False, "error": f"Article {article_id} not found"}

    target = ContentPlatform(platform) if platform else None
    result = await publisher.publish(article, platform=target)

    return {
        "success": result.success,
        "published_url": result.published_url,
        "error": result.error,
        "platform": result.platform.value,
    }


async def content_earnings(**kwargs: Any) -> dict[str, Any]:
    """Get an earnings report across all platforms.

    Returns:
        Dict with total_usd, by_platform, article_count.
    """
    config = _get_config()
    store = _get_store(config)
    tracker = EarningsTracker(store=store)

    return {
        "total_usd": tracker.total_earnings_usd(),
        "by_platform": {k.value: v for k, v in tracker.earnings_by_platform().items()},
        "article_count": store.article_count(),
        "published_count": len(store.list_articles(status=ContentStatus.PUBLISHED))
        + len(store.list_articles(status=ContentStatus.EARNING)),
    }


async def content_topics(platform: str = "mirror", count: int = 5, **kwargs: Any) -> dict[str, Any]:
    """List or suggest content topics.

    Args:
        platform: Target platform for topic suggestions.
        count: Number of topics to suggest.

    Returns:
        Dict with suggested topics list.
    """
    config = _get_config()
    generator = ContentGenerator(config)

    topics = await generator.suggest_topics(
        platform=ContentPlatform(platform),
        count=count,
    )

    return {
        "topics": [
            {
                "topic": t.topic,
                "keywords": t.keywords,
                "platform": t.target_platform.value,
                "estimated_engagement": t.estimated_engagement,
            }
            for t in topics
        ]
    }


CONTENT_TOOLS = [
    {
        "name": "content_generate",
        "category": "content",
        "description": "Generate an article on a topic for a write-to-earn platform",
        "handler": content_generate,
        "parameters": {
            "topic": {"type": "string", "required": True},
            "platform": {"type": "string", "required": False, "default": "mirror"},
            "keywords": {"type": "array", "required": False},
        },
    },
    {
        "name": "content_publish",
        "category": "content",
        "description": "Publish a generated article to a platform",
        "handler": content_publish,
        "parameters": {
            "article_id": {"type": "string", "required": True},
            "platform": {"type": "string", "required": False},
        },
    },
    {
        "name": "content_earnings",
        "category": "content",
        "description": "Get earnings report across all platforms",
        "handler": content_earnings,
        "parameters": {},
    },
    {
        "name": "content_topics",
        "category": "content",
        "description": "Suggest trending content topics for a platform",
        "handler": content_topics,
        "parameters": {
            "platform": {"type": "string", "required": False, "default": "mirror"},
            "count": {"type": "integer", "required": False, "default": 5},
        },
    },
]
