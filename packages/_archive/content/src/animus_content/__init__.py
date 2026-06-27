"""Animus Content — autonomous write-to-earn content module."""

__version__ = "0.1.0"

from .config import ContentConfig
from .models import (
    Article,
    ContentPlatform,
    ContentStatus,
    ContentTopic,
    EarningsRecord,
    PublishResult,
)

__all__ = [
    "Article",
    "ContentConfig",
    "ContentPlatform",
    "ContentStatus",
    "ContentTopic",
    "EarningsRecord",
    "PublishResult",
]
