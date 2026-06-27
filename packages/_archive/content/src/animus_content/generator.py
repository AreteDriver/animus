"""LLM content generation with topic selection."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from .config import ContentConfig
from .models import Article, ContentPlatform, ContentStatus, ContentTopic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert crypto/web3 content writer. Write engaging, informative articles
that educate readers about blockchain technology, DeFi, NFTs, and the crypto ecosystem.

Rules:
- Write in a clear, authoritative voice
- Include actionable insights
- Avoid financial advice disclaimers in the body (add at end if needed)
- Use markdown formatting
- Target the specified word count
"""

ARTICLE_PROMPT_TEMPLATE = """Write an article about: {topic}

Keywords to include: {keywords}
Target platform: {platform}
Target word count: {word_count}
Tags: {tags}

Write the article now. Start with a compelling title on the first line (no # prefix),
then a blank line, then the body in markdown."""


class ContentGenerator:
    """Generates content using LLM providers."""

    def __init__(self, config: ContentConfig) -> None:
        self.config = config

    async def generate(
        self,
        topic: ContentTopic,
        word_count: int | None = None,
        extra_tags: list[str] | None = None,
    ) -> Article:
        """Generate an article for the given topic."""
        target_words = word_count or self.config.min_word_count
        tags = list(set(self.config.default_tags + (extra_tags or []) + topic.keywords))

        prompt = ARTICLE_PROMPT_TEMPLATE.format(
            topic=topic.topic,
            keywords=", ".join(topic.keywords),
            platform=topic.target_platform.value,
            word_count=target_words,
            tags=", ".join(tags),
        )

        raw = await self._call_llm(prompt)
        title, body = self._parse_response(raw)

        article = Article(
            title=title,
            body=body,
            platform=topic.target_platform,
            topic=topic.topic,
            status=ContentStatus.GENERATED,
            tags=tags,
            created_at=datetime.now(),
        )
        article.compute_word_count()
        return article

    async def suggest_topics(
        self,
        platform: ContentPlatform = ContentPlatform.MIRROR,
        count: int = 5,
    ) -> list[ContentTopic]:
        """Use LLM to suggest trending content topics."""
        prompt = (
            f"Suggest {count} trending crypto/web3 topics for articles on {platform.value}. "
            "For each topic, provide: topic name, 3-5 keywords, and estimated engagement (0-1). "
            "Format each as: TOPIC: <name> | KEYWORDS: <comma-separated> | ENGAGEMENT: <float>"
        )
        raw = await self._call_llm(prompt)
        return self._parse_topics(raw, platform)

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        provider = self.config.llm.provider.lower()
        if provider == "ollama":
            return await self._call_ollama(prompt)
        elif provider == "anthropic":
            return await self._call_anthropic(prompt)
        else:
            return await self._call_ollama(prompt)

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama for content generation."""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{self.config.llm.base_url}/api/generate",
                    json={
                        "model": self.config.llm.model,
                        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": self.config.llm.temperature,
                            "num_predict": self.config.llm.max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data.get("response", "")
        except httpx.HTTPError as exc:
            logger.error("Ollama call failed: %s", exc)
            return ""

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API for content generation."""
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    json={
                        "model": self.config.llm.model,
                        "max_tokens": self.config.llm.max_tokens,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    headers={
                        "x-api-key": self.config.llm.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                content_blocks = data.get("content", [])
                return content_blocks[0].get("text", "") if content_blocks else ""
        except httpx.HTTPError as exc:
            logger.error("Anthropic call failed: %s", exc)
            return ""

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, str]:
        """Parse LLM response into title and body."""
        if not raw.strip():
            return ("Untitled", "")
        lines = raw.strip().split("\n")
        title = lines[0].strip().lstrip("# ").strip()
        body_lines = lines[1:]
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        body = "\n".join(body_lines)
        return (title or "Untitled", body)

    @staticmethod
    def _parse_topics(raw: str, platform: ContentPlatform) -> list[ContentTopic]:
        """Parse LLM topic suggestions."""
        topics: list[ContentTopic] = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or "TOPIC:" not in line:
                continue
            try:
                parts: dict[str, str] = {}
                for segment in line.split("|"):
                    segment = segment.strip()
                    if ":" in segment:
                        key, val = segment.split(":", 1)
                        parts[key.strip().upper()] = val.strip()
                topic_name = parts.get("TOPIC", "")
                keywords_str = parts.get("KEYWORDS", "")
                engagement_str = parts.get("ENGAGEMENT", "0.5")
                if topic_name:
                    topics.append(
                        ContentTopic(
                            topic=topic_name,
                            keywords=[k.strip() for k in keywords_str.split(",") if k.strip()],
                            target_platform=platform,
                            estimated_engagement=float(engagement_str),
                        )
                    )
            except (ValueError, KeyError):
                continue
        return topics
