"""Medium platform adapter — Partner Program earnings."""

from __future__ import annotations

import logging

import httpx

from animus_content.models import Article, ContentPlatform, EarningsRecord, PublishResult

from .base import BasePlatform

logger = logging.getLogger(__name__)


class MediumPlatform(BasePlatform):
    """Adapter for Medium Partner Program publishing."""

    platform = ContentPlatform.MEDIUM

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: str) -> None:
        super().__init__(
            api_key=api_key, base_url=base_url or "https://api.medium.com/v1", **kwargs
        )
        self._user_id: str = self.extra.get("user_id", "")

    async def _resolve_user_id(self) -> str:
        """Resolve user ID from API token if not cached."""
        if self._user_id:
            return self._user_id
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.base_url}/me",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            self._user_id = data.get("data", {}).get("id", "")
        return self._user_id

    async def publish(self, article: Article) -> PublishResult:
        """Publish article to Medium."""
        if not self.is_configured():
            return PublishResult(
                article_id=article.id,
                platform=self.platform,
                success=False,
                error="Medium API key not configured",
            )
        try:
            user_id = await self._resolve_user_id()
            if not user_id:
                return PublishResult(
                    article_id=article.id,
                    platform=self.platform,
                    success=False,
                    error="Could not resolve Medium user ID",
                )
            payload = {
                "title": article.title,
                "contentFormat": "markdown",
                "content": article.body,
                "tags": article.tags[:5],
                "publishStatus": "public",
            }
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/users/{user_id}/posts",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return PublishResult(
                    article_id=article.id,
                    platform=self.platform,
                    success=True,
                    published_url=data.get("data", {}).get("url", ""),
                )
        except httpx.HTTPError as exc:
            logger.error("Medium publish failed: %s", exc)
            return PublishResult(
                article_id=article.id,
                platform=self.platform,
                success=False,
                error=str(exc),
            )

    async def fetch_earnings(self, article_id: str) -> list[EarningsRecord]:
        """Fetch Partner Program earnings. Medium has no public earnings API."""
        return []

    async def health_check(self) -> bool:
        """Check if Medium API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.base_url}/me",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return resp.status_code in (200, 401)
        except httpx.HTTPError:
            return False
