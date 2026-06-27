"""Mirror.xyz platform adapter — on-chain publishing with tips."""

from __future__ import annotations

import logging

import httpx

from animus_content.models import Article, ContentPlatform, EarningsRecord, PublishResult

from .base import BasePlatform

logger = logging.getLogger(__name__)


class MirrorPlatform(BasePlatform):
    """Adapter for Mirror.xyz on-chain publishing."""

    platform = ContentPlatform.MIRROR

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: str) -> None:
        super().__init__(api_key=api_key, base_url=base_url or "https://mirror.xyz/api", **kwargs)

    async def publish(self, article: Article) -> PublishResult:
        """Publish article to Mirror.xyz."""
        if not self.is_configured():
            return PublishResult(
                article_id=article.id,
                platform=self.platform,
                success=False,
                error="Mirror API key not configured",
            )
        payload = {
            "title": article.title,
            "body": article.body,
            "tags": article.tags,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/entries",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return PublishResult(
                    article_id=article.id,
                    platform=self.platform,
                    success=True,
                    published_url=data.get("url", ""),
                )
        except httpx.HTTPError as exc:
            logger.error("Mirror publish failed: %s", exc)
            return PublishResult(
                article_id=article.id,
                platform=self.platform,
                success=False,
                error=str(exc),
            )

    async def fetch_earnings(self, article_id: str) -> list[EarningsRecord]:
        """Fetch tip earnings from Mirror."""
        if not self.is_configured():
            return []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self.base_url}/entries/{article_id}/earnings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    EarningsRecord(
                        article_id=article_id,
                        platform=self.platform,
                        amount=item.get("amount", 0.0),
                        token=item.get("token", "ETH"),
                        amount_usd=item.get("amount_usd", 0.0),
                    )
                    for item in data.get("earnings", [])
                ]
        except httpx.HTTPError as exc:
            logger.error("Mirror earnings fetch failed: %s", exc)
            return []

    async def health_check(self) -> bool:
        """Check if Mirror API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
