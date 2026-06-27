"""Generic platform adapter for custom bounty sources."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from animus_bounty.config import BountyConfig
from animus_bounty.models import Bounty, Difficulty, Platform
from animus_bounty.platforms.base import BasePlatform, ScanResult

logger = logging.getLogger("animus_bounty.platforms.generic")


def parse_generic_bounty(data: dict[str, Any]) -> Bounty:
    """Parse a generic bounty dict into a Bounty model.

    Expected keys: title, url, reward_usd (optional), description (optional),
    skills (optional list), difficulty (optional), labels (optional list).
    """
    return Bounty(
        title=data.get("title", "Untitled"),
        url=data.get("url", ""),
        platform=Platform.GENERIC,
        description_snippet=data.get("description", "")[:300],
        reward_usd=float(data.get("reward_usd", 0) or 0),
        reward_token=data.get("reward_token", ""),
        chain=data.get("chain", ""),
        skills_required=data.get("skills", []),
        difficulty=Difficulty(data.get("difficulty", "medium")),
        auto_claimable=data.get("auto_claimable", False),
        labels=data.get("labels", []),
        repo_url=data.get("repo_url", ""),
        raw_data=data,
    )


class GenericPlatform(BasePlatform):
    """Generic adapter that fetches bounties from a configurable JSON endpoint.

    Expects the endpoint to return a JSON array of bounty objects.
    Configure via BountyConfig.credentials or environment variables.
    """

    name = "generic"

    def __init__(self, config: BountyConfig, endpoint: str = ""):
        super().__init__(config)
        self._endpoint = endpoint

    async def _scan(self) -> ScanResult:
        """Fetch bounties from the configured endpoint."""
        if not self._endpoint:
            return ScanResult(
                platform_name=self.name,
                errors=["No endpoint configured for generic platform"],
            )

        bounties: list[Bounty] = []
        errors: list[str] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(self._endpoint)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data if isinstance(data, list) else data.get("bounties", [])
                    for item in items:
                        bounties.append(parse_generic_bounty(item))
                else:
                    errors.append(f"Generic endpoint returned {resp.status_code}")
        except httpx.HTTPError as e:
            errors.append(f"Generic endpoint error: {e}")

        return ScanResult(platform_name=self.name, bounties=bounties, errors=errors)

    async def health_check(self) -> bool:
        """Check if the endpoint is reachable."""
        if not self._endpoint:
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.head(self._endpoint)
                return resp.status_code < 500
        except httpx.HTTPError:
            return False
