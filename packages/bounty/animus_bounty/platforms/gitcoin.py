"""Gitcoin bounty scanner."""

from __future__ import annotations

import logging

import httpx

from animus_bounty.config import BountyConfig
from animus_bounty.models import Bounty, Difficulty, Platform
from animus_bounty.platforms.base import BasePlatform, ScanResult

logger = logging.getLogger("animus_bounty.platforms.gitcoin")

GITCOIN_GRANTS_API = "https://grants-stack-indexer-v2.gitcoin.co/graphql"


def _parse_grant(data: dict) -> Bounty:
    """Parse a Gitcoin grant/bounty into a Bounty model."""
    title = data.get("title", data.get("name", "Untitled"))
    url = data.get("url", data.get("website", ""))
    description = data.get("description", "")
    amount = float(data.get("amountUSD", 0) or 0)

    return Bounty(
        title=title,
        url=url,
        platform=Platform.GITCOIN,
        description_snippet=description[:300],
        reward_usd=amount,
        reward_token=data.get("token", ""),
        chain=data.get("chain", ""),
        skills_required=data.get("tags", []),
        difficulty=Difficulty.MEDIUM,
        auto_claimable=False,
        labels=data.get("tags", []),
        raw_data={"gitcoin_id": data.get("id")},
    )


class GitcoinPlatform(BasePlatform):
    """Scans Gitcoin for active grants and bounties."""

    name = "gitcoin"

    def __init__(self, config: BountyConfig):
        super().__init__(config)
        self._api_key = config.credentials.gitcoin_api_key

    async def _scan(self) -> ScanResult:
        """Fetch active Gitcoin bounties."""
        bounties: list[Bounty] = []
        errors: list[str] = []

        # Gitcoin uses GraphQL — query for active rounds/projects
        query = """
        {
            rounds(first: 10, filter: {chainId: {equalTo: 1}}) {
                id
                roundMetadata
                applicationsStartTime
                applicationsEndTime
            }
        }
        """

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"Content-Type": "application/json"}
                if self._api_key:
                    headers["Authorization"] = f"Bearer {self._api_key}"

                resp = await client.post(
                    GITCOIN_GRANTS_API,
                    json={"query": query},
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    rounds = data.get("data", {}).get("rounds", [])
                    for round_data in rounds:
                        metadata = round_data.get("roundMetadata", {}) or {}
                        bounty = Bounty(
                            title=metadata.get("name", f"Gitcoin Round {round_data.get('id', '')}"),
                            url=f"https://explorer.gitcoin.co/#/round/1/{round_data.get('id', '')}",
                            platform=Platform.GITCOIN,
                            description_snippet=str(
                                metadata.get("eligibility", {}).get("description", "")
                            )[:300],
                            reward_usd=0.0,
                            chain="eth",
                            difficulty=Difficulty.MEDIUM,
                            raw_data=round_data,
                        )
                        bounties.append(bounty)
                else:
                    errors.append(f"Gitcoin API returned {resp.status_code}")
        except httpx.HTTPError as e:
            errors.append(f"Gitcoin API error: {e}")

        return ScanResult(platform_name=self.name, bounties=bounties, errors=errors)

    async def health_check(self) -> bool:
        """Check Gitcoin API reachability."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    GITCOIN_GRANTS_API,
                    json={"query": "{ __typename }"},
                    headers={"Content-Type": "application/json"},
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
