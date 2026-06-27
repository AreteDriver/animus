"""Hunter for social media crypto opportunities (Twitter/X, Reddit, Discord)."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import httpx

from animus_prospector.hunters.aggregator_hunter import _guess_chain
from animus_prospector.hunters.base import BaseHunter
from animus_prospector.models import (
    HunterResult,
    Opportunity,
    OpportunityType,
    SourceType,
)

if TYPE_CHECKING:
    from animus_prospector.config import ProspectorConfig

logger = logging.getLogger("animus_prospector.hunters.social")

REDDIT_SUBREDDITS = [
    "CryptoAirdrops",
    "airdropalert",
    "FreeCrypto",
    "CryptoCurrency",
    "defi",
]

GIVEAWAY_PATTERNS = [
    r"(?i)giveaway.*\$?\d+",
    r"(?i)free\s+(?:btc|eth|sol|sui|crypto|token|nft)",
    r"(?i)airdrop.*(?:live|now|today|open)",
    r"(?i)claim\s+(?:free|your)\s+(?:token|crypto|reward)",
]


class SocialHunter(BaseHunter):
    """Monitors social media for crypto opportunities.

    Scans Reddit (public JSON API) for airdrop/faucet/giveaway posts.
    Twitter/X requires API key — configured via hunters.sources.social.twitter_bearer.
    """

    name = "social"

    def __init__(self, config: ProspectorConfig):
        super().__init__(config)
        social_cfg = config.hunters.sources.get("social", {})
        self.subreddits = social_cfg.get("subreddits", REDDIT_SUBREDDITS)
        self.twitter_bearer = social_cfg.get("twitter_bearer", "")

    async def _scan(self) -> HunterResult:
        opportunities: list[Opportunity] = []
        errors: list[str] = []

        # Reddit scan
        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "AnimusProspector/0.1 (research bot)"},
        ) as client:
            for sub in self.subreddits:
                try:
                    opps = await self._scan_reddit(client, sub)
                    opportunities.extend(opps)
                except Exception as e:
                    errors.append(f"Reddit r/{sub}: {e}")

        # Dedup
        seen: set[str] = set()
        unique: list[Opportunity] = []
        for opp in opportunities:
            if opp.url not in seen:
                seen.add(opp.url)
                unique.append(opp)

        return HunterResult(
            hunter_name=self.name,
            opportunities=unique[: self.config.hunters.max_results_per_scan],
            errors=errors,
        )

    async def _scan_reddit(self, client: httpx.AsyncClient, subreddit: str) -> list[Opportunity]:
        """Scan a subreddit's new posts for opportunities."""
        url = f"https://www.reddit.com/r/{subreddit}/new.json"
        resp = await client.get(url, params={"limit": 25})

        if resp.status_code == 429:
            logger.debug(f"Reddit rate limited on r/{subreddit}")
            return []
        resp.raise_for_status()

        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        opportunities = []

        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            post_url = post_data.get("url", "")
            permalink = post_data.get("permalink", "")

            combined = title + " " + selftext
            if not _matches_opportunity(combined):
                continue

            opp_type = _classify_reddit_post(title, selftext)
            chain = _guess_chain(combined)

            # Use the linked URL if it points elsewhere, otherwise use reddit permalink
            target_url = post_url if post_url and not post_url.startswith("/r/") else ""
            if not target_url or "reddit.com" in target_url:
                target_url = f"https://www.reddit.com{permalink}"

            opp = Opportunity(
                title=title[:200],
                url=target_url,
                opportunity_type=opp_type,
                source_type=SourceType.REDDIT,
                source_url=f"https://www.reddit.com/r/{subreddit}",
                chain=chain,
                tags=["reddit", subreddit],
                raw_data={
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "subreddit": subreddit,
                },
            )
            opportunities.append(opp)

        return opportunities

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.head(
                    "https://www.reddit.com/r/CryptoCurrency.json",
                    headers={"User-Agent": "AnimusProspector/0.1"},
                )
                return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


def _matches_opportunity(text: str) -> bool:
    """Check if text matches any giveaway/opportunity pattern."""
    for pattern in GIVEAWAY_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _classify_reddit_post(title: str, body: str) -> OpportunityType:
    """Classify a Reddit post into an opportunity type."""
    text = (title + " " + body).lower()
    if "giveaway" in text:
        return OpportunityType.GIVEAWAY
    if "faucet" in text:
        return OpportunityType.FAUCET
    if "airdrop" in text:
        return OpportunityType.AIRDROP
    if "testnet" in text:
        return OpportunityType.TESTNET
    if "bounty" in text:
        return OpportunityType.BOUNTY
    if "quest" in text:
        return OpportunityType.QUEST
    return OpportunityType.AIRDROP
