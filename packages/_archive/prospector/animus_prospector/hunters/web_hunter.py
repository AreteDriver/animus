"""Hunter for web-based faucet/airdrop discovery via search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from animus_prospector.hunters.aggregator_hunter import _guess_chain, _is_crypto_opportunity
from animus_prospector.hunters.base import BaseHunter
from animus_prospector.models import (
    HunterResult,
    Opportunity,
    OpportunityType,
    SourceType,
)

if TYPE_CHECKING:
    from animus_prospector.config import ProspectorConfig

logger = logging.getLogger("animus_prospector.hunters.web")

SEARCH_QUERIES = [
    "new crypto faucet 2026",
    "free crypto airdrop this week",
    "new testnet faucet",
    "crypto giveaway legitimate",
    "web3 quest rewards",
    "new DeFi airdrop farming",
    "sui airdrop opportunity",
    "base chain airdrop",
]


class WebHunter(BaseHunter):
    """Discovers opportunities via web search.

    Uses search engines to find new faucets, airdrops, and giveaways
    that haven't yet appeared on aggregator sites.
    """

    name = "web"

    def __init__(self, config: ProspectorConfig):
        super().__init__(config)
        self.queries = config.hunters.sources.get("web", {}).get("queries", SEARCH_QUERIES)
        self.search_endpoint = config.hunters.sources.get("web", {}).get("search_endpoint", "")

    async def _scan(self) -> HunterResult:
        opportunities: list[Opportunity] = []
        errors: list[str] = []

        if not self.search_endpoint:
            return HunterResult(
                hunter_name=self.name,
                opportunities=[],
                errors=["No search endpoint configured — set hunters.sources.web.search_endpoint"],
            )

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AnimusProspector/0.1)"},
        ) as client:
            for query in self.queries:
                try:
                    results = await self._search(client, query)
                    opportunities.extend(results)
                except Exception as e:
                    errors.append(f"Search '{query}': {e}")

        # Dedup by URL
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

    async def _search(self, client: httpx.AsyncClient, query: str) -> list[Opportunity]:
        """Execute a search query and parse results into opportunities."""
        resp = await client.get(
            self.search_endpoint,
            params={"q": query, "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()

        opportunities = []
        results = data.get("results", data.get("items", []))

        for item in results:
            title = item.get("title", "")
            url = item.get("url", item.get("link", ""))
            snippet = item.get("snippet", item.get("description", ""))

            if not url or not _is_crypto_opportunity(title + " " + snippet, url):
                continue

            opp_type = _classify_opportunity(title, snippet)
            chain = _guess_chain(title + " " + snippet)

            opp = Opportunity(
                title=title[:200],
                url=url,
                opportunity_type=opp_type,
                source_type=SourceType.WEB_SCRAPE,
                source_url=self.search_endpoint,
                chain=chain,
                tags=["web_search", query.split()[0]],
                raw_data={"query": query, "snippet": snippet[:500]},
            )
            opportunities.append(opp)

        return opportunities


def _classify_opportunity(title: str, description: str) -> OpportunityType:
    """Classify an opportunity type from text."""
    text = (title + " " + description).lower()
    if "faucet" in text:
        return OpportunityType.FAUCET
    if "airdrop" in text or "token drop" in text:
        return OpportunityType.AIRDROP
    if "giveaway" in text:
        return OpportunityType.GIVEAWAY
    if "quest" in text or "galxe" in text or "layer3" in text or "zealy" in text:
        return OpportunityType.QUEST
    if "bounty" in text:
        return OpportunityType.BOUNTY
    if "grant" in text:
        return OpportunityType.GRANT
    if "testnet" in text:
        return OpportunityType.TESTNET
    if "yield" in text or "apy" in text or "staking" in text:
        return OpportunityType.DEFI_YIELD
    return OpportunityType.AIRDROP  # default
