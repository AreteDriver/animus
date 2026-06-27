"""Hunter for faucet/airdrop aggregator sites."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import httpx

from animus_prospector.hunters.base import BaseHunter
from animus_prospector.models import (
    Chain,
    HunterResult,
    Opportunity,
    OpportunityType,
    SourceType,
)

if TYPE_CHECKING:
    from animus_prospector.config import ProspectorConfig

logger = logging.getLogger("animus_prospector.hunters.aggregator")

# Known aggregator endpoints and their parsers
DEFAULT_SOURCES = [
    {
        "name": "airdrops_io",
        "url": "https://airdrops.io/latest-airdrops/",
        "type": OpportunityType.AIRDROP,
    },
    {
        "name": "defillama_airdrops",
        "url": "https://defillama.com/airdrops",
        "type": OpportunityType.AIRDROP,
    },
    {
        "name": "faucetlink",
        "url": "https://faucetlink.to/",
        "type": OpportunityType.FAUCET,
    },
    {
        "name": "layer3_quests",
        "url": "https://layer3.xyz/quests",
        "type": OpportunityType.QUEST,
    },
    {
        "name": "galxe_campaigns",
        "url": "https://galxe.com/earn",
        "type": OpportunityType.QUEST,
    },
    {
        "name": "zealy_communities",
        "url": "https://zealy.io/explore",
        "type": OpportunityType.QUEST,
    },
]


class AggregatorHunter(BaseHunter):
    """Scrapes known aggregator sites for new opportunities.

    These are high-signal, low-noise sources — curated lists of
    airdrops, faucets, and quests.
    """

    name = "aggregator"

    def __init__(self, config: ProspectorConfig):
        super().__init__(config)
        self.sources = config.hunters.sources.get("aggregator", {}).get("urls", DEFAULT_SOURCES)

    async def _scan(self) -> HunterResult:
        opportunities: list[Opportunity] = []
        errors: list[str] = []

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AnimusProspector/0.1)"},
            follow_redirects=True,
        ) as client:
            for source in self.sources:
                try:
                    opps = await self._scan_source(client, source)
                    opportunities.extend(opps)
                except Exception as e:
                    errors.append(f"{source.get('name', source.get('url'))}: {e}")
                    logger.debug(f"Aggregator scan failed for {source}: {e}")

        return HunterResult(
            hunter_name=self.name,
            opportunities=opportunities,
            errors=errors,
        )

    async def _scan_source(self, client: httpx.AsyncClient, source: dict) -> list[Opportunity]:
        """Scrape a single aggregator source."""
        url = source.get("url", "") if isinstance(source, dict) else source
        name = source.get("name", url) if isinstance(source, dict) else url
        opp_type = (
            source.get("type", OpportunityType.AIRDROP)
            if isinstance(source, dict)
            else OpportunityType.AIRDROP
        )

        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

        # Extract links and titles from HTML
        opportunities = []
        links = _extract_links(html, url)

        for link_url, link_text in links:
            if not _is_crypto_opportunity(link_text, link_url):
                continue

            chain = _guess_chain(link_text + " " + link_url)

            opp = Opportunity(
                title=link_text.strip()[:200],
                url=link_url,
                opportunity_type=opp_type,
                source_type=SourceType.AGGREGATOR,
                source_url=url,
                chain=chain,
                tags=[name],
            )
            opportunities.append(opp)

        return opportunities[: self.config.hunters.max_results_per_scan]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.head("https://airdrops.io/")
                return resp.status_code < 500
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


def _extract_links(html: str, base_url: str) -> list[tuple[str, str]]:
    """Extract (url, text) pairs from HTML."""
    pattern = r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
    results = []
    for href, text in matches:
        clean_text = re.sub(r"<[^>]+>", "", text).strip()
        if not clean_text or len(clean_text) < 3:
            continue
        # Resolve relative URLs
        if href.startswith("/"):
            from urllib.parse import urlparse

            parsed = urlparse(base_url)
            href = f"{parsed.scheme}://{parsed.netloc}{href}"
        elif not href.startswith("http"):
            continue
        results.append((href, clean_text))
    return results


CRYPTO_KEYWORDS = {
    "airdrop",
    "faucet",
    "claim",
    "free",
    "earn",
    "reward",
    "token",
    "nft",
    "quest",
    "bounty",
    "testnet",
    "giveaway",
    "drop",
    "mining",
    "stake",
}


def _is_crypto_opportunity(text: str, url: str) -> bool:
    """Check if a link looks like a crypto opportunity."""
    combined = (text + " " + url).lower()
    return any(kw in combined for kw in CRYPTO_KEYWORDS)


CHAIN_PATTERNS: dict[str, Chain] = {
    "bitcoin": Chain.BTC,
    "btc": Chain.BTC,
    "ethereum": Chain.ETH,
    "eth": Chain.ETH,
    "base": Chain.BASE,
    "sui": Chain.SUI,
    "solana": Chain.SOL,
    "sol": Chain.SOL,
    "arbitrum": Chain.ARB,
    "optimism": Chain.OP,
    "polygon": Chain.MATIC,
    "matic": Chain.MATIC,
}


def _guess_chain(text: str) -> Chain:
    """Guess the blockchain from text content."""
    lower = text.lower()
    for keyword, chain in CHAIN_PATTERNS.items():
        if keyword in lower:
            return chain
    return Chain.UNKNOWN
