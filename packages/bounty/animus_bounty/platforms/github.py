"""GitHub bounty-labeled issue scanner."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from animus_bounty.config import BountyConfig
from animus_bounty.models import Bounty, Difficulty, Platform
from animus_bounty.platforms.base import BasePlatform, ScanResult

logger = logging.getLogger("animus_bounty.platforms.github")

# GitHub labels that commonly indicate bounties
BOUNTY_LABELS = [
    "bounty",
    "reward",
    "paid",
    "help wanted",
    "good first issue",
    "hacktoberfest",
    "cash",
    "funding",
]

# Default repos to scan (high-activity open source with bounties)
DEFAULT_SEARCH_QUERIES = [
    "label:bounty state:open language:python",
    'label:"good first issue" label:bounty state:open',
    "label:reward state:open",
]


def _estimate_difficulty(labels: list[str]) -> Difficulty:
    """Estimate difficulty from GitHub labels."""
    label_set = {label.lower() for label in labels}
    if label_set & {"good first issue", "beginner", "easy", "trivial"}:
        return Difficulty.EASY
    if label_set & {"medium", "intermediate"}:
        return Difficulty.MEDIUM
    if label_set & {"hard", "complex", "expert", "advanced"}:
        return Difficulty.HARD
    return Difficulty.MEDIUM


def _extract_reward(title: str, body: str, labels: list[str]) -> float:
    """Try to extract reward amount from issue text."""
    import re

    combined = f"{title} {body}"
    # Look for dollar amounts
    patterns = [
        r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)",
        r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|usd|dollars?)",
        r"reward[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)",
        r"bounty[:\s]*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined)
        if match:
            value_str = match.group(1).replace(",", "")
            try:
                return float(value_str)
            except ValueError:
                continue
    return 0.0


def _parse_issue(issue: dict[str, Any]) -> Bounty:
    """Parse a GitHub issue into a Bounty."""
    labels = [lbl["name"] if isinstance(lbl, dict) else lbl for lbl in issue.get("labels", [])]
    body = issue.get("body", "") or ""
    title = issue.get("title", "")
    repo_url = issue.get("repository_url", "")

    return Bounty(
        title=title,
        url=issue.get("html_url", ""),
        platform=Platform.GITHUB,
        description_snippet=body[:300],
        reward_usd=_extract_reward(title, body, labels),
        skills_required=[],
        difficulty=_estimate_difficulty(labels),
        auto_claimable=False,
        labels=labels,
        repo_url=repo_url,
        raw_data={"issue_number": issue.get("number"), "state": issue.get("state")},
    )


class GitHubPlatform(BasePlatform):
    """Scans GitHub for bounty-labeled issues."""

    name = "github"

    def __init__(self, config: BountyConfig):
        super().__init__(config)
        self._token = config.credentials.github_token

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _scan(self) -> ScanResult:
        """Search GitHub for bounty-labeled issues."""
        bounties: list[Bounty] = []
        errors: list[str] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in DEFAULT_SEARCH_QUERIES:
                try:
                    resp = await client.get(
                        "https://api.github.com/search/issues",
                        params={"q": query, "per_page": 30, "sort": "created", "order": "desc"},
                        headers=self._headers(),
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("items", []):
                            bounties.append(_parse_issue(item))
                    else:
                        errors.append(f"GitHub search returned {resp.status_code} for: {query}")
                except httpx.HTTPError as e:
                    errors.append(f"GitHub search failed for {query}: {e}")

        return ScanResult(platform_name=self.name, bounties=bounties, errors=errors)

    async def health_check(self) -> bool:
        """Check GitHub API reachability."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.github.com/rate_limit",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
