"""
MER source — EVE Online Monthly Economic Report availability tracker.

CCP publishes the MER's raw-data dump (a zip of CSVs) at a stable, monthly
URL:  ``https://web.ccpgamescdn.com/aws/community/EVEOnline_MER_<YYYYMM>.zip``
and the report itself at ``eveonline.com/news/view/monthly-economic-report-
<month>-<year>``.

This source does *not* download the ~65 MB zip — it issues a cheap HTTP HEAD
for the last ``lookback_months`` months and emits a ``SourceItem`` for each
month whose zip exists, so "is there a new MER?" becomes part of Lugh's
normal scan cadence. The downstream synthesis pass (Ogma / the eve-market-
intel board) does the actual download + CSV digest; ``metadata["zip_url"]``
points the way.

Fail-open: a missing network, a timeout, or a non-200/404 never raises — the
caller gets fewer items, not an exception. Stdlib only.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone

from animus.lugh.sources.base import SourceItem

logger = logging.getLogger(__name__)

USER_AGENT = "animus-lugh/1.0 (+https://github.com/your-org/animus)"
ZIP_URL = "https://web.ccpgamescdn.com/aws/community/EVEOnline_MER_{ym}.zip"
NEWS_URL = "https://www.eveonline.com/news/view/monthly-economic-report-{month}-{year}"
HTTP_TIMEOUT_SECONDS = 15
DEFAULT_LOOKBACK_MONTHS = 3
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def _recent_year_months(n: int, today: date | None = None) -> list[tuple[int, int]]:
    """The (year, month) pairs for the last ``n`` calendar months, oldest first."""
    today = today or datetime.now(timezone.utc).date()
    y, m = today.year, today.month
    out: list[tuple[int, int]] = []
    for _ in range(max(1, n)):
        out.append((y, m))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return list(reversed(out))


def _zip_exists(url: str, timeout: float = HTTP_TIMEOUT_SECONDS) -> bool:
    """True iff a HEAD on ``url`` returns 200. Any error → False (fail-open)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        if e.code != 404:
            logger.warning("MER HEAD %s on %s: %s", e.code, url, e.reason)
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("MER HEAD transport error on %s: %s", url, e)
        return False


@dataclass
class MERSource:
    """Tracks publication of the EVE Online Monthly Economic Report.

    Attributes:
        lookback_months: how many recent months to probe each fetch (covers
            the publish lag — the May report drops in mid-June, etc.).
        tags: default tags applied to every emitted item.
    """

    lookback_months: int = DEFAULT_LOOKBACK_MONTHS
    tags: list[str] | None = None

    @property
    def source_id(self) -> str:
        return "mer"

    def fetch(self, limit: int | None = None) -> Iterable[SourceItem]:
        pairs = _recent_year_months(self.lookback_months)
        if limit is not None:
            pairs = pairs[-limit:]
        for year, month in pairs:
            ym = f"{year}{month:02d}"
            zip_url = ZIP_URL.format(ym=ym)
            if not _zip_exists(zip_url):
                continue
            month_name = _MONTHS[month - 1]
            news_url = NEWS_URL.format(month=month_name, year=year)
            yield SourceItem(
                source_id=self.source_id,
                item_id=ym,  # one item per report month — stable, dedups naturally
                title=f"EVE Online Monthly Economic Report — {month_name.capitalize()} {year}",
                url=news_url,
                published=datetime(year, month, 1, tzinfo=timezone.utc),  # report month
                summary=(
                    f"MER raw-data dump available for {month_name.capitalize()} {year}: "
                    f"{zip_url} (zip of CSVs — sinks/faucets, money supply, regional "
                    "figures, mining by region, economy indices). Synthesis pass should "
                    "download + digest; this item only marks availability."
                ),
                author="CCP Games",
                tags=list(self.tags) if self.tags else ["eve-economy", "eve"],
                raw_text="",
                metadata={
                    "report_month": ym,
                    "zip_url": zip_url,
                    "news_url": news_url,
                },
            )


def default_mer_sources() -> list[MERSource]:
    """Single MER source — there's only one report series."""
    return [MERSource()]
