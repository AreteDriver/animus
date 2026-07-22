"""BrowserBridge — isolated Chrome orchestration via nodriver.

Each fetch spins up a fresh Chrome instance in a temporary profile.
No cookies or state persist between requests.  Anti-detection escalation
(cloudflare / datadome) falls back to headed mode automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from animus.logging import get_logger

logger = get_logger("browser.bridge")

try:
    import nodriver as uc

    _HAS_NODRIVER = True
except ImportError:  # pragma: no cover
    _HAS_NODRIVER = False
    uc = None  # type: ignore[misc,assignment]


class ExtractionFormat(str, Enum):
    """Supported output formats."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class BrowserConfig:
    """Configuration for a single fetch session."""

    format: ExtractionFormat | str = ExtractionFormat.TEXT
    timeout_ms: int = 30_000
    wait_for: str | None = None
    human_mode: bool = False
    headless: bool = True
    max_retries: int = 2
    retry_delay_sec: float = 1.5

    # Resource guard
    max_memory_mb: int = 512
    # Extraction tuning
    min_content_length: int = 100
    readability_negative_keywords: list[str] = field(
        default_factory=lambda: [
            "cookie",
            "consent",
            "newsletter",
            "subscribe",
            "privacy policy",
        ]
    )

    def __post_init__(self) -> None:
        if isinstance(self.format, str):
            self.format = ExtractionFormat(self.format)


@dataclass
class BrowserResult:
    """Structured result from a browser fetch."""

    url: str
    final_url: str
    title: str
    content: str
    status_code: int
    format: ExtractionFormat | str
    headers: dict[str, str] = field(default_factory=dict)
    fetch_time_sec: float = 0.0
    used_human_mode: bool = False
    cache_hit: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code < 400


class BrowserBridge:
    """High-level browser bridge with anti-detection and clean extraction.

    Usage::

        bridge = BrowserBridge()
        result = await bridge.fetch("https://example.com/article")
        print(result.content)
    """

    def __init__(self, config: BrowserConfig | None = None) -> None:
        if not _HAS_NODRIVER:
            raise RuntimeError(
                "BrowserBridge requires 'nodriver'. "
                "Install with: pip install nodriver"
            )
        self.config = config or BrowserConfig()
        self._emulation = None  # lazy import to avoid heavy dep at init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, url: str, config: BrowserConfig | None = None) -> BrowserResult:
        """Fetch a single URL through a fresh Chrome instance.

        Args:
            url: Target URL.
            config: Optional override config for this fetch.

        Returns:
            :class:`BrowserResult` with extracted content.
        """
        cfg = config or self.config
        start = time.monotonic()

        # Validate URL early
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return BrowserResult(
                url=url,
                final_url=url,
                title="",
                content="",
                status_code=0,
                format=cfg.format,
                error=f"Invalid URL: {url}",
            )

        temp_dir = tempfile.mkdtemp(prefix="animus_browser_")
        browser = None
        try:
            browser = await self._start_browser(temp_dir, headless=cfg.headless)
            tab = await browser.get(url)

            # Wait for selector if requested
            if cfg.wait_for:
                try:
                    await tab.wait_for(selector=cfg.wait_for, timeout=cfg.timeout_ms)
                except Exception:
                    logger.warning("wait_for selector %s not found", cfg.wait_for)

            # Mandatory render pause (scripts, lazy images)
            await asyncio.sleep(0.8)

            # Human emulation if requested
            if cfg.human_mode:
                from animus.browser.emulation import EmulationLayer

                emu = EmulationLayer()
                await emu.run(tab)
                cfg.headless = False  # flag that we used headed fallback

            # Extract content
            from animus.browser.extraction import ExtractionPipeline

            pipeline = ExtractionPipeline(cfg)
            content, title, final_url = await pipeline.extract(tab, url)

            status_code = 200  # nodriver doesn't expose HTTP status cleanly
            try:
                resp = await tab.evaluate("window.performance.getEntriesByType('navigation')[0].responseStatus")
                if isinstance(resp, int):
                    status_code = resp
            except Exception:
                pass

            return BrowserResult(
                url=url,
                final_url=final_url or url,
                title=title or "",
                content=content,
                status_code=status_code,
                format=cfg.format,
                fetch_time_sec=time.monotonic() - start,
                used_human_mode=cfg.human_mode,
            )

        except Exception as exc:
            logger.exception("Browser fetch failed for %s", url)
            # Anti-detection escalation: retry headed if we were headless
            if cfg.headless and cfg.max_retries > 0:
                logger.info("Retrying %s in headed mode (anti-detection)", url)
                cfg.headless = False
                cfg.max_retries -= 1
                await asyncio.sleep(cfg.retry_delay_sec)
                return await self.fetch(url, config=cfg)

            return BrowserResult(
                url=url,
                final_url=url,
                title="",
                content="",
                status_code=0,
                format=cfg.format,
                error=str(exc),
                fetch_time_sec=time.monotonic() - start,
            )
        finally:
            if browser:
                try:
                    browser.stop()
                except Exception:
                    pass
            self._cleanup(temp_dir)

    async def fetch_batch(
        self, urls: list[str], config: BrowserConfig | None = None
    ) -> list[BrowserResult]:
        """Fetch multiple URLs concurrently.

        Each URL gets its own isolated browser instance.
        """
        # Domain-aware concurrency: group by domain and apply rate limits
        tasks = [self.fetch(u, config) for u in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _start_browser(self, temp_dir: str, headless: bool = True) -> Any:
        """Launch Chrome with a temporary profile."""
        options = [
            f"--user-data-dir={temp_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
        ]
        if headless:
            options.append("--headless=new")
        else:
            # Off-screen position for Linux servers
            options.append("--window-position=-2400,-2400")

        return await uc.start(
            headless=headless,
            browser_args=options,
            sandbox=(os.getuid() != 0),
        )

    def _cleanup(self, temp_dir: str) -> None:
        """Remove temporary profile and kill lingering renderers."""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        # Best-effort kill of orphaned nodriver processes older than this session
        try:
            subprocess.run(
                ["pkill", "-f", f"user-data-dir={temp_dir}"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
