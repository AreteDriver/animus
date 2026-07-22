"""Content extraction pipeline.

Attempts Readability algorithm first; falls back to deduplicated raw text
when Readability produces cookie banners, repetitive stubs, or unusually
short output.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any

from animus.logging import get_logger

logger = get_logger("browser.extraction")

try:
    from readability import Document

    _HAS_READABILITY = True
except ImportError:  # pragma: no cover
    _HAS_READABILITY = False


class ExtractionPipeline:
    """Extract clean content from a rendered page."""

    def __init__(self, config: Any) -> None:
        self.config = config

    async def extract(self, tab: Any, original_url: str) -> tuple[str, str, str]:
        """Return (content, title, final_url)."""
        raw_html = await tab.evaluate("document.documentElement.outerHTML")
        title = await tab.evaluate("document.title") or ""
        final_url = await tab.evaluate("window.location.href") or original_url

        if self.config.format.value == "html":
            return raw_html, title, final_url

        # Try Readability first
        content = ""
        if _HAS_READABILITY:
            try:
                doc = Document(raw_html)
                summary = doc.summary()
                content = self._html_to_text(summary) if self.config.format.value == "text" else summary
                # Quality gate: reject cookie/consent noise or very short output
                if self._is_low_quality(content):
                    logger.debug("Readability output rejected as low quality; falling back")
                    content = ""
            except Exception:
                logger.debug("Readability parse failed; falling back to raw")
                content = ""

        # Fallback: raw body text
        if not content:
            raw_text = await tab.evaluate("document.body.innerText")
            content = self._clean_text(raw_text)

        return content, title, final_url

    def _html_to_text(self, html_str: str) -> str:
        """Crude HTML→text for Readability output."""
        # Strip tags
        text = re.sub(r"<[^>]+>", "", html_str)
        text = html.unescape(text)
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Deduplicate lines, remove short/noise lines."""
        lines = text.splitlines()
        seen = set()
        out = []
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 3:
                continue
            # Deduplicate exact repeats (cookie banners often repeat)
            if stripped in seen:
                continue
            seen.add(stripped)
            out.append(stripped)
        return "\n".join(out)

    def _is_low_quality(self, text: str) -> bool:
        """Heuristic: reject cookie-oriented, repetitive, or stub output."""
        lowered = text.lower()
        # Cookie/consent dominance
        kw = self.config.readability_negative_keywords
        hits = sum(1 for k in kw if k in lowered)
        if hits >= 3:
            return True
        # Unusually short
        if len(text) < self.config.min_content_length:
            return True
        # Excessive repetition
        lines = text.splitlines()
        if lines:
            unique = len(set(l.strip() for l in lines if l.strip()))
            if unique / len(lines) < 0.3:
                return True
        return False
