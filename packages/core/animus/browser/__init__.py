"""Browser automation bridge for Animus Research Guild and MCP tooling.

Owns real-browser page fetching with anti-detection emulation, content
extraction, and DurableObjectStore-backed caching.  Inspired by the
turbowebfetch concept (MIT), reimplemented as first-class Animus infra.

Architecture:
    BrowserBridge -- orchestrates nodriver lifecycle + anti-detection
    EmulationLayer -- scroll / timing / mouse humanisation
    ExtractionPipeline -- Readability → clean-text fallback
    BrowserCache -- DurableObjectStore integration for rendered pages
    rate_limit -- PolicyDecisionPoint-aware domain throttling
    mcp_tools -- fetch / fetch_batch MCP surface

Optional dependency: ``nodriver`` and ``readability-lxml``.  When
unavailable, imports succeed but :class:`BrowserBridge` raises
:exc:`RuntimeError` on instantiation with an install hint.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = [
    "BrowserBridge",
    "BrowserConfig",
    "BrowserResult",
    "EmulationLayer",
    "ExtractionPipeline",
    "BrowserCache",
    "RateLimitedDomain",
]

try:
    from animus.browser.bridge import BrowserBridge, BrowserConfig, BrowserResult
    from animus.browser.emulation import EmulationLayer
    from animus.browser.extraction import ExtractionPipeline
    from animus.browser.cache import BrowserCache
    from animus.browser.rate_limit import RateLimitedDomain

    _BROWSER_AVAILABLE = True
except ImportError:
    _BROWSER_AVAILABLE = False
