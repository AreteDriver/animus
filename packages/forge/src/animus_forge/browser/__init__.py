"""Browser automation module using Playwright.

Provides browser control capabilities for Clawdbot-style operation:
- Navigate to URLs
- Click elements
- Type text
- Take screenshots
- Extract page content
- Fill forms

Requires:
    pip install playwright
    playwright install chromium
"""

from .automation import (
    BrowserAutomation,
    BrowserConfig,
    PageAction,
    PageResult,
    create_browser_automation,
)

__all__ = [
    "BrowserAutomation",
    "BrowserConfig",
    "PageAction",
    "PageResult",
    "create_browser_automation",
]
