"""Browser automation using Playwright.

Provides high-level browser control for automated web interactions.
Enables Clawdbot-style operation where Gorgon can interact with websites.
"""

from __future__ import annotations

import base64
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Try to import playwright
try:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        async_playwright,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None
    Playwright = None


class ActionType(str, Enum):
    """Types of browser actions."""

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    FILL = "fill"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    SCREENSHOT = "screenshot"
    PDF = "pdf"
    SCROLL = "scroll"
    WAIT = "wait"
    EVALUATE = "evaluate"
    EXTRACT = "extract"


@dataclass
class BrowserConfig:
    """Configuration for browser automation."""

    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_ms: int = 30000
    user_agent: str | None = None
    locale: str = "en-US"
    timezone: str = "America/New_York"
    geolocation: dict | None = None
    permissions: list[str] = field(default_factory=list)
    blocked_urls: list[str] = field(default_factory=list)
    screenshots_dir: str = str(Path(tempfile.gettempdir()) / "gorgon_screenshots")

    # Security settings
    disable_javascript: bool = False
    block_downloads: bool = True
    max_page_loads: int = 100  # Prevent infinite navigation loops


@dataclass
class PageAction:
    """A browser action to perform."""

    action: ActionType
    selector: str | None = None
    value: str | None = None
    options: dict = field(default_factory=dict)


@dataclass
class PageResult:
    """Result of a browser action."""

    success: bool
    action: ActionType
    data: Any = None
    error: str | None = None
    screenshot_path: str | None = None
    url: str | None = None
    title: str | None = None


class BrowserAutomation:
    """High-level browser automation using Playwright.

    This class provides safe, controlled browser automation with:
    - URL validation and blocking
    - Action timeout and limits
    - Screenshot capture
    - Content extraction
    - Form interaction

    Usage:
        async with BrowserAutomation() as browser:
            # Navigate to a page
            result = await browser.navigate("https://example.com")

            # Click a button
            result = await browser.click("button#submit")

            # Fill a form
            result = await browser.fill("input[name='email']", "user@example.com")

            # Take a screenshot
            result = await browser.screenshot()

            # Extract page content
            result = await browser.extract_content()
    """

    def __init__(self, config: BrowserConfig | None = None):
        """Initialize browser automation.

        Args:
            config: Browser configuration.
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is not installed. "
                "Install with: pip install playwright && playwright install chromium"
            )

        self.config = config or BrowserConfig()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._page_loads = 0

        # Create screenshots directory
        Path(self.config.screenshots_dir).mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> BrowserAutomation:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Start the browser."""
        if self._browser:
            return

        logger.info(f"Starting {self.config.browser_type} browser...")
        self._playwright = await async_playwright().start()

        # Get browser type
        if self.config.browser_type == "firefox":
            browser_type = self._playwright.firefox
        elif self.config.browser_type == "webkit":
            browser_type = self._playwright.webkit
        else:
            browser_type = self._playwright.chromium

        # Launch browser
        self._browser = await browser_type.launch(
            headless=self.config.headless,
        )

        # Create context with settings
        context_options = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            "locale": self.config.locale,
            "timezone_id": self.config.timezone,
            "java_script_enabled": not self.config.disable_javascript,
        }

        if self.config.user_agent:
            context_options["user_agent"] = self.config.user_agent

        if self.config.geolocation:
            context_options["geolocation"] = self.config.geolocation
            context_options["permissions"] = ["geolocation"]

        self._context = await self._browser.new_context(**context_options)

        # Set default timeout
        self._context.set_default_timeout(self.config.timeout_ms)

        # Block URLs if configured
        if self.config.blocked_urls:
            await self._context.route(
                lambda url: any(b in url for b in self.config.blocked_urls),
                lambda route: route.abort(),
            )

        # Create initial page
        self._page = await self._context.new_page()
        logger.info("Browser started")

    async def stop(self) -> None:
        """Stop the browser."""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Browser stopped")

    def _validate_url(self, url: str) -> bool:
        """Validate URL is safe to navigate to.

        Args:
            url: URL to validate.

        Returns:
            True if URL is safe.
        """
        try:
            parsed = urlparse(url)

            # Must have scheme
            if parsed.scheme not in ("http", "https"):
                logger.warning(f"Blocked non-HTTP URL: {url}")
                return False

            # Check blocked patterns
            for pattern in self.config.blocked_urls:
                if pattern in url:
                    logger.warning(f"Blocked URL matching pattern '{pattern}': {url}")
                    return False

            return True

        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return False

    async def navigate(self, url: str, wait_until: str = "load") -> PageResult:
        """Navigate to a URL.

        Args:
            url: URL to navigate to.
            wait_until: When to consider navigation done.
                        Options: "load", "domcontentloaded", "networkidle"

        Returns:
            PageResult with navigation outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.NAVIGATE,
                error="Browser not started",
            )

        # Check page load limit
        self._page_loads += 1
        if self._page_loads > self.config.max_page_loads:
            return PageResult(
                success=False,
                action=ActionType.NAVIGATE,
                error=f"Max page loads ({self.config.max_page_loads}) exceeded",
            )

        # Validate URL
        if not self._validate_url(url):
            return PageResult(
                success=False,
                action=ActionType.NAVIGATE,
                error=f"URL blocked or invalid: {url}",
            )

        try:
            response = await self._page.goto(url, wait_until=wait_until)

            return PageResult(
                success=True,
                action=ActionType.NAVIGATE,
                data={
                    "status": response.status if response else None,
                    "url": self._page.url,
                },
                url=self._page.url,
                title=await self._page.title(),
            )

        except Exception as e:
            logger.exception(f"Navigation failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.NAVIGATE,
                error=str(e),
            )

    async def click(
        self,
        selector: str,
        button: str = "left",
        click_count: int = 1,
    ) -> PageResult:
        """Click an element.

        Args:
            selector: CSS selector or XPath for the element.
            button: Mouse button (left, right, middle).
            click_count: Number of clicks.

        Returns:
            PageResult with click outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.CLICK,
                error="Browser not started",
            )

        try:
            await self._page.click(
                selector,
                button=button,
                click_count=click_count,
            )

            return PageResult(
                success=True,
                action=ActionType.CLICK,
                data={"selector": selector},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Click failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.CLICK,
                error=str(e),
            )

    async def fill(self, selector: str, value: str) -> PageResult:
        """Fill a form field.

        Args:
            selector: CSS selector for the input field.
            value: Value to fill.

        Returns:
            PageResult with fill outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.FILL,
                error="Browser not started",
            )

        try:
            await self._page.fill(selector, value)

            return PageResult(
                success=True,
                action=ActionType.FILL,
                data={"selector": selector, "filled": True},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Fill failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.FILL,
                error=str(e),
            )

    async def type_text(
        self,
        selector: str,
        text: str,
        delay: int = 50,
    ) -> PageResult:
        """Type text character by character (simulates real typing).

        Args:
            selector: CSS selector for the input field.
            text: Text to type.
            delay: Delay between keystrokes in ms.

        Returns:
            PageResult with type outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.TYPE,
                error="Browser not started",
            )

        try:
            await self._page.type(selector, text, delay=delay)

            return PageResult(
                success=True,
                action=ActionType.TYPE,
                data={"selector": selector, "typed": True},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Type failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.TYPE,
                error=str(e),
            )

    async def select(self, selector: str, value: str | list[str]) -> PageResult:
        """Select option(s) from a dropdown.

        Args:
            selector: CSS selector for the select element.
            value: Value or list of values to select.

        Returns:
            PageResult with select outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.SELECT,
                error="Browser not started",
            )

        try:
            values = [value] if isinstance(value, str) else value
            selected = await self._page.select_option(selector, values)

            return PageResult(
                success=True,
                action=ActionType.SELECT,
                data={"selector": selector, "selected": selected},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Select failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.SELECT,
                error=str(e),
            )

    async def check(self, selector: str) -> PageResult:
        """Check a checkbox.

        Args:
            selector: CSS selector for the checkbox.

        Returns:
            PageResult with check outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.CHECK,
                error="Browser not started",
            )

        try:
            await self._page.check(selector)

            return PageResult(
                success=True,
                action=ActionType.CHECK,
                data={"selector": selector, "checked": True},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Check failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.CHECK,
                error=str(e),
            )

    async def screenshot(
        self,
        path: str | None = None,
        full_page: bool = False,
        selector: str | None = None,
    ) -> PageResult:
        """Take a screenshot.

        Args:
            path: Path to save screenshot (auto-generated if not provided).
            full_page: Capture full scrollable page.
            selector: Capture only this element.

        Returns:
            PageResult with screenshot path.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.SCREENSHOT,
                error="Browser not started",
            )

        try:
            # Generate path if not provided
            if not path:
                import time

                timestamp = int(time.time() * 1000)
                path = f"{self.config.screenshots_dir}/screenshot_{timestamp}.png"

            # Take screenshot
            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    await element.screenshot(path=path)
                else:
                    return PageResult(
                        success=False,
                        action=ActionType.SCREENSHOT,
                        error=f"Element not found: {selector}",
                    )
            else:
                await self._page.screenshot(path=path, full_page=full_page)

            return PageResult(
                success=True,
                action=ActionType.SCREENSHOT,
                data={"path": path, "full_page": full_page},
                screenshot_path=path,
                url=self._page.url,
                title=await self._page.title(),
            )

        except Exception as e:
            logger.exception(f"Screenshot failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.SCREENSHOT,
                error=str(e),
            )

    async def screenshot_base64(self, full_page: bool = False) -> PageResult:
        """Take a screenshot and return as base64.

        Args:
            full_page: Capture full scrollable page.

        Returns:
            PageResult with base64 image data.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.SCREENSHOT,
                error="Browser not started",
            )

        try:
            screenshot_bytes = await self._page.screenshot(full_page=full_page)
            b64_data = base64.b64encode(screenshot_bytes).decode("utf-8")

            return PageResult(
                success=True,
                action=ActionType.SCREENSHOT,
                data={"base64": b64_data, "full_page": full_page},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Screenshot failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.SCREENSHOT,
                error=str(e),
            )

    async def scroll(
        self,
        direction: str = "down",
        amount: int = 500,
        selector: str | None = None,
    ) -> PageResult:
        """Scroll the page or element.

        Args:
            direction: Scroll direction (up, down, left, right).
            amount: Scroll amount in pixels.
            selector: Scroll within this element.

        Returns:
            PageResult with scroll outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.SCROLL,
                error="Browser not started",
            )

        try:
            scroll_x, scroll_y = 0, 0

            if direction == "down":
                scroll_y = amount
            elif direction == "up":
                scroll_y = -amount
            elif direction == "right":
                scroll_x = amount
            elif direction == "left":
                scroll_x = -amount

            if selector:
                await self._page.evaluate(
                    f"""
                    const el = document.querySelector('{selector}');
                    if (el) {{ el.scrollBy({scroll_x}, {scroll_y}); }}
                    """
                )
            else:
                await self._page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

            return PageResult(
                success=True,
                action=ActionType.SCROLL,
                data={"direction": direction, "amount": amount},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Scroll failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.SCROLL,
                error=str(e),
            )

    async def wait_for_selector(
        self,
        selector: str,
        state: str = "visible",
        timeout: int | None = None,
    ) -> PageResult:
        """Wait for an element.

        Args:
            selector: CSS selector to wait for.
            state: State to wait for (attached, detached, visible, hidden).
            timeout: Timeout in ms (uses default if not provided).

        Returns:
            PageResult with wait outcome.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.WAIT,
                error="Browser not started",
            )

        try:
            await self._page.wait_for_selector(
                selector,
                state=state,
                timeout=timeout,
            )

            return PageResult(
                success=True,
                action=ActionType.WAIT,
                data={"selector": selector, "state": state},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Wait failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.WAIT,
                error=str(e),
            )

    async def evaluate(self, script: str) -> PageResult:
        """Execute JavaScript in the page context.

        Args:
            script: JavaScript code to execute.

        Returns:
            PageResult with script result.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.EVALUATE,
                error="Browser not started",
            )

        try:
            result = await self._page.evaluate(script)

            return PageResult(
                success=True,
                action=ActionType.EVALUATE,
                data={"result": result},
                url=self._page.url,
            )

        except Exception as e:
            logger.exception(f"Evaluate failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.EVALUATE,
                error=str(e),
            )

    async def extract_content(
        self,
        selector: str | None = None,
        extract_links: bool = True,
        extract_tables: bool = True,
    ) -> PageResult:
        """Extract content from the page.

        Args:
            selector: Extract from this element only.
            extract_links: Include links in extraction.
            extract_tables: Include table data in extraction.

        Returns:
            PageResult with extracted content.
        """
        if not self._page:
            return PageResult(
                success=False,
                action=ActionType.EXTRACT,
                error="Browser not started",
            )

        try:
            # Get page info
            title = await self._page.title()
            url = self._page.url

            # Extract text content
            if selector:
                element = await self._page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                else:
                    text = ""
            else:
                text = await self._page.inner_text("body")

            result = {
                "title": title,
                "url": url,
                "text": text[:50000],  # Limit text size
            }

            # Extract links
            if extract_links:
                links = await self._page.evaluate(
                    """
                    () => Array.from(document.querySelectorAll('a[href]'))
                        .slice(0, 100)
                        .map(a => ({
                            text: a.innerText.trim().slice(0, 100),
                            href: a.href
                        }))
                    """
                )
                result["links"] = links

            # Extract tables
            if extract_tables:
                tables = await self._page.evaluate(
                    """
                    () => Array.from(document.querySelectorAll('table'))
                        .slice(0, 10)
                        .map(table => {
                            const rows = Array.from(table.querySelectorAll('tr')).slice(0, 50);
                            return rows.map(row =>
                                Array.from(row.querySelectorAll('td, th'))
                                    .map(cell => cell.innerText.trim().slice(0, 200))
                            );
                        })
                    """
                )
                result["tables"] = tables

            return PageResult(
                success=True,
                action=ActionType.EXTRACT,
                data=result,
                url=url,
                title=title,
            )

        except Exception as e:
            logger.exception(f"Extract failed: {e}")
            return PageResult(
                success=False,
                action=ActionType.EXTRACT,
                error=str(e),
            )

    async def get_cookies(self) -> PageResult:
        """Get all cookies.

        Returns:
            PageResult with cookies.
        """
        if not self._context:
            return PageResult(
                success=False,
                action=ActionType.EXTRACT,
                error="Browser not started",
            )

        try:
            cookies = await self._context.cookies()
            return PageResult(
                success=True,
                action=ActionType.EXTRACT,
                data={"cookies": cookies},
            )

        except Exception as e:
            return PageResult(
                success=False,
                action=ActionType.EXTRACT,
                error=str(e),
            )

    async def execute_actions(self, actions: list[PageAction]) -> list[PageResult]:
        """Execute a sequence of actions.

        Args:
            actions: List of PageAction to execute.

        Returns:
            List of PageResult for each action.
        """
        results = []

        for action in actions:
            if action.action == ActionType.NAVIGATE:
                result = await self.navigate(action.value or "")
            elif action.action == ActionType.CLICK:
                result = await self.click(action.selector or "")
            elif action.action == ActionType.FILL:
                result = await self.fill(action.selector or "", action.value or "")
            elif action.action == ActionType.TYPE:
                result = await self.type_text(action.selector or "", action.value or "")
            elif action.action == ActionType.SCREENSHOT:
                result = await self.screenshot()
            elif action.action == ActionType.SCROLL:
                result = await self.scroll(
                    action.options.get("direction", "down"),
                    action.options.get("amount", 500),
                )
            elif action.action == ActionType.WAIT:
                result = await self.wait_for_selector(action.selector or "")
            elif action.action == ActionType.EXTRACT:
                result = await self.extract_content(action.selector)
            else:
                result = PageResult(
                    success=False,
                    action=action.action,
                    error=f"Unknown action: {action.action}",
                )

            results.append(result)

            # Stop on failure if configured
            if not result.success and action.options.get("stop_on_error", True):
                break

        return results


def create_browser_automation(
    headless: bool = True,
    browser_type: str = "chromium",
    **kwargs: Any,
) -> BrowserAutomation:
    """Create a browser automation instance.

    Args:
        headless: Run in headless mode.
        browser_type: Browser to use (chromium, firefox, webkit).
        **kwargs: Additional BrowserConfig options.

    Returns:
        Configured BrowserAutomation instance.
    """
    config = BrowserConfig(
        headless=headless,
        browser_type=browser_type,
        **kwargs,
    )
    return BrowserAutomation(config)
