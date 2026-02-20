"""Tests for browser automation module.

Comprehensively tests BrowserAutomation with all Playwright dependencies mocked.
"""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_forge.browser.automation import (
    ActionType,
    BrowserAutomation,
    BrowserConfig,
    PageAction,
    PageResult,
    create_browser_automation,
)

# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestActionType:
    """Tests for ActionType enum."""

    def test_all_action_types_exist(self):
        assert ActionType.NAVIGATE == "navigate"
        assert ActionType.CLICK == "click"
        assert ActionType.TYPE == "type"
        assert ActionType.FILL == "fill"
        assert ActionType.SELECT == "select"
        assert ActionType.CHECK == "check"
        assert ActionType.UNCHECK == "uncheck"
        assert ActionType.SCREENSHOT == "screenshot"
        assert ActionType.PDF == "pdf"
        assert ActionType.SCROLL == "scroll"
        assert ActionType.WAIT == "wait"
        assert ActionType.EVALUATE == "evaluate"
        assert ActionType.EXTRACT == "extract"

    def test_is_str_enum(self):
        assert isinstance(ActionType.NAVIGATE, str)


class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_defaults(self):
        config = BrowserConfig()
        assert config.headless is True
        assert config.browser_type == "chromium"
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.timeout_ms == 30000
        assert config.user_agent is None
        assert config.locale == "en-US"
        assert config.timezone == "America/New_York"
        assert config.geolocation is None
        assert config.permissions == []
        assert config.blocked_urls == []
        assert config.disable_javascript is False
        assert config.block_downloads is True
        assert config.max_page_loads == 100

    def test_custom_config(self):
        config = BrowserConfig(
            headless=False,
            browser_type="firefox",
            viewport_width=1920,
            viewport_height=1080,
            timeout_ms=60000,
            user_agent="TestBot/1.0",
            blocked_urls=["ads.example.com"],
            max_page_loads=50,
        )
        assert config.headless is False
        assert config.browser_type == "firefox"
        assert config.viewport_width == 1920
        assert config.user_agent == "TestBot/1.0"
        assert config.blocked_urls == ["ads.example.com"]
        assert config.max_page_loads == 50


class TestPageAction:
    """Tests for PageAction dataclass."""

    def test_basic_action(self):
        action = PageAction(action=ActionType.CLICK, selector="#btn")
        assert action.action == ActionType.CLICK
        assert action.selector == "#btn"
        assert action.value is None
        assert action.options == {}

    def test_action_with_value(self):
        action = PageAction(
            action=ActionType.FILL,
            selector="input[name='email']",
            value="test@example.com",
        )
        assert action.value == "test@example.com"

    def test_action_with_options(self):
        action = PageAction(
            action=ActionType.SCROLL,
            options={"direction": "down", "amount": 300},
        )
        assert action.options["direction"] == "down"
        assert action.options["amount"] == 300


class TestPageResult:
    """Tests for PageResult dataclass."""

    def test_success_result(self):
        result = PageResult(
            success=True,
            action=ActionType.NAVIGATE,
            data={"status": 200},
            url="https://example.com",
            title="Example",
        )
        assert result.success is True
        assert result.action == ActionType.NAVIGATE
        assert result.data["status"] == 200
        assert result.error is None

    def test_error_result(self):
        result = PageResult(
            success=False,
            action=ActionType.CLICK,
            error="Element not found",
        )
        assert result.success is False
        assert result.error == "Element not found"
        assert result.data is None

    def test_screenshot_result(self):
        result = PageResult(
            success=True,
            action=ActionType.SCREENSHOT,
            screenshot_path="/tmp/screenshot.png",
        )
        assert result.screenshot_path == "/tmp/screenshot.png"


# ---------------------------------------------------------------------------
# BrowserAutomation tests (all Playwright mocked)
# ---------------------------------------------------------------------------


def _mock_playwright_available():
    """Patch PLAYWRIGHT_AVAILABLE to True."""
    return patch("animus_forge.browser.automation.PLAYWRIGHT_AVAILABLE", True)


def _make_browser(config=None, tmp_path=None):
    """Create a BrowserAutomation with mocked Playwright availability."""
    if config is None:
        config = BrowserConfig(
            screenshots_dir=str(tmp_path) if tmp_path else "/tmp/gorgon_test_screenshots"
        )
    with _mock_playwright_available(), patch("pathlib.Path.mkdir"):
        return BrowserAutomation(config)


def _setup_started_browser(browser):
    """Set internal state as if browser was started."""
    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.title = AsyncMock(return_value="Example Page")
    mock_page.screenshot = AsyncMock(return_value=b"fake_png_bytes")
    mock_page.evaluate = AsyncMock(return_value=None)
    mock_page.inner_text = AsyncMock(return_value="Page text content")

    mock_context = AsyncMock()
    mock_browser_obj = AsyncMock()
    mock_playwright = AsyncMock()

    browser._page = mock_page
    browser._context = mock_context
    browser._browser = mock_browser_obj
    browser._playwright = mock_playwright

    return mock_page, mock_context, mock_browser_obj, mock_playwright


class TestBrowserAutomationInit:
    """Tests for BrowserAutomation initialization."""

    def test_init_without_playwright_raises(self):
        with patch("animus_forge.browser.automation.PLAYWRIGHT_AVAILABLE", False):
            with pytest.raises(ImportError, match="Playwright is not installed"):
                BrowserAutomation()

    def test_init_with_default_config(self, tmp_path):
        browser = _make_browser(tmp_path=tmp_path)
        assert browser.config.headless is True
        assert browser._page is None
        assert browser._browser is None
        assert browser._page_loads == 0

    def test_init_with_custom_config(self, tmp_path):
        config = BrowserConfig(
            headless=False,
            browser_type="firefox",
            screenshots_dir=str(tmp_path),
        )
        browser = _make_browser(config=config)
        assert browser.config.headless is False
        assert browser.config.browser_type == "firefox"

    def test_init_creates_screenshots_dir(self, tmp_path):
        config = BrowserConfig(screenshots_dir=str(tmp_path / "screenshots"))
        with _mock_playwright_available():
            BrowserAutomation(config)
        assert (tmp_path / "screenshots").exists()


class TestBrowserAutomationStart:
    """Tests for browser start/stop lifecycle."""

    def test_start_chromium(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_pw_instance = AsyncMock()
            mock_browser_type = AsyncMock()
            mock_pw_instance.chromium = mock_browser_type
            mock_browser_type.launch = AsyncMock(return_value=AsyncMock())
            mock_context = AsyncMock()
            mock_browser_type.launch.return_value.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=AsyncMock())
            mock_context.set_default_timeout = MagicMock()

            mock_async_pw = AsyncMock()
            mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

            with patch(
                "animus_forge.browser.automation.async_playwright",
                return_value=mock_async_pw,
            ):
                await browser.start()
                assert browser._browser is not None
                assert browser._page is not None

        asyncio.run(_test())

    def test_start_firefox(self, tmp_path):
        async def _test():
            config = BrowserConfig(browser_type="firefox", screenshots_dir=str(tmp_path))
            browser = _make_browser(config=config)
            mock_pw_instance = AsyncMock()
            mock_browser_type = AsyncMock()
            mock_pw_instance.firefox = mock_browser_type
            mock_browser_type.launch = AsyncMock(return_value=AsyncMock())
            mock_context = AsyncMock()
            mock_browser_type.launch.return_value.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=AsyncMock())
            mock_context.set_default_timeout = MagicMock()

            mock_async_pw = AsyncMock()
            mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

            with patch(
                "animus_forge.browser.automation.async_playwright",
                return_value=mock_async_pw,
            ):
                await browser.start()

        asyncio.run(_test())

    def test_start_webkit(self, tmp_path):
        async def _test():
            config = BrowserConfig(browser_type="webkit", screenshots_dir=str(tmp_path))
            browser = _make_browser(config=config)
            mock_pw_instance = AsyncMock()
            mock_browser_type = AsyncMock()
            mock_pw_instance.webkit = mock_browser_type
            mock_browser_type.launch = AsyncMock(return_value=AsyncMock())
            mock_context = AsyncMock()
            mock_browser_type.launch.return_value.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=AsyncMock())
            mock_context.set_default_timeout = MagicMock()

            mock_async_pw = AsyncMock()
            mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

            with patch(
                "animus_forge.browser.automation.async_playwright",
                return_value=mock_async_pw,
            ):
                await browser.start()

        asyncio.run(_test())

    def test_start_noop_if_already_started(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            browser._browser = AsyncMock()  # pretend already started
            await browser.start()
            # Should return early without launching again

        asyncio.run(_test())

    def test_start_with_user_agent(self, tmp_path):
        async def _test():
            config = BrowserConfig(
                user_agent="CustomBot/1.0",
                screenshots_dir=str(tmp_path),
            )
            browser = _make_browser(config=config)
            mock_pw_instance = AsyncMock()
            mock_browser_type = AsyncMock()
            mock_pw_instance.chromium = mock_browser_type
            mock_launched = AsyncMock()
            mock_browser_type.launch = AsyncMock(return_value=mock_launched)
            mock_context = AsyncMock()
            mock_launched.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=AsyncMock())
            mock_context.set_default_timeout = MagicMock()

            mock_async_pw = AsyncMock()
            mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

            with patch(
                "animus_forge.browser.automation.async_playwright",
                return_value=mock_async_pw,
            ):
                await browser.start()
                call_kwargs = mock_launched.new_context.call_args[1]
                assert call_kwargs["user_agent"] == "CustomBot/1.0"

        asyncio.run(_test())

    def test_start_with_geolocation(self, tmp_path):
        async def _test():
            config = BrowserConfig(
                geolocation={"latitude": 40.7, "longitude": -74.0},
                screenshots_dir=str(tmp_path),
            )
            browser = _make_browser(config=config)
            mock_pw_instance = AsyncMock()
            mock_browser_type = AsyncMock()
            mock_pw_instance.chromium = mock_browser_type
            mock_launched = AsyncMock()
            mock_browser_type.launch = AsyncMock(return_value=mock_launched)
            mock_context = AsyncMock()
            mock_launched.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=AsyncMock())
            mock_context.set_default_timeout = MagicMock()

            mock_async_pw = AsyncMock()
            mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

            with patch(
                "animus_forge.browser.automation.async_playwright",
                return_value=mock_async_pw,
            ):
                await browser.start()
                call_kwargs = mock_launched.new_context.call_args[1]
                assert call_kwargs["geolocation"] == {
                    "latitude": 40.7,
                    "longitude": -74.0,
                }
                assert "geolocation" in call_kwargs["permissions"]

        asyncio.run(_test())

    def test_start_with_blocked_urls(self, tmp_path):
        async def _test():
            config = BrowserConfig(
                blocked_urls=["ads.example.com"],
                screenshots_dir=str(tmp_path),
            )
            browser = _make_browser(config=config)
            mock_pw_instance = AsyncMock()
            mock_browser_type = AsyncMock()
            mock_pw_instance.chromium = mock_browser_type
            mock_launched = AsyncMock()
            mock_browser_type.launch = AsyncMock(return_value=mock_launched)
            mock_context = AsyncMock()
            mock_launched.new_context = AsyncMock(return_value=mock_context)
            mock_context.new_page = AsyncMock(return_value=AsyncMock())
            mock_context.set_default_timeout = MagicMock()

            mock_async_pw = AsyncMock()
            mock_async_pw.start = AsyncMock(return_value=mock_pw_instance)

            with patch(
                "animus_forge.browser.automation.async_playwright",
                return_value=mock_async_pw,
            ):
                await browser.start()
                mock_context.route.assert_called_once()

        asyncio.run(_test())

    def test_stop(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, mock_context, mock_browser_obj, mock_playwright = _setup_started_browser(
                browser
            )

            await browser.stop()

            mock_context.close.assert_awaited_once()
            mock_browser_obj.close.assert_awaited_once()
            mock_playwright.stop.assert_awaited_once()
            assert browser._context is None
            assert browser._browser is None
            assert browser._playwright is None

        asyncio.run(_test())

    def test_stop_handles_none_state(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            # All are None
            await browser.stop()
            # Should not raise

        asyncio.run(_test())

    def test_async_context_manager(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            browser.start = AsyncMock()
            browser.stop = AsyncMock()

            async with browser as b:
                assert b is browser
                browser.start.assert_awaited_once()

            browser.stop.assert_awaited_once()

        asyncio.run(_test())


class TestURLValidation:
    """Tests for URL validation logic."""

    def test_valid_http_url(self, tmp_path):
        browser = _make_browser(tmp_path=tmp_path)
        assert browser._validate_url("http://example.com") is True

    def test_valid_https_url(self, tmp_path):
        browser = _make_browser(tmp_path=tmp_path)
        assert browser._validate_url("https://example.com") is True

    def test_blocked_ftp_url(self, tmp_path):
        browser = _make_browser(tmp_path=tmp_path)
        assert browser._validate_url("ftp://files.example.com") is False

    def test_blocked_file_url(self, tmp_path):
        browser = _make_browser(tmp_path=tmp_path)
        assert browser._validate_url("file:///etc/passwd") is False

    def test_blocked_javascript_url(self, tmp_path):
        browser = _make_browser(tmp_path=tmp_path)
        assert browser._validate_url("javascript:alert(1)") is False

    def test_blocked_url_pattern(self, tmp_path):
        config = BrowserConfig(
            blocked_urls=["evil.com", "malware.net"],
            screenshots_dir=str(tmp_path),
        )
        browser = _make_browser(config=config)
        assert browser._validate_url("https://evil.com/page") is False
        assert browser._validate_url("https://malware.net/download") is False
        assert browser._validate_url("https://safe.com") is True

    def test_invalid_url_returns_false(self, tmp_path):
        browser = _make_browser(tmp_path=tmp_path)
        # urlparse doesn't raise on bad strings, but the scheme check catches it
        assert browser._validate_url("") is False
        assert browser._validate_url("not-a-url") is False


class TestNavigate:
    """Tests for the navigate method."""

    def test_navigate_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.navigate("https://example.com")
            assert result.success is False
            assert result.error == "Browser not started"
            assert result.action == ActionType.NAVIGATE

        asyncio.run(_test())

    def test_navigate_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_page.goto = AsyncMock(return_value=mock_response)

            result = await browser.navigate("https://example.com")
            assert result.success is True
            assert result.data["status"] == 200
            assert result.url == "https://example.com"
            assert result.title == "Example Page"

        asyncio.run(_test())

    def test_navigate_blocked_url(self, tmp_path):
        async def _test():
            config = BrowserConfig(
                blocked_urls=["blocked.com"],
                screenshots_dir=str(tmp_path),
            )
            browser = _make_browser(config=config)
            _setup_started_browser(browser)

            result = await browser.navigate("https://blocked.com/page")
            assert result.success is False
            assert "blocked" in result.error.lower()

        asyncio.run(_test())

    def test_navigate_exceeds_max_page_loads(self, tmp_path):
        async def _test():
            config = BrowserConfig(max_page_loads=2, screenshots_dir=str(tmp_path))
            browser = _make_browser(config=config)
            mock_page, *_ = _setup_started_browser(browser)

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_page.goto = AsyncMock(return_value=mock_response)

            await browser.navigate("https://example.com/1")
            await browser.navigate("https://example.com/2")
            result = await browser.navigate("https://example.com/3")
            assert result.success is False
            assert "Max page loads" in result.error

        asyncio.run(_test())

    def test_navigate_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.goto = AsyncMock(side_effect=Exception("Network error"))

            result = await browser.navigate("https://example.com")
            assert result.success is False
            assert "Network error" in result.error

        asyncio.run(_test())

    def test_navigate_null_response(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.goto = AsyncMock(return_value=None)

            result = await browser.navigate("https://example.com")
            assert result.success is True
            assert result.data["status"] is None

        asyncio.run(_test())


class TestClick:
    """Tests for the click method."""

    def test_click_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.click("#btn")
            assert result.success is False
            assert result.error == "Browser not started"

        asyncio.run(_test())

    def test_click_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.click = AsyncMock()

            result = await browser.click("#submit")
            assert result.success is True
            assert result.data["selector"] == "#submit"
            mock_page.click.assert_awaited_once_with("#submit", button="left", click_count=1)

        asyncio.run(_test())

    def test_click_right_button(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.click = AsyncMock()

            result = await browser.click("#menu", button="right")
            assert result.success is True
            mock_page.click.assert_awaited_once_with("#menu", button="right", click_count=1)

        asyncio.run(_test())

    def test_click_double(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.click = AsyncMock()

            result = await browser.click("#item", click_count=2)
            assert result.success is True
            mock_page.click.assert_awaited_once_with("#item", button="left", click_count=2)

        asyncio.run(_test())

    def test_click_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.click = AsyncMock(side_effect=Exception("Element not found"))

            result = await browser.click("#missing")
            assert result.success is False
            assert "Element not found" in result.error

        asyncio.run(_test())


class TestFill:
    """Tests for the fill method."""

    def test_fill_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.fill("input", "value")
            assert result.success is False

        asyncio.run(_test())

    def test_fill_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.fill = AsyncMock()

            result = await browser.fill("input[name='email']", "user@test.com")
            assert result.success is True
            assert result.data["filled"] is True
            mock_page.fill.assert_awaited_once_with("input[name='email']", "user@test.com")

        asyncio.run(_test())

    def test_fill_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.fill = AsyncMock(side_effect=Exception("Input disabled"))

            result = await browser.fill("input", "value")
            assert result.success is False
            assert "Input disabled" in result.error

        asyncio.run(_test())


class TestTypeText:
    """Tests for the type_text method."""

    def test_type_text_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.type_text("input", "hello")
            assert result.success is False

        asyncio.run(_test())

    def test_type_text_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.type = AsyncMock()

            result = await browser.type_text("#search", "query", delay=100)
            assert result.success is True
            assert result.data["typed"] is True
            mock_page.type.assert_awaited_once_with("#search", "query", delay=100)

        asyncio.run(_test())

    def test_type_text_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.type = AsyncMock(side_effect=Exception("Timeout"))

            result = await browser.type_text("input", "text")
            assert result.success is False

        asyncio.run(_test())


class TestSelect:
    """Tests for the select method."""

    def test_select_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.select("select", "opt1")
            assert result.success is False

        asyncio.run(_test())

    def test_select_single_value(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.select_option = AsyncMock(return_value=["opt1"])

            result = await browser.select("select#color", "red")
            assert result.success is True
            assert result.data["selected"] == ["opt1"]
            mock_page.select_option.assert_awaited_once_with("select#color", ["red"])

        asyncio.run(_test())

    def test_select_multiple_values(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.select_option = AsyncMock(return_value=["a", "b"])

            result = await browser.select("select#multi", ["a", "b"])
            assert result.success is True
            mock_page.select_option.assert_awaited_once_with("select#multi", ["a", "b"])

        asyncio.run(_test())

    def test_select_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.select_option = AsyncMock(side_effect=Exception("Not a select"))

            result = await browser.select("div", "val")
            assert result.success is False

        asyncio.run(_test())


class TestCheck:
    """Tests for the check method."""

    def test_check_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.check("input[type=checkbox]")
            assert result.success is False

        asyncio.run(_test())

    def test_check_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.check = AsyncMock()

            result = await browser.check("#agree")
            assert result.success is True
            assert result.data["checked"] is True

        asyncio.run(_test())

    def test_check_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.check = AsyncMock(side_effect=Exception("Not checkable"))

            result = await browser.check("#broken")
            assert result.success is False

        asyncio.run(_test())


class TestScreenshot:
    """Tests for screenshot methods."""

    def test_screenshot_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.screenshot()
            assert result.success is False

        asyncio.run(_test())

    def test_screenshot_with_path(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.screenshot = AsyncMock()

            path = str(tmp_path / "test.png")
            result = await browser.screenshot(path=path)
            assert result.success is True
            assert result.screenshot_path == path
            mock_page.screenshot.assert_awaited_once_with(path=path, full_page=False)

        asyncio.run(_test())

    def test_screenshot_auto_path(self, tmp_path):
        async def _test():
            config = BrowserConfig(screenshots_dir=str(tmp_path))
            browser = _make_browser(config=config)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.screenshot = AsyncMock()

            result = await browser.screenshot()
            assert result.success is True
            assert result.screenshot_path is not None
            assert str(tmp_path) in result.screenshot_path

        asyncio.run(_test())

    def test_screenshot_full_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.screenshot = AsyncMock()

            path = str(tmp_path / "full.png")
            result = await browser.screenshot(path=path, full_page=True)
            assert result.success is True
            assert result.data["full_page"] is True
            mock_page.screenshot.assert_awaited_once_with(path=path, full_page=True)

        asyncio.run(_test())

    def test_screenshot_element(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_element = AsyncMock()
            mock_page.query_selector = AsyncMock(return_value=mock_element)

            path = str(tmp_path / "element.png")
            result = await browser.screenshot(path=path, selector="#header")
            assert result.success is True
            mock_element.screenshot.assert_awaited_once()

        asyncio.run(_test())

    def test_screenshot_element_not_found(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.query_selector = AsyncMock(return_value=None)

            path = str(tmp_path / "missing.png")
            result = await browser.screenshot(path=path, selector="#missing")
            assert result.success is False
            assert "Element not found" in result.error

        asyncio.run(_test())

    def test_screenshot_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.screenshot = AsyncMock(side_effect=Exception("Render error"))

            result = await browser.screenshot(path="/tmp/test.png")
            assert result.success is False
            assert "Render error" in result.error

        asyncio.run(_test())


class TestScreenshotBase64:
    """Tests for screenshot_base64 method."""

    def test_base64_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.screenshot_base64()
            assert result.success is False

        asyncio.run(_test())

    def test_base64_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            fake_bytes = b"fake_image_data"
            mock_page.screenshot = AsyncMock(return_value=fake_bytes)

            result = await browser.screenshot_base64()
            assert result.success is True
            expected_b64 = base64.b64encode(fake_bytes).decode("utf-8")
            assert result.data["base64"] == expected_b64

        asyncio.run(_test())

    def test_base64_full_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.screenshot = AsyncMock(return_value=b"data")

            result = await browser.screenshot_base64(full_page=True)
            assert result.success is True
            assert result.data["full_page"] is True

        asyncio.run(_test())

    def test_base64_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot error"))

            result = await browser.screenshot_base64()
            assert result.success is False

        asyncio.run(_test())


class TestScroll:
    """Tests for the scroll method."""

    def test_scroll_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.scroll()
            assert result.success is False

        asyncio.run(_test())

    def test_scroll_down(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)

            result = await browser.scroll(direction="down", amount=500)
            assert result.success is True
            assert result.data["direction"] == "down"
            mock_page.evaluate.assert_awaited_once()

        asyncio.run(_test())

    def test_scroll_up(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            _mock_page, *_ = _setup_started_browser(browser)

            result = await browser.scroll(direction="up", amount=300)
            assert result.success is True
            assert result.data["direction"] == "up"

        asyncio.run(_test())

    def test_scroll_left(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            _mock_page, *_ = _setup_started_browser(browser)

            result = await browser.scroll(direction="left", amount=200)
            assert result.success is True

        asyncio.run(_test())

    def test_scroll_right(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            _mock_page, *_ = _setup_started_browser(browser)

            result = await browser.scroll(direction="right", amount=200)
            assert result.success is True

        asyncio.run(_test())

    def test_scroll_within_element(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)

            result = await browser.scroll(selector="#scrollable")
            assert result.success is True
            # Should use element-specific scroll JS
            call_arg = mock_page.evaluate.call_args[0][0]
            assert "#scrollable" in call_arg

        asyncio.run(_test())

    def test_scroll_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.evaluate = AsyncMock(side_effect=Exception("JS error"))

            result = await browser.scroll()
            assert result.success is False

        asyncio.run(_test())


class TestWaitForSelector:
    """Tests for the wait_for_selector method."""

    def test_wait_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.wait_for_selector("#loading")
            assert result.success is False

        asyncio.run(_test())

    def test_wait_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.wait_for_selector = AsyncMock()

            result = await browser.wait_for_selector("#content", state="visible")
            assert result.success is True
            assert result.data["selector"] == "#content"
            assert result.data["state"] == "visible"

        asyncio.run(_test())

    def test_wait_with_timeout(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.wait_for_selector = AsyncMock()

            result = await browser.wait_for_selector("#el", timeout=5000)
            assert result.success is True
            mock_page.wait_for_selector.assert_awaited_once_with(
                "#el", state="visible", timeout=5000
            )

        asyncio.run(_test())

    def test_wait_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.wait_for_selector = AsyncMock(side_effect=Exception("Timeout exceeded"))

            result = await browser.wait_for_selector("#missing")
            assert result.success is False
            assert "Timeout exceeded" in result.error

        asyncio.run(_test())


class TestEvaluate:
    """Tests for the evaluate method."""

    def test_evaluate_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.evaluate("1 + 1")
            assert result.success is False

        asyncio.run(_test())

    def test_evaluate_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.evaluate = AsyncMock(return_value=42)

            result = await browser.evaluate("21 * 2")
            assert result.success is True
            assert result.data["result"] == 42

        asyncio.run(_test())

    def test_evaluate_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.evaluate = AsyncMock(side_effect=Exception("SyntaxError"))

            result = await browser.evaluate("invalid{")
            assert result.success is False

        asyncio.run(_test())


class TestExtractContent:
    """Tests for the extract_content method."""

    def test_extract_no_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.extract_content()
            assert result.success is False

        asyncio.run(_test())

    def test_extract_full_page(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.inner_text = AsyncMock(return_value="Page body text")
            mock_page.evaluate = AsyncMock(return_value=[])

            result = await browser.extract_content()
            assert result.success is True
            assert result.data["title"] == "Example Page"
            assert result.data["text"] == "Page body text"
            assert result.data["links"] == []
            assert result.data["tables"] == []

        asyncio.run(_test())

    def test_extract_with_selector(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_element = AsyncMock()
            mock_element.inner_text = AsyncMock(return_value="Section text")
            mock_page.query_selector = AsyncMock(return_value=mock_element)
            mock_page.evaluate = AsyncMock(return_value=[])

            result = await browser.extract_content(selector="#content")
            assert result.success is True
            assert result.data["text"] == "Section text"

        asyncio.run(_test())

    def test_extract_selector_not_found(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.query_selector = AsyncMock(return_value=None)
            mock_page.evaluate = AsyncMock(return_value=[])

            result = await browser.extract_content(selector="#missing")
            assert result.success is True
            assert result.data["text"] == ""

        asyncio.run(_test())

    def test_extract_without_links(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.inner_text = AsyncMock(return_value="text")
            mock_page.evaluate = AsyncMock(return_value=[])

            result = await browser.extract_content(extract_links=False)
            assert result.success is True
            assert "links" not in result.data

        asyncio.run(_test())

    def test_extract_without_tables(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.inner_text = AsyncMock(return_value="text")
            mock_page.evaluate = AsyncMock(return_value=[{"text": "link", "href": "/"}])

            result = await browser.extract_content(extract_tables=False)
            assert result.success is True
            assert "tables" not in result.data
            assert "links" in result.data

        asyncio.run(_test())

    def test_extract_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.title = AsyncMock(side_effect=Exception("Page crashed"))

            result = await browser.extract_content()
            assert result.success is False

        asyncio.run(_test())


class TestGetCookies:
    """Tests for the get_cookies method."""

    def test_cookies_no_context(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            result = await browser.get_cookies()
            assert result.success is False
            assert result.error == "Browser not started"

        asyncio.run(_test())

    def test_cookies_success(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            _setup_started_browser(browser)
            fake_cookies = [{"name": "session", "value": "abc123"}]
            browser._context.cookies = AsyncMock(return_value=fake_cookies)

            result = await browser.get_cookies()
            assert result.success is True
            assert result.data["cookies"] == fake_cookies

        asyncio.run(_test())

    def test_cookies_exception(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            _setup_started_browser(browser)
            browser._context.cookies = AsyncMock(side_effect=Exception("Cookie error"))

            result = await browser.get_cookies()
            assert result.success is False

        asyncio.run(_test())


class TestExecuteActions:
    """Tests for the execute_actions method."""

    def test_navigate_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_page.goto = AsyncMock(return_value=mock_response)

            actions = [
                PageAction(action=ActionType.NAVIGATE, value="https://example.com"),
            ]
            results = await browser.execute_actions(actions)
            assert len(results) == 1
            assert results[0].success is True

        asyncio.run(_test())

    def test_click_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.click = AsyncMock()

            actions = [PageAction(action=ActionType.CLICK, selector="#btn")]
            results = await browser.execute_actions(actions)
            assert len(results) == 1
            assert results[0].success is True

        asyncio.run(_test())

    def test_fill_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.fill = AsyncMock()

            actions = [
                PageAction(
                    action=ActionType.FILL,
                    selector="input",
                    value="hello",
                ),
            ]
            results = await browser.execute_actions(actions)
            assert len(results) == 1
            assert results[0].success is True

        asyncio.run(_test())

    def test_type_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.type = AsyncMock()

            actions = [
                PageAction(action=ActionType.TYPE, selector="input", value="text"),
            ]
            results = await browser.execute_actions(actions)
            assert results[0].success is True

        asyncio.run(_test())

    def test_screenshot_action(self, tmp_path):
        async def _test():
            config = BrowserConfig(screenshots_dir=str(tmp_path))
            browser = _make_browser(config=config)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.screenshot = AsyncMock()

            actions = [PageAction(action=ActionType.SCREENSHOT)]
            results = await browser.execute_actions(actions)
            assert results[0].success is True

        asyncio.run(_test())

    def test_scroll_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            _mock_page, *_ = _setup_started_browser(browser)

            actions = [
                PageAction(
                    action=ActionType.SCROLL,
                    options={"direction": "up", "amount": 300},
                ),
            ]
            results = await browser.execute_actions(actions)
            assert results[0].success is True

        asyncio.run(_test())

    def test_wait_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.wait_for_selector = AsyncMock()

            actions = [PageAction(action=ActionType.WAIT, selector="#el")]
            results = await browser.execute_actions(actions)
            assert results[0].success is True

        asyncio.run(_test())

    def test_extract_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.inner_text = AsyncMock(return_value="text")
            mock_page.evaluate = AsyncMock(return_value=[])

            actions = [PageAction(action=ActionType.EXTRACT)]
            results = await browser.execute_actions(actions)
            assert results[0].success is True

        asyncio.run(_test())

    def test_unknown_action(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            _setup_started_browser(browser)

            actions = [PageAction(action=ActionType.UNCHECK)]
            results = await browser.execute_actions(actions)
            assert len(results) == 1
            assert results[0].success is False
            assert "Unknown action" in results[0].error

        asyncio.run(_test())

    def test_stop_on_error_default(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.click = AsyncMock(side_effect=Exception("fail"))
            mock_page.fill = AsyncMock()

            actions = [
                PageAction(action=ActionType.CLICK, selector="#missing"),
                PageAction(action=ActionType.FILL, selector="input", value="val"),
            ]
            results = await browser.execute_actions(actions)
            # Should stop after the first failure
            assert len(results) == 1
            assert results[0].success is False

        asyncio.run(_test())

    def test_continue_on_error(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_page.click = AsyncMock(side_effect=Exception("fail"))
            mock_page.fill = AsyncMock()

            actions = [
                PageAction(
                    action=ActionType.CLICK,
                    selector="#missing",
                    options={"stop_on_error": False},
                ),
                PageAction(action=ActionType.FILL, selector="input", value="val"),
            ]
            results = await browser.execute_actions(actions)
            # Should continue past failure
            assert len(results) == 2
            assert results[0].success is False
            assert results[1].success is True

        asyncio.run(_test())

    def test_multiple_actions_sequence(self, tmp_path):
        async def _test():
            browser = _make_browser(tmp_path=tmp_path)
            mock_page, *_ = _setup_started_browser(browser)
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_page.goto = AsyncMock(return_value=mock_response)
            mock_page.fill = AsyncMock()
            mock_page.click = AsyncMock()

            actions = [
                PageAction(action=ActionType.NAVIGATE, value="https://example.com"),
                PageAction(action=ActionType.FILL, selector="#email", value="a@b.com"),
                PageAction(action=ActionType.CLICK, selector="#submit"),
            ]
            results = await browser.execute_actions(actions)
            assert len(results) == 3
            assert all(r.success for r in results)

        asyncio.run(_test())


class TestCreateBrowserAutomation:
    """Tests for the create_browser_automation factory function."""

    def test_default_creation(self):
        with _mock_playwright_available(), patch("pathlib.Path.mkdir"):
            browser = create_browser_automation()
            assert browser.config.headless is True
            assert browser.config.browser_type == "chromium"

    def test_custom_creation(self):
        with _mock_playwright_available(), patch("pathlib.Path.mkdir"):
            browser = create_browser_automation(
                headless=False,
                browser_type="firefox",
                viewport_width=1920,
            )
            assert browser.config.headless is False
            assert browser.config.browser_type == "firefox"
            assert browser.config.viewport_width == 1920

    def test_kwargs_passed_to_config(self):
        with _mock_playwright_available(), patch("pathlib.Path.mkdir"):
            browser = create_browser_automation(
                max_page_loads=5,
                disable_javascript=True,
            )
            assert browser.config.max_page_loads == 5
            assert browser.config.disable_javascript is True
