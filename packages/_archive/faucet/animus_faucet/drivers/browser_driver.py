"""Playwright-based browser driver for JS-heavy faucets."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from animus_faucet.drivers.base import BaseDriver
from animus_faucet.models import ClaimResult, ClaimStatus

if TYPE_CHECKING:
    from animus_faucet.config import FaucetConfig
    from animus_faucet.models import FaucetDefinition

logger = logging.getLogger("animus_faucet.drivers.browser")


class BrowserDriver(BaseDriver):
    """Driver for faucets requiring full browser interaction.

    Uses Playwright for headless Chromium. Supports human-like mouse movement
    and timing jitter to avoid bot detection.

    Config expects `driver_config` with:
        - steps: list of action dicts describing the claim flow
        - address_selector: CSS selector for the wallet address input
        - submit_selector: CSS selector for the submit/claim button
        - success_selector: CSS selector that appears on success
        - amount_selector: optional CSS selector for payout amount text
    """

    def __init__(self, faucet: FaucetDefinition, config: FaucetConfig):
        super().__init__(faucet, config)
        dc = self.faucet.driver_config
        self.address_selector = dc.get("address_selector", "input[name='address']")
        self.submit_selector = dc.get("submit_selector", "button[type='submit']")
        self.success_selector = dc.get("success_selector")
        self.amount_selector = dc.get("amount_selector")
        self.steps = dc.get("steps", [])

    async def claim(self, wallet_address: str) -> ClaimResult:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return self._failure(
                ClaimStatus.UNKNOWN_ERROR,
                "playwright not installed — pip install animus-faucet[browser]",
            )

        proxy_url = self.config.proxy.proxy_url if self.faucet.requires_proxy else None

        try:
            async with async_playwright() as p:
                browser_args = {}
                if proxy_url:
                    browser_args["proxy"] = {"server": proxy_url}

                browser = await p.chromium.launch(headless=self.config.headless, **browser_args)
                context = await browser.new_context(
                    user_agent=self.config.user_agent,
                    viewport={"width": 1920, "height": 1080},
                )
                page = await context.new_page()

                await page.goto(self.faucet.url, wait_until="networkidle")
                await _human_delay(1.0, 3.0)

                if self.steps:
                    result = await self._execute_steps(page, wallet_address)
                else:
                    result = await self._default_flow(page, wallet_address)

                await browser.close()
                return result

        except Exception as e:
            logger.exception(f"Browser driver error for {self.faucet.name}")
            return self._failure(ClaimStatus.UNKNOWN_ERROR, str(e))

    async def _default_flow(self, page, wallet_address: str) -> ClaimResult:
        """Standard flow: fill address, click submit, check result."""
        # Type address with human-like delays
        address_el = page.locator(self.address_selector)
        await address_el.click()
        await _human_delay(0.3, 0.8)

        for char in wallet_address:
            await address_el.press(char)
            await _human_delay(0.02, 0.08)

        await _human_delay(0.5, 1.5)

        # Click submit
        submit_el = page.locator(self.submit_selector)
        await _human_move_and_click(page, submit_el)

        # Wait for result
        await _human_delay(2.0, 5.0)

        if self.success_selector:
            try:
                await page.wait_for_selector(self.success_selector, timeout=15000)
            except Exception:
                return self._failure(
                    ClaimStatus.UNKNOWN_ERROR, "Success selector not found after submit"
                )

        amount = None
        if self.amount_selector:
            try:
                amount_text = await page.locator(self.amount_selector).text_content()
                if amount_text:
                    amount = _parse_amount(amount_text)
            except Exception:
                pass

        return self._success(amount=amount)

    async def _execute_steps(self, page, wallet_address: str) -> ClaimResult:
        """Execute a list of configured step actions."""
        for step in self.steps:
            action = step.get("action", "")
            selector = step.get("selector", "")
            value = step.get("value", "").replace("{address}", wallet_address)

            if action == "fill":
                el = page.locator(selector)
                await el.click()
                await _human_delay(0.2, 0.5)
                await el.fill(value)
            elif action == "click":
                el = page.locator(selector)
                await _human_move_and_click(page, el)
            elif action == "wait":
                timeout = step.get("timeout", 5000)
                await page.wait_for_selector(selector, timeout=timeout)
            elif action == "delay":
                seconds = step.get("seconds", 1.0)
                await _human_delay(seconds * 0.8, seconds * 1.2)

            await _human_delay(0.3, 1.0)

        return self._success()

    async def health_check(self) -> bool:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return False

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                resp = await page.goto(self.faucet.url, timeout=15000)
                ok = resp is not None and resp.status < 500
                await browser.close()
                return ok
        except Exception:
            return False


async def _human_delay(min_s: float, max_s: float) -> None:
    """Sleep for a random duration to mimic human timing."""
    await asyncio.sleep(random.uniform(min_s, max_s))


async def _human_move_and_click(page, locator) -> None:
    """Move mouse in a natural curve then click."""
    box = await locator.bounding_box()
    if box:
        # Target a random point within the element
        x = box["x"] + random.uniform(box["width"] * 0.2, box["width"] * 0.8)
        y = box["y"] + random.uniform(box["height"] * 0.2, box["height"] * 0.8)
        await page.mouse.move(x, y, steps=random.randint(5, 15))
        await _human_delay(0.05, 0.2)
        await page.mouse.click(x, y)
    else:
        await locator.click()


def _parse_amount(text: str) -> float | None:
    """Extract a numeric amount from faucet success text."""
    import re

    match = re.search(r"[\d.]+", text.replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None
