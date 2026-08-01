"""Probabilistic scroll emulation.

Movement lengths draw from a bell-shaped distribution centered on a
configurable setpoint.  ~12 % chance of overshoot-and-backtrack.
Idle intervals are right-skewed to mirror natural reading pauses.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animus.browser.emulation.timing import TimingEmulator


class ScrollEmulator:
    """Emulate organic page scrolling."""

    def __init__(
        self,
        seed: int | None = None,
        mean_step_px: int = 400,
        step_sigma: int = 120,
        overshoot_prob: float = 0.12,
    ) -> None:
        self.rng = random.Random(seed)
        self.mean_step = mean_step_px
        self.sigma = step_sigma
        self.overshoot_prob = overshoot_prob

    def _gaussian_step(self) -> int:
        """Sample a scroll distance from a truncated Gaussian."""
        step = self.rng.gauss(self.mean_step, self.sigma)
        # Truncate to reasonable bounds
        step = max(100, min(step, self.mean_step * 2.5))
        return int(step)

    async def traverse_page(self, tab: Any, timing: TimingEmulator) -> None:
        """Scroll through ~90 % of page height with natural pauses."""

        # Determine total scrollable height
        height_js = "document.body.scrollHeight - window.innerHeight"
        try:
            total_height = await tab.evaluate(height_js)
        except Exception:
            total_height = 0

        if not total_height or total_height <= 0:
            return

        current = 0
        while current < total_height * 0.9:
            step = self._gaussian_step()

            # Occasional overshoot + backtrack
            if self.rng.random() < self.overshoot_prob:
                step = int(step * self.rng.uniform(1.1, 1.4))
                await tab.evaluate(f"window.scrollBy(0, {step})")
                await timing.scroll_pause()
                backtrack = int(step * self.rng.uniform(0.15, 0.35))
                await tab.evaluate(f"window.scrollBy(0, -{backtrack})")
                current += step - backtrack
            else:
                await tab.evaluate(f"window.scrollBy(0, {step})")
                current += step

            await timing.scroll_pause()

            # Occasional extended pause ("reading" or "distraction")
            if self.rng.random() < 0.08:
                await timing.reading_delay()

    async def scroll_to_element(self, tab: Any, selector: str, timing: TimingEmulator) -> None:
        """Smooth-scroll to an internal anchor with deceleration."""

        js = f"""
            const el = document.querySelector({repr(selector)});
            if (el) el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            return !!el;
        """
        found = await tab.evaluate(js)
        if found:
            await timing.micro_delay()
