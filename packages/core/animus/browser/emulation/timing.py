"""Human timing emulation.

Uses truncated Gaussian rejection sampling for delays, plus micro-jitter
and occasional multi-second "distraction" events.
"""

from __future__ import annotations

import random


class TimingEmulator:
    """Emulate human browsing rhythms."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Gaussian helpers
    # ------------------------------------------------------------------

    def _truncated_gaussian(self, mean: float, sigma: float, low: float, high: float) -> float:
        """Rejection-sample a Gaussian bounded by [low, high]."""
        for _ in range(100):
            val = self.rng.gauss(mean, sigma)
            if low <= val <= high:
                return val
        return mean

    def _jitter(self, base: float, pct: float = 0.10) -> float:
        """Add ±pct noise to a base delay."""
        return base * (1 + self.rng.uniform(-pct, pct))

    # ------------------------------------------------------------------
    # Public delays
    # ------------------------------------------------------------------

    async def initial_delay(self) -> None:
        """Pause before first interaction (page orientation)."""
        import asyncio

        delay = self._truncated_gaussian(0.4, 0.2, 0.1, 1.2)
        delay = self._jitter(delay)
        await asyncio.sleep(delay)

    async def scroll_pause(self) -> None:
        """Pause between scroll steps; position-dependent."""
        import asyncio

        # Slightly longer near top (reading headlines), shorter deep in page
        base = self._truncated_gaussian(0.35, 0.15, 0.1, 1.0)
        await asyncio.sleep(self._jitter(base))

    async def reading_delay(self) -> None:
        """Extended pause simulating reading a paragraph."""
        import asyncio

        base = self._truncated_gaussian(1.5, 0.5, 0.5, 4.0)
        await asyncio.sleep(self._jitter(base))

    async def thinking_delay(self) -> None:
        """Pause before clicking (decision latency)."""
        import asyncio

        base = self._truncated_gaussian(0.8, 0.3, 0.2, 2.5)
        await asyncio.sleep(self._jitter(base))

    async def micro_delay(self) -> None:
        """Tiny pause for keystrokes / fine motor actions."""
        import asyncio

        # ~10 % chance of a brief hesitation
        if self.rng.random() < 0.10:
            await asyncio.sleep(self._jitter(0.25))
        else:
            await asyncio.sleep(self._jitter(0.08, pct=0.20))

    async def final_delay(self) -> None:
        """Pause before closing tab or taking screenshot."""
        import asyncio

        base = self._truncated_gaussian(0.3, 0.1, 0.1, 0.8)
        await asyncio.sleep(self._jitter(base))

    async def maybe_distraction(self, prob: float = 0.03) -> None:
        """Rare multi-second interruption (tab switch, notification)."""
        import asyncio

        if self.rng.random() < prob:
            delay = self._truncated_gaussian(3.0, 1.0, 1.5, 8.0)
            await asyncio.sleep(self._jitter(delay))
