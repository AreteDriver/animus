"""Human emulation layer for anti-detection.

Ports probabilistic scroll, timing, and mouse patterns from turbowebfetch
(MIT) into Animus-native async Python.  Each pattern is deterministic-
seedable for reproducible tests while still passing heuristic bot checks.
"""

from __future__ import annotations

from typing import Any

from animus.browser.emulation.mouse import MouseEmulator
from animus.browser.emulation.scroll import ScrollEmulator
from animus.browser.emulation.timing import TimingEmulator

__all__ = ["EmulationLayer", "ScrollEmulator", "TimingEmulator", "MouseEmulator"]


class EmulationLayer:
    """Composable human-behavior emulator.

    Runs a realistic browsing sequence: scroll → micro-pause → scroll
    with occasional overshoot and backtrack.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.scroll = ScrollEmulator(seed=seed)
        self.timing = TimingEmulator(seed=seed)
        self.mouse = MouseEmulator(seed=seed)

    async def run(self, tab: Any) -> None:
        """Execute a full emulation pass on *tab*."""
        await self.timing.initial_delay()
        await self.scroll.traverse_page(tab, self.timing)
        await self.mouse.wander(tab, self.timing)
        await self.timing.final_delay()
