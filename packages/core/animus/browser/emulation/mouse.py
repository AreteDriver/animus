"""Mouse movement emulation (viewport-relative trajectories).

Lightweight — moves the cursor to random viewport positions with
cognitive pauses.  Helps trigger hover states and banner dismissal.
"""

from __future__ import annotations

import random
from typing import Any


class MouseEmulator:
    """Emulate organic mouse wandering."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    async def wander(self, tab: Any, timing: Any) -> None:
        """Move cursor to a few random viewport positions."""
        import asyncio

        moves = self.rng.randint(2, 5)
        for _ in range(moves):
            x = self.rng.randint(50, 900)
            y = self.rng.randint(100, 700)
            try:
                # nodriver doesn't expose a direct mouse_move; we use JS
                await tab.evaluate(
                    f"document.dispatchEvent(new MouseEvent('mousemove', "
                    f"{{clientX: {x}, clientY: {y}, bubbles: true}}))"
                )
            except Exception:
                pass
            await timing.micro_delay()
            # Occasional hover pause
            if self.rng.random() < 0.2:
                await asyncio.sleep(self.rng.uniform(0.3, 0.8))
