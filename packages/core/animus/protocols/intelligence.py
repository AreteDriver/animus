"""Protocol for language model backends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class IntelligenceProvider(Protocol):
    """Structural interface for language model backends."""

    def generate(self, prompt: str, system: str | None = None) -> str: ...
    async def generate_stream(
        self, prompt: str, system: str | None = None
    ) -> AsyncIterator[str]: ...
