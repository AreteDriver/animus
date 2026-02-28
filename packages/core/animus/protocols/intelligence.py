"""Protocol for language model backends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IntelligenceProvider(Protocol):
    """Structural interface for language model backends."""

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a text response from a prompt."""

    async def generate_stream(self, prompt: str, system: str | None = None) -> AsyncIterator[str]:
        """Generate a streaming text response."""

    def generate_with_tools(
        self,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
    ) -> Any:
        """Generate a response with tool-use support."""
