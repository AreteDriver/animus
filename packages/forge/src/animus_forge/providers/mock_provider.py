"""Deterministic mock provider for CI evaluation runs.

Returns hash-based responses from prompts — no API calls needed.
Useful for testing the eval pipeline end-to-end without credentials.
"""

from __future__ import annotations

import hashlib

from .base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderConfig,
    ProviderType,
)


class MockProvider(Provider):
    """Provider that returns deterministic responses without API calls.

    Args:
        responses: Optional lookup table mapping prompt substrings to responses.
            If a prompt contains a key, the corresponding value is returned.
        config: Optional provider config (auto-created if not supplied).
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        config: ProviderConfig | None = None,
    ):
        super().__init__(
            config or ProviderConfig(provider_type=ProviderType.OPENAI, api_key="mock")
        )
        self._responses = responses or {}
        self._initialized = True

    @property
    def name(self) -> str:
        return "mock"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def _get_fallback_model(self) -> str:
        return "mock-model"

    def is_configured(self) -> bool:
        return True

    def initialize(self) -> None:
        pass

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Return a deterministic response based on the prompt.

        Checks the responses lookup table first, then falls back to
        a hash-based generated response.
        """
        prompt = request.prompt

        # Check lookup table
        for key, value in self._responses.items():
            if key in prompt:
                return self._make_response(value, prompt)

        # Hash-based deterministic response
        content = self._generate_from_hash(prompt)
        return self._make_response(content, prompt)

    def _generate_from_hash(self, prompt: str) -> str:
        """Generate a deterministic response seeded by the prompt hash."""
        h = hashlib.sha256(prompt.encode()).hexdigest()
        # Build a plausible multi-paragraph response
        words = [
            "analysis",
            "implementation",
            "testing",
            "review",
            "architecture",
            "security",
            "performance",
            "validation",
            "integration",
            "deployment",
            "monitoring",
            "documentation",
            "refactoring",
            "optimization",
            "verification",
            "assessment",
        ]
        # Use hash bytes to select words deterministically
        selected = [words[int(h[i : i + 2], 16) % len(words)] for i in range(0, 32, 2)]
        return (
            f"Based on the request, here is a detailed {selected[0]} plan:\n\n"
            f"1. First, perform {selected[1]} of the existing system.\n"
            f"2. Next, focus on {selected[2]} and {selected[3]}.\n"
            f"3. Ensure proper {selected[4]} throughout the process.\n"
            f"4. Validate with {selected[5]} and {selected[6]}.\n"
            f"5. Complete with {selected[7]} and final {selected[8]}.\n\n"
            f"Key considerations: {selected[9]}, {selected[10]}, "
            f"and {selected[11]}.\n\n"
            f"Additional steps: {selected[12]}, {selected[13]}, "
            f"{selected[14]}, {selected[15]}."
        )

    def _make_response(self, content: str, prompt: str) -> CompletionResponse:
        """Build a CompletionResponse with realistic token estimates."""
        input_tokens = len(prompt.split()) * 2
        output_tokens = len(content.split()) * 2
        return CompletionResponse(
            content=content,
            model="mock-model",
            provider="mock",
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="stop",
        )
