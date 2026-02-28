"""LLM backend protocol and implementations."""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from animus_bootstrap.gateway.cognitive_types import CognitiveResponse, ToolCall

logger = logging.getLogger(__name__)


class CognitiveBackend(Protocol):
    """Protocol that all LLM backends must satisfy."""

    async def generate_response(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text response from messages."""

    async def generate_structured(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> CognitiveResponse:
        """Generate a structured response with optional tool calls."""


class AnthropicBackend:
    """Anthropic Messages API backend."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._api_key = api_key
        self._model = model

    async def generate_response(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Call Anthropic Messages API and return the text response."""
        # Convert OpenAI-format messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append(
                {"role": msg["role"], "content": msg["content"]},
            )

        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(self.API_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()

        # Extract text from content blocks
        content_blocks = data.get("content", [])
        texts = [block["text"] for block in content_blocks if block.get("type") == "text"]
        return "\n".join(texts)

    async def generate_structured(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> CognitiveResponse:
        """Call Anthropic Messages API and return a structured response with tool calls."""
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append(
                {"role": msg["role"], "content": msg["content"]},
            )

        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(self.API_URL, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()

        # Parse content blocks
        content_blocks = data.get("content", [])
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if block.get("type") == "text":
                texts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block.get("input", {}),
                    )
                )

        return CognitiveResponse(
            text="\n".join(texts),
            tool_calls=tool_calls,
            stop_reason=data.get("stop_reason", "end_turn"),
            usage=data.get("usage", {}),
        )


class OllamaBackend:
    """Ollama local LLM backend."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")

    async def generate_response(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Call Ollama chat API and return the response text."""
        ollama_messages: list[dict] = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            ollama_messages.append({"role": msg["role"], "content": msg["content"]})

        payload = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._host}/api/chat",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()

        return data.get("message", {}).get("content", "")

    async def generate_structured(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> CognitiveResponse:
        """Ollama wrapper — calls generate_response and wraps in CognitiveResponse."""
        text = await self.generate_response(
            messages, system_prompt=system_prompt, max_tokens=max_tokens
        )
        return CognitiveResponse(text=text)


class ForgeBackend:
    """Animus Forge orchestration API backend."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str = "",
    ) -> None:
        self._base_url = f"http://{host}:{port}"
        self._api_key = api_key

    async def generate_response(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        """Call Forge API and return the response text."""
        payload: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt

        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/api/v1/chat",
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

        return data.get("response", data.get("text", ""))

    async def generate_structured(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> CognitiveResponse:
        """Forge wrapper — calls generate_response and wraps in CognitiveResponse."""
        text = await self.generate_response(
            messages, system_prompt=system_prompt, max_tokens=max_tokens
        )
        return CognitiveResponse(text=text)
