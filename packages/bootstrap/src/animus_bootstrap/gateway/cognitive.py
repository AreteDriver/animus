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
        temperature: float = 0.3,
        repeat_penalty: float = 1.2,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._temperature = temperature
        self._repeat_penalty = repeat_penalty

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
            "options": {
                "num_predict": max_tokens,
                "temperature": self._temperature,
                "repeat_penalty": self._repeat_penalty,
            },
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


class DualOllamaBackend:
    """Dual-model Ollama backend — conversation model + code specialist.

    Routes to the code model when the user message contains code-related
    keywords (write, debug, refactor, implement, function, class, etc.).
    Falls back to conversation model for everything else.
    """

    _CODE_KEYWORDS = frozenset(
        {
            "code",
            "function",
            "class",
            "debug",
            "refactor",
            "implement",
            "write",
            "fix",
            "bug",
            "error",
            "test",
            "compile",
            "build",
            "syntax",
            "import",
            "module",
            "deploy",
            "api",
            "endpoint",
            "database",
            "query",
            "sql",
            "schema",
            "migration",
            "dockerfile",
            "script",
            "lint",
            "type",
            "variable",
            "loop",
            "async",
            "await",
            "exception",
            "traceback",
            "stacktrace",
            "coverage",
            "pytest",
            "cargo",
            "rust",
            "python",
            "typescript",
            "javascript",
        }
    )

    def __init__(
        self,
        chat_model: str = "qwen2.5:14b",
        code_model: str = "deepseek-coder-v2",
        host: str = "http://localhost:11434",
        temperature: float = 0.3,
        repeat_penalty: float = 1.2,
    ) -> None:
        self._chat = OllamaBackend(
            model=chat_model,
            host=host,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )
        self._code = OllamaBackend(
            model=code_model,
            host=host,
            temperature=0.2,
            repeat_penalty=repeat_penalty,
        )
        self._chat_model = chat_model
        self._code_model = code_model

    def _pick_backend(self, messages: list[dict]) -> OllamaBackend:
        """Select backend based on message content."""
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "").lower()
                break

        words = set(last_user.split())
        if words & self._CODE_KEYWORDS:
            logger.info("Routing to code model: %s", self._code_model)
            return self._code
        logger.info("Routing to chat model: %s", self._chat_model)
        return self._chat

    async def generate_response(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        backend = self._pick_backend(messages)
        return await backend.generate_response(
            messages, system_prompt=system_prompt, max_tokens=max_tokens
        )

    async def generate_structured(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> CognitiveResponse:
        backend = self._pick_backend(messages)
        return await backend.generate_structured(
            messages, system_prompt=system_prompt, max_tokens=max_tokens, tools=tools
        )


class HybridBackend:
    """Hybrid backend — Anthropic for complex/meta queries, Ollama for casual chat.

    Falls back to Ollama if Anthropic key is missing or API call fails.
    """

    _COMPLEX_KEYWORDS = frozenset(
        {
            # Self-referential / meta (about Animus itself)
            "animus",
            "yourself",
            "self-improve",
            "self-improvement",
            "architecture",
            "persona",
            "identity",
            "cognitive",
            "backend",
            "gateway",
            "reflection",
            "proposal",
            "improvement",
            "evolve",
            "meta",
            "capabilities",
            "tools",
            "proactive",
            # Complex analysis
            "analyze",
            "analyse",
            "explain",
            "compare",
            "evaluate",
            "design",
            "architect",
            "tradeoff",
            "trade-off",
            "strategy",
            "recommend",
            "philosophy",
            "implications",
            "comprehensive",
            "detailed",
            "nuanced",
            "reasoning",
            "summarize",
            "critique",
            "review",
            "assess",
            "portfolio",
        }
    )

    def __init__(
        self,
        anthropic_backend: AnthropicBackend | None,
        ollama_backend: OllamaBackend | DualOllamaBackend,
    ) -> None:
        self._anthropic = anthropic_backend
        self._ollama = ollama_backend
        if anthropic_backend is None:
            logger.warning("HybridBackend: no Anthropic key, all queries route to ollama")

    def _classify_query(
        self, messages: list[dict]
    ) -> tuple[AnthropicBackend | OllamaBackend | DualOllamaBackend, str]:
        """Classify query complexity and pick the appropriate backend."""
        if self._anthropic is None:
            return self._ollama, "anthropic_unavailable"

        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "").lower()
                break

        words = set(last_user.split())
        matched = words & self._COMPLEX_KEYWORDS
        if matched:
            return (
                self._anthropic,
                f"matched keywords: {', '.join(sorted(matched))}",
            )

        # Long questions are often complex
        if "?" in last_user and len(last_user.split()) > 40:
            return self._anthropic, "long question (>40 words)"

        return self._ollama, "no complexity indicators"

    async def generate_response(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        backend, reason = self._classify_query(messages)
        backend_name = "anthropic" if backend is self._anthropic else "ollama"
        logger.info("HybridBackend routing to %s: %s", backend_name, reason)

        if backend is self._anthropic:
            try:
                return await backend.generate_response(
                    messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
            except Exception:
                logger.warning(
                    "HybridBackend: anthropic failed, falling back to ollama",
                    exc_info=True,
                )
                return await self._ollama.generate_response(
                    messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )

        return await self._ollama.generate_response(
            messages, system_prompt=system_prompt, max_tokens=max_tokens
        )

    async def generate_structured(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> CognitiveResponse:
        backend, reason = self._classify_query(messages)
        backend_name = "anthropic" if backend is self._anthropic else "ollama"
        logger.info(
            "HybridBackend structured routing to %s: %s",
            backend_name,
            reason,
        )

        if backend is self._anthropic:
            try:
                return await backend.generate_structured(
                    messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    tools=tools,
                )
            except Exception:
                logger.warning(
                    "HybridBackend: anthropic structured failed, falling back to ollama",
                    exc_info=True,
                )
                return await self._ollama.generate_structured(
                    messages,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    tools=tools,
                )

        return await self._ollama.generate_structured(
            messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            tools=tools,
        )


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
