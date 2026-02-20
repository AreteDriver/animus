"""Ollama provider for local LLM inference."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import (
    CompletionRequest,
    CompletionResponse,
    ModelTier,
    Provider,
    ProviderConfig,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderType,
    RateLimitError,
    StreamChunk,
)

try:
    import httpx
except ImportError:
    httpx = None  # Optional import: httpx package not installed


# Ordered preference lists per tier — first available model wins
DEFAULT_TIER_MODELS: dict[str, list[str]] = {
    "reasoning": [
        "qwen2.5:72b",
        "deepseek-r1:70b",
        "llama3.1:70b",
        "qwen2.5:32b",
        "deepseek-r1:32b",
    ],
    "standard": [
        "qwen2.5:14b",
        "qwen2.5",
        "llama3.2",
        "mistral",
        "deepseek-r1:8b",
        "gemma2",
    ],
    "fast": [
        "qwen2.5:3b",
        "llama3.2:1b",
        "phi3",
    ],
    "embedding": [
        "nomic-embed-text",
        "all-minilm",
        "mxbai-embed-large",
    ],
}


class OllamaProvider(Provider):
    """Ollama provider for running local LLMs.

    Supports any model available through Ollama including:
    - Llama 3.2, Llama 3.1
    - Mistral, Mixtral
    - Qwen 2.5
    - DeepSeek
    - Phi-3
    - CodeLlama
    - And many more

    Requires Ollama to be running locally or on a specified host.
    Default endpoint: http://localhost:11434
    """

    # Common models available on Ollama
    MODELS = [
        "llama3.2",
        "llama3.2:1b",
        "llama3.1",
        "llama3.1:70b",
        "mistral",
        "mistral-nemo",
        "mixtral",
        "qwen2.5",
        "qwen2.5:72b",
        "qwen2.5-coder",
        "deepseek-r1:8b",
        "deepseek-r1:70b",
        "phi3",
        "phi3:medium",
        "codellama",
        "gemma2",
        "gemma2:27b",
    ]

    def __init__(
        self,
        config: ProviderConfig | None = None,
        host: str = "http://localhost:11434",
        model: str | None = None,
        tier_models: dict[str, list[str]] | None = None,
    ):
        """Initialize Ollama provider.

        Args:
            config: Provider configuration
            host: Ollama server URL
            model: Default model to use
            tier_models: Override default tier→model preference lists
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                base_url=host,
                default_model=model,
            )
        super().__init__(config)
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._tier_models = tier_models or DEFAULT_TIER_MODELS

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA

    def _get_fallback_model(self) -> str:
        return "llama3.2"

    @property
    def base_url(self) -> str:
        return self.config.base_url or "http://localhost:11434"

    def is_configured(self) -> bool:
        """Check if Ollama is available."""
        if not httpx:
            return False
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def initialize(self) -> None:
        """Initialize HTTP clients."""
        if not httpx:
            raise ProviderNotConfiguredError("httpx package not installed")

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.config.timeout,
        )
        self._async_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.config.timeout,
        )
        self._initialized = True

    def _build_request(self, request: CompletionRequest, stream: bool = False) -> dict:
        """Build request payload for Ollama API."""
        model = request.model or self.default_model

        # Build messages for chat endpoint
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        if request.messages:
            messages.extend(request.messages)
        else:
            messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        if request.stop_sequences:
            payload["options"]["stop"] = request.stop_sequences

        return payload

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Ollama."""
        if not self._initialized:
            self.initialize()

        if not self._client:
            raise ProviderNotConfiguredError("Ollama client not initialized")

        payload = self._build_request(request, stream=False)
        model = request.model or self.default_model

        start_time = time.time()
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e))
            raise ProviderError(f"Ollama API error: {e}")
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
            )
        except Exception as e:
            raise ProviderError(f"Ollama error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        # Extract content from response
        content = data.get("message", {}).get("content", "")

        # Extract token counts
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return CompletionResponse(
            content=content,
            model=data.get("model", model),
            provider=self.name,
            tokens_used=prompt_eval_count + eval_count,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            finish_reason="stop" if data.get("done") else None,
            latency_ms=latency_ms,
            metadata={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "eval_duration": data.get("eval_duration"),
            },
        )

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion asynchronously."""
        if not self._initialized:
            self.initialize()

        if not self._async_client:
            raise ProviderNotConfiguredError("Ollama async client not initialized")

        payload = self._build_request(request, stream=False)
        model = request.model or self.default_model

        start_time = time.time()
        try:
            response = await self._async_client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e))
            raise ProviderError(f"Ollama API error: {e}")
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
            )
        except Exception as e:
            raise ProviderError(f"Ollama error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        content = data.get("message", {}).get("content", "")
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return CompletionResponse(
            content=content,
            model=data.get("model", model),
            provider=self.name,
            tokens_used=prompt_eval_count + eval_count,
            input_tokens=prompt_eval_count,
            output_tokens=eval_count,
            finish_reason="stop" if data.get("done") else None,
            latency_ms=latency_ms,
            metadata={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "eval_duration": data.get("eval_duration"),
            },
        )

    @property
    def supports_streaming(self) -> bool:
        return True

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Generate a streaming completion."""
        if not self._initialized:
            self.initialize()

        if not self._client:
            raise ProviderNotConfiguredError("Ollama client not initialized")

        payload = self._build_request(request, stream=True)
        model = request.model or self.default_model

        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        is_done = data.get("done", False)

                        if content or is_done:
                            yield StreamChunk(
                                content=content,
                                model=data.get("model", model),
                                provider=self.name,
                                finish_reason="stop" if is_done else None,
                                is_final=is_done,
                                output_tokens=data.get("eval_count", 0) if is_done else 0,
                            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e))
            raise ProviderError(f"Ollama streaming error: {e}")
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
            )
        except Exception as e:
            raise ProviderError(f"Ollama streaming error: {e}")

    async def complete_stream_async(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Generate an async streaming completion."""
        if not self._initialized:
            self.initialize()

        if not self._async_client:
            raise ProviderNotConfiguredError("Ollama async client not initialized")

        payload = self._build_request(request, stream=True)
        model = request.model or self.default_model

        try:
            async with self._async_client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        is_done = data.get("done", False)

                        if content or is_done:
                            yield StreamChunk(
                                content=content,
                                model=data.get("model", model),
                                provider=self.name,
                                finish_reason="stop" if is_done else None,
                                is_final=is_done,
                                output_tokens=data.get("eval_count", 0) if is_done else 0,
                            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e))
            raise ProviderError(f"Ollama streaming error: {e}")
        except httpx.ConnectError:
            raise ProviderNotConfiguredError(
                f"Cannot connect to Ollama at {self.base_url}. Make sure Ollama is running."
            )
        except Exception as e:
            raise ProviderError(f"Ollama streaming error: {e}")

    def list_models(self) -> list[str]:
        """List models available on the Ollama server."""
        if not self._initialized:
            try:
                self.initialize()
            except Exception:
                return self.MODELS.copy()

        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return self.MODELS.copy()

    def pull_model(self, model: str) -> bool:
        """Pull a model from the Ollama library.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        if not self._initialized:
            self.initialize()

        try:
            response = self._client.post(
                "/api/pull",
                json={"name": model},
                timeout=None,  # Pulling can take a long time
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_model_info(self, model: str) -> dict:
        """Get information about an Ollama model."""
        if not self._initialized:
            try:
                self.initialize()
            except Exception:
                return {"model": model, "provider": self.name}

        try:
            response = self._client.post("/api/show", json={"name": model})
            if response.status_code == 200:
                data = response.json()
                return {
                    "model": model,
                    "provider": self.name,
                    "modelfile": data.get("modelfile"),
                    "parameters": data.get("parameters"),
                    "template": data.get("template"),
                    "details": data.get("details"),
                }
        except Exception:
            pass  # Non-critical fallback: model details optional, fall back to basic info

        return {"model": model, "provider": self.name}

    def health_check(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            if not self._client:
                client = httpx.Client(base_url=self.base_url, timeout=5.0)
                response = client.get("/api/tags")
                client.close()
            else:
                response = self._client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    @staticmethod
    def validate_output(response: CompletionResponse) -> bool:
        """Validate that an Ollama response is usable.

        Checks for empty content and missing token counts.

        Returns:
            True if the response is valid.
        """
        if not response.content or not response.content.strip():
            return False
        return True

    def select_model_for_tier(
        self, tier: ModelTier, available_models: list[str] | None = None
    ) -> str | None:
        """Select the best locally-available model for a capability tier.

        Walks the preference list for the tier and returns the first model
        that is actually pulled on the Ollama server.

        Args:
            tier: Capability tier to select for
            available_models: Pre-fetched model list (avoids extra API call)

        Returns:
            Model name or None if no matching model is available
        """
        candidates = self._tier_models.get(tier.value, [])
        if not candidates:
            return None

        if available_models is None:
            available_models = self.list_models()

        # Normalize: strip tag suffixes for comparison (e.g. "llama3.2:latest" matches "llama3.2")
        available_set = set(available_models)
        available_bases = {m.split(":")[0] for m in available_models}

        for model in candidates:
            if model in available_set:
                return model
            # Check without tag — "llama3.2" matches "llama3.2:latest"
            base = model.split(":")[0]
            if base in available_bases and ":" not in model:
                return model

        return None

    def supports_tier(self, tier: ModelTier) -> bool:
        """Check if any model is available for the given tier.

        Args:
            tier: Capability tier to check

        Returns:
            True if at least one model is available for the tier
        """
        return self.select_model_for_tier(tier) is not None
