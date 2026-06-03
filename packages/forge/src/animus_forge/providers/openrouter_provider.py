"""OpenRouter provider implementation.

OpenRouter (https://openrouter.ai) is an OpenAI-compatible gateway to many
models behind a single key. This provider is the local-default / cloud-on-
consent inversion of the Hermes-Agent pattern: cloud reach is available, but
constrained by two provider-level guarantees on top of the shared egress
helper (``animus_forge.network``):

1. **PUBLIC-only egress.** Only ``Sensitivity.PUBLIC`` requests may leave the
   box through OpenRouter. PERSONAL/CONFIDENTIAL/SECRET are refused here,
   *before* the wire, regardless of env flags. This is stricter than the
   shared helper (which only forces PERSONAL local under ``ANIMUS_LOCAL_ONLY``)
   and is intentional: OpenRouter is a broker that fans out to many downstream
   providers, so the bar for what may reach it is the highest.

2. **Open-weights only.** Closed-vendor model slugs (``openai/*``,
   ``anthropic/*``, ``google/*`` …) are rejected at config-load and per
   request, so cloud behaviour stays open-weight end-to-end — consistent with
   the local Ollama default. This is a guarantee, not a default.

See ``docs/specs/openrouter-provider.md``.
"""

from __future__ import annotations

import os
import time
from typing import Any

from animus_types import Sensitivity

from animus_forge.utils.retry import async_with_retry, with_retry

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
)

try:
    from openai import AsyncOpenAI, OpenAI
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAI = None  # Optional import: openai package not installed
    AsyncOpenAI = None
    OpenAIRateLimitError = Exception

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Conservative denylist of known closed-weight vendor namespaces on OpenRouter.
# Open-weights-only is enforced by rejecting these prefixes. A denylist (not an
# allowlist) is used because the open-weight catalogue is large and churns; the
# closed-vendor set is small and stable. Override via ``config.metadata``
# ``closed_vendor_prefixes`` if the catalogue shifts.
CLOSED_VENDOR_PREFIXES: tuple[str, ...] = (
    "openai/",
    "anthropic/",
    "google/",
    "x-ai/",
    "cohere/",
    "perplexity/",
)

# Default open-weight slugs per capability tier. Env-overridable; exact slugs
# should be confirmed against the live OpenRouter catalogue. These are
# long-lived open-weight models chosen for stability.
DEFAULT_TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.REASONING: "deepseek/deepseek-r1",
    ModelTier.STANDARD: "meta-llama/llama-3.1-70b-instruct",
    ModelTier.FAST: "meta-llama/llama-3.1-8b-instruct",
}

# Env var overrides per tier.
_TIER_ENV: dict[ModelTier, str] = {
    ModelTier.REASONING: "ANIMUS_OPENROUTER_MODEL_REASONING",
    ModelTier.STANDARD: "ANIMUS_OPENROUTER_MODEL_STANDARD",
    ModelTier.FAST: "ANIMUS_OPENROUTER_MODEL_FAST",
}


class OpenWeightViolationError(ProviderError):
    """Raised when a closed-vendor model slug is used with OpenRouter."""


class OpenRouterProvider(Provider):
    """OpenRouter gateway provider (OpenAI-compatible, open-weights only)."""

    def __init__(self, config: ProviderConfig | None = None, api_key: str | None = None):
        """Initialize OpenRouter provider.

        Args:
            config: Provider configuration.
            api_key: API key (alternative to config).
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OPENROUTER,
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
            )
        if not config.base_url:
            config.base_url = OPENROUTER_BASE_URL
        super().__init__(config)
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENROUTER

    def _get_fallback_model(self) -> str:
        return DEFAULT_TIER_MODELS[ModelTier.STANDARD]

    @property
    def _closed_vendor_prefixes(self) -> tuple[str, ...]:
        override = self.config.metadata.get("closed_vendor_prefixes")
        if override:
            return tuple(override)
        return CLOSED_VENDOR_PREFIXES

    def is_configured(self) -> bool:
        """Check if OpenRouter API key is available."""
        if not OpenAI:
            return False
        return bool(self.config.api_key)

    # ------------------------------------------------------------------ #
    # Open-weights guarantee                                             #
    # ------------------------------------------------------------------ #
    def _assert_open_weight(self, model: str) -> None:
        """Reject closed-vendor slugs before any network call."""
        slug = model.lower().strip()
        for prefix in self._closed_vendor_prefixes:
            if slug.startswith(prefix):
                raise OpenWeightViolationError(
                    f"Model '{model}' is a closed-vendor slug; OpenRouter provider "
                    "is restricted to open-weight models. Use an open-weight slug "
                    f"or override via env (e.g. {_TIER_ENV[ModelTier.STANDARD]})."
                )

    def _resolve_model(self, request: CompletionRequest) -> str:
        """Resolve the concrete model slug for a request and assert it is open-weight.

        Precedence: explicit ``request.model`` > env override for the request
        tier > default for the tier > provider default model.
        """
        model = request.model
        if not model and request.model_tier is not None:
            env_var = _TIER_ENV.get(request.model_tier)
            if env_var and os.environ.get(env_var):
                model = os.environ[env_var]
            elif request.model_tier in DEFAULT_TIER_MODELS:
                model = DEFAULT_TIER_MODELS[request.model_tier]
        if not model:
            model = self.default_model
        self._assert_open_weight(model)
        return model

    # ------------------------------------------------------------------ #
    # Egress gate — PUBLIC-only                                          #
    # ------------------------------------------------------------------ #
    def _check_request_egress(self, request: CompletionRequest) -> None:
        """Refuse any non-PUBLIC request before the cloud client is invoked.

        Stricter than the shared helper: OpenRouter only ever sees PUBLIC
        content. PERSONAL/CONFIDENTIAL/SECRET fail closed here regardless of
        env flags. The shared ``is_egress_allowed`` is also consulted so
        ``ANIMUS_OFFLINE`` and loopback rules still apply.
        """
        from animus_forge.network import EgressDeniedError, is_egress_allowed

        endpoint = getattr(self, "_egress_endpoint", OPENROUTER_BASE_URL)

        if request.sensitivity is not Sensitivity.PUBLIC:
            raise EgressDeniedError(
                f"Request with sensitivity={request.sensitivity.value} blocked from "
                f"OpenRouter. Only PUBLIC content may egress to the model gateway; "
                "route this through a local provider (Ollama)."
            )
        if not is_egress_allowed(endpoint, sensitivity=request.sensitivity):
            raise EgressDeniedError(
                f"Egress to {endpoint} blocked by policy (offline mode). "
                "Route this through a local provider."
            )

    def initialize(self) -> None:
        """Initialize the OpenRouter client (OpenAI-compatible)."""
        if not OpenAI:
            raise ProviderNotConfiguredError("openai package not installed")

        if not self.config.api_key:
            try:
                from animus_forge.config import get_settings

                self.config.api_key = get_settings().openrouter_api_key
            except Exception:
                pass  # Non-critical fallback: settings unavailable, check api_key below

        if not self.config.api_key:
            raise ProviderNotConfiguredError("OpenRouter API key not configured")

        # Block client construction when offline.
        from animus_forge.network import EgressDeniedError, is_egress_allowed

        endpoint = self.config.base_url or OPENROUTER_BASE_URL
        if not is_egress_allowed(endpoint):
            raise EgressDeniedError(
                f"ANIMUS_OFFLINE=1 — OpenRouter provider blocked from {endpoint}. "
                "Unset the env var or route through a local provider."
            )
        self._egress_endpoint = endpoint

        # OpenRouter ranking headers — static Animus identifier.
        default_headers = {
            "HTTP-Referer": "https://github.com/AreteDriver/animus",
            "X-Title": "Animus",
        }

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=endpoint,
            timeout=self.config.timeout,
            default_headers=default_headers,
        )
        if AsyncOpenAI:
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=endpoint,
                timeout=self.config.timeout,
                default_headers=default_headers,
            )
        self._initialized = True

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using OpenRouter."""
        if not self._initialized:
            self.initialize()
        if not self._client:
            raise ProviderNotConfiguredError("OpenRouter client not initialized")

        self._check_request_egress(request)
        model = self._resolve_model(request)
        messages = self._build_messages(request)

        start_time = time.time()
        try:
            response = self._call_api(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
                extra_body=self._reasoning_extra_body(request),
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"OpenRouter API error: {e}")

        latency_ms = (time.time() - start_time) * 1000
        return self._to_response(response, latency_ms)

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion asynchronously using OpenRouter."""
        if not self._initialized:
            self.initialize()
        if not self._async_client:
            raise ProviderNotConfiguredError("OpenRouter async client not initialized")

        self._check_request_egress(request)
        model = self._resolve_model(request)
        messages = self._build_messages(request)

        start_time = time.time()
        try:
            response = await self._call_api_async(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
                extra_body=self._reasoning_extra_body(request),
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(str(e))
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"OpenRouter API error: {e}")

        latency_ms = (time.time() - start_time) * 1000
        return self._to_response(response, latency_ms)

    def _to_response(self, response: Any, latency_ms: float) -> CompletionResponse:
        """Map an OpenAI-compatible response to CompletionResponse."""
        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            provider=self.name,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
            latency_ms=latency_ms,
            metadata={"id": response.id},
        )

    def _reasoning_extra_body(self, request: CompletionRequest) -> dict[str, Any]:
        """Translate ``metadata['reasoning_effort']`` into OpenRouter's reasoning param.

        OpenRouter exposes a unified ``reasoning`` control; the OpenAI SDK passes
        non-standard params through ``extra_body``. Models like DeepSeek V4 Flash
        collapse in non-think mode on long-context tasks, so callers (e.g. Ogma
        synthesis) request high reasoning effort via the request metadata. Unknown
        or empty values are ignored (normal completion). Returns the ``extra_body``
        dict to merge, or ``{}`` when no valid effort is requested.
        """
        valid = {"minimal", "low", "medium", "high", "xhigh"}
        effort = request.metadata.get("reasoning_effort")
        if not effort:
            return {}
        effort = str(effort).strip().lower()
        if effort not in valid:
            return {}
        return {"reasoning": {"effort": effort}}

    def _build_messages(self, request: CompletionRequest) -> list[dict]:
        """Build message list for the API."""
        messages: list[dict] = []
        if request.messages:
            messages.extend(request.messages)
        else:
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
        return messages

    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _call_api(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        """Make API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        if extra_body:
            kwargs["extra_body"] = extra_body
        return self._client.chat.completions.create(**kwargs)

    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _call_api_async(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        """Make async API call with retry logic."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = stop
        if extra_body:
            kwargs["extra_body"] = extra_body
        return await self._async_client.chat.completions.create(**kwargs)

    def list_models(self) -> list[str]:
        """List the default open-weight tier models."""
        return list(dict.fromkeys(DEFAULT_TIER_MODELS.values()))
