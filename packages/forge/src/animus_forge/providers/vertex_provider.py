"""Google Vertex AI provider implementation."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    ProviderConfig,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderType,
    RateLimitError,
    StreamChunk,
)

try:
    import vertexai
    from google.api_core.exceptions import GoogleAPIError, ResourceExhausted
    from vertexai.generative_models import GenerationConfig, GenerativeModel
except ImportError:
    vertexai = None
    GenerativeModel = None
    GenerationConfig = None
    ResourceExhausted = Exception
    GoogleAPIError = Exception


class VertexProvider(Provider):
    """Google Vertex AI provider for Gemini and other models.

    Requires:
        - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
        - Or: gcloud auth application-default login
        - GOOGLE_CLOUD_PROJECT: GCP project ID
        - GOOGLE_CLOUD_LOCATION: Region (default: us-central1)
    """

    MODELS = [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
        "gemini-1.0-pro-vision",
    ]

    def __init__(
        self,
        config: ProviderConfig | None = None,
        project: str | None = None,
        location: str = "us-central1",
    ):
        """Initialize Vertex AI provider.

        Args:
            config: Provider configuration
            project: GCP project ID
            location: GCP region for Vertex AI
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.VERTEX,
                metadata={"project": project, "location": location},
            )
        super().__init__(config)
        self._model_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "vertex"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VERTEX

    def _get_fallback_model(self) -> str:
        return "gemini-1.5-flash"

    def is_configured(self) -> bool:
        """Check if Vertex AI is properly configured."""
        if not vertexai:
            return False
        try:
            import os

            from animus_forge.config.settings import get_settings

            settings = get_settings()
            # Check for credentials
            has_creds = bool(
                settings.google_application_credentials
                or os.path.exists(
                    os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
                )
            )
            has_project = bool(
                self.config.metadata.get("project")
                or settings.google_cloud_project
                or os.environ.get("GCLOUD_PROJECT")
            )
            return has_creds and has_project
        except Exception:
            return False

    def initialize(self) -> None:
        """Initialize Vertex AI SDK."""
        if not vertexai:
            raise ProviderNotConfiguredError("google-cloud-aiplatform not installed")

        import os

        from animus_forge.config.settings import get_settings

        settings = get_settings()
        project = (
            self.config.metadata.get("project")
            or settings.google_cloud_project
            or os.environ.get("GCLOUD_PROJECT")
        )
        location = (
            self.config.metadata.get("location") or settings.google_cloud_location or "us-central1"
        )

        if not project:
            raise ProviderNotConfiguredError("GCP project not configured")

        try:
            vertexai.init(project=project, location=location)
            self._initialized = True
        except Exception as e:
            raise ProviderNotConfiguredError(f"Failed to initialize Vertex AI: {e}")

    def _get_model(self, model_name: str) -> Any:
        """Get or create a model instance."""
        if model_name not in self._model_cache:
            self._model_cache[model_name] = GenerativeModel(model_name)
        return self._model_cache[model_name]

    def _build_contents(self, request: CompletionRequest) -> list[Any]:
        """Build contents list for Gemini API."""
        if request.messages:
            contents = []
            for msg in request.messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            return contents

        return [request.prompt]

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Vertex AI."""
        if not self._initialized:
            self.initialize()

        model_name = request.model or self.default_model
        model = self._get_model(model_name)

        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens or 8192,
            stop_sequences=request.stop_sequences or [],
        )

        contents = self._build_contents(request)

        start_time = time.time()
        try:
            if request.system_prompt:
                # Use system instruction for Gemini 1.5+
                model = GenerativeModel(model_name, system_instruction=request.system_prompt)

            response = model.generate_content(
                contents,
                generation_config=generation_config,
            )
        except ResourceExhausted as e:
            raise RateLimitError(str(e))
        except GoogleAPIError as e:
            raise ProviderError(f"Vertex AI API error: {e}")
        except Exception as e:
            raise ProviderError(f"Vertex AI error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        # Extract content
        content = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                content = candidate.content.parts[0].text

        # Extract usage
        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        output_tokens = (
            getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        )

        return CompletionResponse(
            content=content,
            model=model_name,
            provider=self.name,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=response.candidates[0].finish_reason.name
            if response.candidates
            else None,
            latency_ms=latency_ms,
        )

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion asynchronously."""
        if not self._initialized:
            self.initialize()

        model_name = request.model or self.default_model
        model = self._get_model(model_name)

        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens or 8192,
            stop_sequences=request.stop_sequences or [],
        )

        contents = self._build_contents(request)

        start_time = time.time()
        try:
            if request.system_prompt:
                model = GenerativeModel(model_name, system_instruction=request.system_prompt)

            response = await model.generate_content_async(
                contents,
                generation_config=generation_config,
            )
        except ResourceExhausted as e:
            raise RateLimitError(str(e))
        except GoogleAPIError as e:
            raise ProviderError(f"Vertex AI API error: {e}")
        except Exception as e:
            raise ProviderError(f"Vertex AI error: {e}")

        latency_ms = (time.time() - start_time) * 1000

        content = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                content = candidate.content.parts[0].text

        usage_metadata = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0
        output_tokens = (
            getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0
        )

        return CompletionResponse(
            content=content,
            model=model_name,
            provider=self.name,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=response.candidates[0].finish_reason.name
            if response.candidates
            else None,
            latency_ms=latency_ms,
        )

    @property
    def supports_streaming(self) -> bool:
        return True

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Generate a streaming completion."""
        if not self._initialized:
            self.initialize()

        model_name = request.model or self.default_model
        model = self._get_model(model_name)

        if request.system_prompt:
            model = GenerativeModel(model_name, system_instruction=request.system_prompt)

        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens or 8192,
            stop_sequences=request.stop_sequences or [],
        )

        contents = self._build_contents(request)

        try:
            response_stream = model.generate_content(
                contents,
                generation_config=generation_config,
                stream=True,
            )

            for chunk in response_stream:
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text = candidate.content.parts[0].text
                        if text:
                            yield StreamChunk(
                                content=text,
                                model=model_name,
                                provider=self.name,
                                is_final=False,
                            )

            yield StreamChunk(
                content="",
                model=model_name,
                provider=self.name,
                is_final=True,
                finish_reason="stop",
            )

        except ResourceExhausted as e:
            raise RateLimitError(str(e))
        except GoogleAPIError as e:
            raise ProviderError(f"Vertex AI streaming error: {e}")
        except Exception as e:
            raise ProviderError(f"Vertex AI streaming error: {e}")

    async def complete_stream_async(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Generate an async streaming completion."""
        if not self._initialized:
            self.initialize()

        model_name = request.model or self.default_model
        model = self._get_model(model_name)

        if request.system_prompt:
            model = GenerativeModel(model_name, system_instruction=request.system_prompt)

        generation_config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens or 8192,
            stop_sequences=request.stop_sequences or [],
        )

        contents = self._build_contents(request)

        try:
            response_stream = await model.generate_content_async(
                contents,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response_stream:
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text = candidate.content.parts[0].text
                        if text:
                            yield StreamChunk(
                                content=text,
                                model=model_name,
                                provider=self.name,
                                is_final=False,
                            )

            yield StreamChunk(
                content="",
                model=model_name,
                provider=self.name,
                is_final=True,
                finish_reason="stop",
            )

        except ResourceExhausted as e:
            raise RateLimitError(str(e))
        except GoogleAPIError as e:
            raise ProviderError(f"Vertex AI streaming error: {e}")
        except Exception as e:
            raise ProviderError(f"Vertex AI streaming error: {e}")

    def list_models(self) -> list[str]:
        """List available Vertex AI models."""
        return self.MODELS.copy()

    def get_model_info(self, model: str) -> dict:
        """Get information about a Vertex AI model."""
        model_info = {
            "gemini-2.0-flash-exp": {
                "context_window": 1048576,
                "description": "Next-gen fast model (experimental)",
            },
            "gemini-1.5-pro": {
                "context_window": 2097152,
                "description": "Most capable Gemini model",
            },
            "gemini-1.5-flash": {
                "context_window": 1048576,
                "description": "Fast and efficient",
            },
            "gemini-1.5-flash-8b": {
                "context_window": 1048576,
                "description": "Lightweight flash model",
            },
            "gemini-1.0-pro": {
                "context_window": 32760,
                "description": "Balanced performance",
            },
            "gemini-1.0-pro-vision": {
                "context_window": 16384,
                "description": "Multimodal capabilities",
            },
        }
        info = model_info.get(model, {})
        return {
            "model": model,
            "provider": self.name,
            **info,
        }
