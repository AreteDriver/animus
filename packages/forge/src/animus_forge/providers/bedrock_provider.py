"""AWS Bedrock provider implementation."""

from __future__ import annotations

import json
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
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception


class BedrockProvider(Provider):
    """AWS Bedrock provider for foundation models.

    Supports Claude, Llama, Mistral, and other models available on Bedrock.

    Requires AWS credentials configured via:
        - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        - AWS credentials file (~/.aws/credentials)
        - IAM role (when running on AWS)
    """

    # Model ID mapping for Bedrock
    MODELS = {
        # Anthropic Claude models
        "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        # Meta Llama models
        "llama-3-2-90b": "meta.llama3-2-90b-instruct-v1:0",
        "llama-3-2-11b": "meta.llama3-2-11b-instruct-v1:0",
        "llama-3-1-405b": "meta.llama3-1-405b-instruct-v1:0",
        "llama-3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
        "llama-3-1-8b": "meta.llama3-1-8b-instruct-v1:0",
        # Mistral models
        "mistral-large": "mistral.mistral-large-2407-v1:0",
        "mistral-small": "mistral.mistral-small-2402-v1:0",
        "mixtral-8x7b": "mistral.mixtral-8x7b-instruct-v0:1",
        # Amazon Titan
        "titan-text-premier": "amazon.titan-text-premier-v1:0",
        "titan-text-express": "amazon.titan-text-express-v1",
        # Cohere
        "cohere-command-r-plus": "cohere.command-r-plus-v1:0",
        "cohere-command-r": "cohere.command-r-v1:0",
    }

    def __init__(
        self,
        config: ProviderConfig | None = None,
        region: str = "us-east-1",
        profile: str | None = None,
    ):
        """Initialize AWS Bedrock provider.

        Args:
            config: Provider configuration
            region: AWS region for Bedrock
            profile: AWS profile name (optional)
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.BEDROCK,
                metadata={"region": region, "profile": profile},
            )
        super().__init__(config)
        self._client: Any | None = None
        self._runtime_client: Any | None = None

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.BEDROCK

    def _get_fallback_model(self) -> str:
        return "claude-3-5-sonnet"

    def is_configured(self) -> bool:
        """Check if AWS credentials are available."""
        if not boto3:
            return False
        try:
            # Try to get credentials
            session = boto3.Session(profile_name=self.config.metadata.get("profile"))
            credentials = session.get_credentials()
            return credentials is not None
        except Exception:
            return False

    def initialize(self) -> None:
        """Initialize Bedrock client."""
        if not boto3:
            raise ProviderNotConfiguredError("boto3 package not installed")

        region = self.config.metadata.get("region", "us-east-1")
        profile = self.config.metadata.get("profile")

        try:
            session = boto3.Session(profile_name=profile, region_name=region)
            self._client = session.client("bedrock")
            self._runtime_client = session.client("bedrock-runtime")
            self._initialized = True
        except Exception as e:
            raise ProviderNotConfiguredError(f"Failed to initialize Bedrock: {e}")

    def _resolve_model_id(self, model: str) -> str:
        """Resolve short model name to full Bedrock model ID."""
        return self.MODELS.get(model, model)

    def _get_model_family(self, model_id: str) -> str:
        """Determine model family from model ID."""
        if "anthropic" in model_id:
            return "anthropic"
        elif "meta" in model_id:
            return "meta"
        elif "mistral" in model_id:
            return "mistral"
        elif "amazon" in model_id:
            return "amazon"
        elif "cohere" in model_id:
            return "cohere"
        return "unknown"

    def _build_request_body(
        self,
        model_id: str,
        request: CompletionRequest,
    ) -> dict:
        """Build request body based on model family."""
        family = self._get_model_family(model_id)

        if family == "anthropic":
            messages = []
            if request.messages:
                messages = [m for m in request.messages if m.get("role") != "system"]
            else:
                messages = [{"role": "user", "content": request.prompt}]

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature,
            }
            if request.system_prompt:
                body["system"] = request.system_prompt
            if request.stop_sequences:
                body["stop_sequences"] = request.stop_sequences
            return body

        elif family == "meta":
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{request.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            return {
                "prompt": prompt,
                "max_gen_len": request.max_tokens or 2048,
                "temperature": request.temperature,
            }

        elif family == "mistral":
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            if request.messages:
                messages.extend(request.messages)
            else:
                messages.append({"role": "user", "content": request.prompt})
            return {
                "messages": messages,
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature,
            }

        elif family == "amazon":
            config = {"maxTokenCount": request.max_tokens or 4096}
            if request.stop_sequences:
                config["stopSequences"] = request.stop_sequences
            return {
                "inputText": request.prompt,
                "textGenerationConfig": config,
            }

        elif family == "cohere":
            return {
                "message": request.prompt,
                "max_tokens": request.max_tokens or 4096,
                "temperature": request.temperature,
                "preamble": request.system_prompt,
            }

        raise ProviderError(f"Unsupported model family: {family}")

    def _parse_response(self, model_id: str, response_body: dict) -> tuple[str, dict]:
        """Parse response based on model family."""
        family = self._get_model_family(model_id)

        if family == "anthropic":
            content = ""
            if response_body.get("content"):
                content = response_body["content"][0].get("text", "")
            usage = {
                "input_tokens": response_body.get("usage", {}).get("input_tokens", 0),
                "output_tokens": response_body.get("usage", {}).get("output_tokens", 0),
            }
            return content, usage

        elif family == "meta":
            return response_body.get("generation", ""), {}

        elif family == "mistral":
            choices = response_body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", ""), {}
            return "", {}

        elif family == "amazon":
            results = response_body.get("results", [])
            if results:
                return results[0].get("outputText", ""), {}
            return "", {}

        elif family == "cohere":
            return response_body.get("text", ""), {}

        return "", {}

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using Bedrock."""
        if not self._initialized:
            self.initialize()

        if not self._runtime_client:
            raise ProviderNotConfiguredError("Bedrock client not initialized")

        model = request.model or self.default_model
        model_id = self._resolve_model_id(model)
        body = self._build_request_body(model_id, request)

        start_time = time.time()
        try:
            response = self._runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read())
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ThrottlingException":
                raise RateLimitError(str(e))
            raise ProviderError(f"Bedrock API error: {e}")
        except Exception as e:
            raise ProviderError(f"Bedrock API error: {e}")

        latency_ms = (time.time() - start_time) * 1000
        content, usage = self._parse_response(model_id, response_body)

        return CompletionResponse(
            content=content,
            model=model_id,
            provider=self.name,
            tokens_used=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_ms=latency_ms,
            metadata={"model_id": model_id},
        )

    async def complete_async(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion asynchronously.

        Note: boto3 doesn't have native async support, so this wraps the sync call.
        For true async, consider using aioboto3.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.complete, request)

    @property
    def supports_streaming(self) -> bool:
        return True

    def complete_stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Generate a streaming completion."""
        if not self._initialized:
            self.initialize()

        if not self._runtime_client:
            raise ProviderNotConfiguredError("Bedrock client not initialized")

        model = request.model or self.default_model
        model_id = self._resolve_model_id(model)
        body = self._build_request_body(model_id, request)

        try:
            response = self._runtime_client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

            family = self._get_model_family(model_id)

            for event in response.get("body", []):
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk.get("bytes", b"{}").decode())
                    content = self._extract_stream_content(family, chunk_data)
                    if content:
                        yield StreamChunk(
                            content=content,
                            model=model_id,
                            provider=self.name,
                            is_final=False,
                        )

            # Final chunk
            yield StreamChunk(
                content="",
                model=model_id,
                provider=self.name,
                is_final=True,
                finish_reason="stop",
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ThrottlingException":
                raise RateLimitError(str(e))
            raise ProviderError(f"Bedrock streaming error: {e}")
        except Exception as e:
            raise ProviderError(f"Bedrock streaming error: {e}")

    def _extract_stream_content(self, family: str, chunk_data: dict) -> str:
        """Extract content from streaming chunk based on model family."""
        if family == "anthropic":
            if chunk_data.get("type") == "content_block_delta":
                return chunk_data.get("delta", {}).get("text", "")
        elif family == "meta":
            return chunk_data.get("generation", "")
        elif family == "mistral":
            choices = chunk_data.get("choices", [])
            if choices:
                return choices[0].get("delta", {}).get("content", "")
        elif family == "amazon":
            return chunk_data.get("outputText", "")
        elif family == "cohere":
            return chunk_data.get("text", "")
        return ""

    async def complete_stream_async(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Generate an async streaming completion.

        Note: Wraps sync streaming since boto3 lacks native async.
        """
        import asyncio

        loop = asyncio.get_event_loop()

        # Run sync generator in executor
        def sync_stream():
            return list(self.complete_stream(request))

        chunks = await loop.run_in_executor(None, sync_stream)
        for chunk in chunks:
            yield chunk

    def list_models(self) -> list[str]:
        """List available Bedrock models."""
        return list(self.MODELS.keys())

    def get_model_info(self, model: str) -> dict:
        """Get information about a Bedrock model."""
        model_id = self._resolve_model_id(model)
        family = self._get_model_family(model_id)
        return {
            "model": model,
            "provider": self.name,
            "model_id": model_id,
            "family": family,
            "description": f"{family.title()} model on AWS Bedrock",
        }
