"""Tool-call validation and retry for Animus Head.

Validates model-generated tool_calls against registered schemas, handling
both JSON (OpenAI/Anthropic/Ollama) and XML (Hermes) formats. When a call
is invalid, the validator informs the model and triggers a retry.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from animus_kernel.contracts.base import ContractViolation
from animus_kernel.tools.registry import ForgeToolRegistry
from animus_kernel.tools.schema_validator import (
    parse_hermes_tool_call,
    parse_json_tool_call,
    validate_tool_call,
)

logger = logging.getLogger(__name__)


class ToolCallResult:
    """Result of a single tool call attempt."""

    def __init__(
        self,
        tool_name: str = "",
        arguments: dict | None = None,
        valid: bool = False,
        error: str = "",
        raw: Any = None,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.valid = valid
        self.error = error
        self.raw = raw


class HeadToolValidator:
    """Validates and normalises tool calls from model responses.

    Supports:
    - Ollama/OpenAI/Anthropic JSON tool_call format
    - Hermes XML tool_call format (fallback)
    """

    def __init__(self, registry: ForgeToolRegistry | None = None) -> None:
        self._registry = registry

    def extract_tool_calls(self, response_text: str) -> list[ToolCallResult]:
        """Extract tool calls from raw model response text.

        Handles:
        1. Ollama native tool_calls (already parsed by provider)
        2. JSON object with function call
        3. Hermes XML <tool_call> blocks

        Returns:
            List of ToolCallResult (may be empty if no calls found).
        """
        results: list[ToolCallResult] = []

        # 1. Try JSON object(s) in the response text
        if response_text.strip().startswith("{"):
            try:
                data = json.loads(response_text.strip())
                # Single tool call
                if "name" in data or "function" in data:
                    name, args = parse_json_tool_call(data)
                    results.append(
                        ToolCallResult(tool_name=name, arguments=args, raw=data)
                    )
                # Multiple tool calls
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            name, args = parse_json_tool_call(item)
                            results.append(
                                ToolCallResult(tool_name=name, arguments=args, raw=item)
                            )
            except json.JSONDecodeError:
                pass

        # 2. Try Hermes XML
        if "<tool_call>" in response_text:
            try:
                name, args = parse_hermes_tool_call(response_text)
                if name:
                    results.append(
                        ToolCallResult(tool_name=name, arguments=args, raw=response_text)
                    )
            except ContractViolation:
                pass

        return results

    def validate(self, tool_name: str, arguments: dict) -> ToolCallResult:
        """Validate a tool call against the registry schema.

        Args:
            tool_name: Tool name.
            arguments: Parsed arguments dict.

        Returns:
            ToolCallResult with valid=True if passes, or valid=False with error.
        """
        if self._registry is None:
            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                valid=True,  # No registry to validate against
            )

        try:
            validate_tool_call(tool_name, arguments, self._registry)
            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                valid=True,
            )
        except ContractViolation as exc:
            return ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                valid=False,
                error=f"{exc.field}: {exc.message}",
            )

    def build_retry_prompt(self, invalid_calls: list[ToolCallResult]) -> str:
        """Build a prompt that informs the model about invalid tool calls.

        Args:
            invalid_calls: List of failed ToolCallResult.

        Returns:
            Retry prompt message.
        """
        lines = [
            "Your previous tool call(s) were invalid. Please correct and try again.",
            "",
        ]
        for call in invalid_calls:
            lines.append(f"Tool: {call.tool_name}")
            lines.append(f"Arguments: {json.dumps(call.arguments)}")
            lines.append(f"Error: {call.error}")
            lines.append("")

        lines.append("Use the correct tool name and arguments format.")
        return "\n".join(lines)


class RetryableToolExecutor:
    """Executor that validates tool calls and retries on failure.

    Wraps HeadToolOrchestrator with validation + retry logic.
    """

    def __init__(
        self,
        orchestrator,
        registry: ForgeToolRegistry | None = None,
        max_retries: int = 3,
    ) -> None:
        self.orchestrator = orchestrator
        self.validator = HeadToolValidator(registry=registry)
        self.max_retries = max_retries

    def execute_with_retry(
        self,
        tool_name: str,
        arguments: dict,
        messages: list[dict],
        model_callback,
    ) -> str:
        """Execute a tool call with validation and retry.

        Args:
            tool_name: Tool name from model.
            arguments: Tool arguments from model.
            messages: Current conversation messages (mutated on retry).
            model_callback: Callable(messages, tools) -> response.

        Returns:
            Tool execution result string.
        """
        for attempt in range(self.max_retries):
            # Validate
            result = self.validator.validate(tool_name, arguments)
            if result.valid:
                return self.orchestrator.execute(tool_name, arguments)

            # Invalid — inform model and retry
            logger.warning(
                "Tool call invalid (attempt %d/%d): %s - %s",
                attempt + 1,
                self.max_retries,
                tool_name,
                result.error,
            )

            retry_prompt = self.validator.build_retry_prompt([result])
            messages.append({"role": "user", "content": retry_prompt})

            # Re-call model for corrected tool call
            response = model_callback()
            if not response:
                return f"[ERROR: Model did not respond during retry attempt {attempt + 1}]"

            # Extract new tool calls from response
            if response.tool_calls:
                # Ollama native format
                tc = response.tool_calls[0]
                tool_name = tc.name
                arguments = tc.arguments
            else:
                # Try text extraction
                extracted = self.validator.extract_tool_calls(response.content or "")
                if extracted:
                    tool_name = extracted[0].tool_name
                    arguments = extracted[0].arguments
                else:
                    return f"[ERROR: Model did not provide a valid tool call after {attempt + 1} retries]"

        return f"[ERROR: Tool call failed validation after {self.max_retries} retries]"
