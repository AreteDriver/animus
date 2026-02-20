"""AI provider step handlers for workflow execution.

Mixin class providing handlers for Claude and OpenAI API steps.
"""

from __future__ import annotations

import logging

from .executor_clients import _get_claude_client, _get_openai_client
from .loader import StepConfig

logger = logging.getLogger(__name__)


class AIHandlersMixin:
    """Mixin providing AI provider step handlers.

    Expects the following attributes from the host class:
    - dry_run: bool
    - memory_manager: WorkflowMemoryManager | None
    """

    def _execute_claude_code(self, step: StepConfig, context: dict) -> dict:
        """Execute a Claude Code step using the Anthropic API.

        Params:
            prompt: The task/prompt to send
            role: Agent role (planner, builder, tester, reviewer)
            model: Claude model to use (default: claude-sonnet-4-20250514)
            max_tokens: Maximum tokens in response (default: 4096)
            system_prompt: Optional custom system prompt (overrides role)
            use_memory: Enable memory context injection (default: True)
        """
        prompt = step.params.get("prompt", "")
        role = step.params.get("role", "builder")
        model = step.params.get("model", "claude-sonnet-4-20250514")
        max_tokens = step.params.get("max_tokens", 4096)
        system_prompt = step.params.get("system_prompt")
        use_memory = step.params.get("use_memory", True)

        # Substitute context variables in prompt
        for key, value in context.items():
            if isinstance(value, str):
                prompt = prompt.replace(f"${{{key}}}", value)

        # Inject memory context if available
        if use_memory and self.memory_manager:
            prompt = self.memory_manager.inject_context(role, prompt)

        # Inject budget context if available
        budget_mgr = getattr(self, "budget_manager", None)
        if budget_mgr:
            budget_ctx = budget_mgr.get_budget_context()
            if budget_ctx:
                prompt = prompt + "\n\n" + budget_ctx

        # Dry run mode - return mock response
        if self.dry_run:
            output = {
                "role": role,
                "prompt": prompt,
                "response": f"[DRY RUN] Claude {role} would process: {prompt[:100]}...",
                "tokens_used": step.params.get("estimated_tokens", 1000),
                "model": model,
                "dry_run": True,
            }
            # Store output in memory even for dry run
            if self.memory_manager:
                self.memory_manager.store_output(role, step.id, output)
            return output

        # Get Claude client
        client = _get_claude_client()
        if not client:
            raise RuntimeError("Claude Code client not available. Check API key configuration.")

        if not client.is_configured():
            raise RuntimeError("Claude Code client not configured. Set ANTHROPIC_API_KEY.")

        # Execute via API
        if system_prompt:
            # Custom system prompt - use generate_completion
            result = client.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
            )
        else:
            # Role-based execution
            result = client.execute_agent(
                role=role,
                task=prompt,
                context=context.get("_previous_output"),
                model=model,
                max_tokens=max_tokens,
            )

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            # Store error in memory
            if self.memory_manager:
                self.memory_manager.store_error(role, step.id, error_msg)
            raise RuntimeError(f"Claude API error: {error_msg}")

        # Estimate tokens (actual count would require API response metadata)
        response_text = result.get("output", "")
        estimated_tokens = len(response_text) // 4 + len(prompt) // 4

        output = {
            "role": role,
            "prompt": prompt,
            "response": response_text,
            "tokens_used": estimated_tokens,
            "model": model,
        }

        # Propagate consensus metadata into step output
        if result.get("consensus"):
            output["consensus"] = result["consensus"]
        if result.get("pending_user_confirmation"):
            output["pending_user_confirmation"] = True

        # Store output in memory
        if self.memory_manager:
            self.memory_manager.store_output(role, step.id, output)

        return output

    def _execute_openai(self, step: StepConfig, context: dict) -> dict:
        """Execute an OpenAI step using the OpenAI API.

        Params:
            prompt: The prompt to send
            model: OpenAI model to use (default: gpt-4o-mini)
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (optional)
            use_memory: Enable memory context injection (default: True)
        """
        prompt = step.params.get("prompt", "")
        model = step.params.get("model", "gpt-4o-mini")
        system_prompt = step.params.get("system_prompt")
        temperature = step.params.get("temperature", 0.7)
        max_tokens = step.params.get("max_tokens")
        use_memory = step.params.get("use_memory", True)

        # Substitute context variables in prompt
        for key, value in context.items():
            if isinstance(value, str):
                prompt = prompt.replace(f"${{{key}}}", value)

        # Inject memory context if available
        agent_id = f"openai-{model}"
        if use_memory and self.memory_manager:
            prompt = self.memory_manager.inject_context(agent_id, prompt)

        # Dry run mode - return mock response
        if self.dry_run:
            output = {
                "model": model,
                "prompt": prompt,
                "response": f"[DRY RUN] OpenAI {model} would process: {prompt[:100]}...",
                "tokens_used": step.params.get("estimated_tokens", 1000),
                "dry_run": True,
            }
            if self.memory_manager:
                self.memory_manager.store_output(agent_id, step.id, output)
            return output

        # Get OpenAI client
        client = _get_openai_client()
        if not client:
            raise RuntimeError("OpenAI client not available. Check API key configuration.")

        try:
            response_text = client.generate_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
        except Exception as e:
            error_msg = str(e)
            if self.memory_manager:
                self.memory_manager.store_error(agent_id, step.id, error_msg)
            raise RuntimeError(f"OpenAI API error: {e}")

        # Estimate tokens (actual count would require API response metadata)
        estimated_tokens = len(response_text) // 4 + len(prompt) // 4

        output = {
            "model": model,
            "prompt": prompt,
            "response": response_text,
            "tokens_used": estimated_tokens,
        }

        # Store output in memory
        if self.memory_manager:
            self.memory_manager.store_output(agent_id, step.id, output)

        return output
