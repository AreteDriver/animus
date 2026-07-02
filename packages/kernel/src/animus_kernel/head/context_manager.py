"""Head context manager — token budgeting, pruning, and summarization.

Wraps the REPL message list with automatic context-window management:
- Token counting (char/4 estimate, model-specific limits)
- Pruning when approaching the limit (keeps tool-call pairs intact)
- Optional summarization of dropped messages
- Stats reporting for /session and /context commands
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ContextStats:
    """Context window statistics."""

    max_tokens: int = 0
    reserve_tokens: int = 0
    total_tokens: int = 0
    message_count: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_messages: int = 0
    system_tokens: int = 0
    summary_tokens: int = 0
    available_tokens: int = 0
    utilization_percent: float = 0.0
    dropped_messages: int = 0


class HeadContextManager:
    """Manages REPL message history with token budgeting and pruning.

    Args:
        model: Ollama model name (used to look up context-window limit)
        reserve_tokens: Tokens to reserve for the model's response
        summary_dir: Directory to persist generated summaries
        summarizer: Optional callable(messages) -> str for summarizing dropped context
    """

    # Context-window limits for common Ollama models (tokens)
    MODEL_LIMITS: dict[str, int] = {
        # Qwen2.5 family
        "qwen2.5:32b": 32768,
        "qwen2.5:14b": 32768,
        "qwen2.5:7b": 32768,
        "qwen2.5:0.5b": 32768,
        "qwen2.5-coder:32b": 32768,
        "qwen2.5-coder:14b": 32768,
        "qwen2.5-coder:7b": 32768,
        # Llama 3.1 family
        "llama3.1:8b": 8192,
        "llama3.1:70b": 128000,
        "llama3.1:405b": 128000,
        # Hermes 3 family
        "hermes3:8b": 8192,
        "hermes3:70b": 128000,
        # Mistral family
        "mistral:7b": 32768,
        "mistral-nemo": 128000,
        "mixtral:8x7b": 32768,
        "mixtral:8x22b": 64000,
        # Phi family
        "phi4:14b": 16384,
        "phi3:14b": 16384,
        "phi3:3.8b": 16384,
        # Gemma family
        "gemma2:27b": 8192,
        "gemma2:9b": 8192,
        "gemma2:2b": 8192,
        # DeepSeek family
        "deepseek-coder:33b": 16384,
        "deepseek-coder:6.7b": 16384,
        # Generic fallback
        "default": 8192,
    }

    def __init__(
        self,
        model: str = "qwen2.5:32b",
        reserve_tokens: int = 2048,
        summary_dir: str | Path | None = None,
        summarizer: Callable | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = self._resolve_limit(model)
        self.reserve_tokens = reserve_tokens
        self.summarizer = summarizer
        self.dropped_messages = 0

        # Message store — raw dicts compatible with Ollama API
        self._messages: list[dict] = []
        self._summary = ""
        self._summary_tokens = 0

        if summary_dir is None:
            summary_dir = Path.home() / ".animus" / "summaries"
        self.summary_dir = Path(summary_dir)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, msg: dict) -> None:
        """Add a message to the context.

        Automatically prunes if the budget is exceeded.
        """
        if not isinstance(msg, dict) or "role" not in msg:
            logger.warning("Skipping invalid message: %s", msg)
            return

        self._messages.append(msg)
        self._prune_if_needed()

    def get_messages(self) -> list[dict]:
        """Return messages ready for the API call.

        Ensures the system message is first and includes the summary
        as a system message when one exists.
        """
        result: list[dict] = []

        # System message first (extracted from _messages[0] if present)
        system = self._system_message()
        if system:
            result.append(system)

        # Summary as a secondary system message
        if self._summary:
            result.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{self._summary}",
            })

        # Conversation messages — skip the first system message if we already added it
        start = 1 if self._messages and self._messages[0].get("role") == "system" else 0
        result.extend(self._messages[start:])
        return result

    def set_summary(self, summary: str) -> None:
        """Set or overwrite the conversation summary."""
        self._summary = summary.strip()
        self._summary_tokens = self._estimate_tokens(self._summary)

    def clear(self) -> None:
        """Clear all messages (keeps summary)."""
        self._messages = []
        self.dropped_messages = 0

    def get_stats(self) -> ContextStats:
        """Return current context window statistics."""
        system = self._system_message()
        sys_tokens = self._estimate_tokens(system.get("content", "")) if system else 0

        user_count = sum(1 for m in self._messages if m.get("role") == "user")
        assistant_count = sum(1 for m in self._messages if m.get("role") == "assistant")
        tool_count = sum(1 for m in self._messages if m.get("role") == "tool")

        total = sys_tokens + self._summary_tokens + self._total_message_tokens() + self.reserve_tokens
        utilization = (total / self.max_tokens) * 100 if self.max_tokens > 0 else 0.0

        return ContextStats(
            max_tokens=self.max_tokens,
            reserve_tokens=self.reserve_tokens,
            total_tokens=total,
            message_count=len(self._messages),
            user_messages=user_count,
            assistant_messages=assistant_count,
            tool_messages=tool_count,
            system_tokens=sys_tokens,
            summary_tokens=self._summary_tokens,
            available_tokens=max(0, self.max_tokens - total),
            utilization_percent=round(utilization, 1),
            dropped_messages=self.dropped_messages,
        )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_if_needed(self) -> None:
        """Remove oldest messages until under the token budget.

        Strategy:
        1. Never remove the system message (it's reconstructed from _messages).
        2. Preserve tool-call pairs: an assistant message with tool_calls
           and its corresponding tool responses are removed as a unit.
        3. If a summarizer is available, summarize removed messages first.
        4. Always keep at least the last 2 user turns to preserve immediate context.
        """
        budget = self.max_tokens - self.reserve_tokens
        system = self._system_message()
        sys_tokens = self._estimate_tokens(system.get("content", "")) if system else 0
        available = budget - sys_tokens - self._summary_tokens

        if self._total_message_tokens() <= available:
            return

        # Build removal units: each unit is a list of message indices that must
        # be removed together (tool-call pairs).
        units = self._build_removal_units()

        removed_count = 0
        removed_content: list[str] = []

        # Remove oldest units until under budget, but preserve at least
        # the last 2 user-message blocks.
        user_blocks = self._count_user_blocks()
        min_keep = min(2, user_blocks)

        indices_to_remove: set[int] = set()

        while units and self._total_message_tokens() > available:
            # Count how many user blocks would remain after removing this unit
            if not self._can_remove_unit(units[0], min_keep):
                break

            unit = units.pop(0)
            for idx in unit:
                indices_to_remove.add(idx)

        if indices_to_remove:
            # Rebuild messages excluding removed indices
            new_messages = []
            for idx, msg in enumerate(self._messages):
                if idx in indices_to_remove:
                    removed_count += 1
                    self.dropped_messages += 1
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if content:
                        removed_content.append(f"[{role}] {content[:200]}")
                else:
                    new_messages.append(msg)
            self._messages = new_messages

        if removed_count > 0:
            logger.info("Pruned %d messages to stay within token budget", removed_count)
            self._maybe_summarize(removed_content)

    def _build_removal_units(self) -> list[list[int]]:
        """Build removal units preserving tool-call pairs.

        Returns list of index lists, each representing a unit that must
        be removed together. Oldest units first.
        """
        units: list[list[int]] = []
        i = 0
        while i < len(self._messages):
            msg = self._messages[i]
            role = msg.get("role")

            if role == "assistant" and msg.get("tool_calls"):
                # Find all tool responses that match this assistant's tool_calls
                unit = [i]
                tool_call_ids = {tc.get("id") for tc in msg.get("tool_calls", [])}
                j = i + 1
                while j < len(self._messages):
                    next_msg = self._messages[j]
                    if (
                        next_msg.get("role") == "tool"
                        and next_msg.get("tool_call_id") in tool_call_ids
                    ):
                        unit.append(j)
                        j += 1
                    else:
                        break
                units.append(unit)
                i = j
            else:
                units.append([i])
                i += 1
        return units

    def _can_remove_unit(self, unit: list[int], min_user_blocks: int) -> bool:
        """Check if removing a unit would leave enough user context."""
        # Simulate removal
        remaining = [m for idx, m in enumerate(self._messages) if idx not in unit]
        remaining_user_blocks = self._count_user_blocks_in(remaining)
        return remaining_user_blocks >= min_user_blocks

    def _count_user_blocks(self) -> int:
        """Count user message blocks in current messages."""
        return self._count_user_blocks_in(self._messages)

    @staticmethod
    def _count_user_blocks_in(messages: list[dict]) -> int:
        """Count user message blocks."""
        return sum(1 for m in messages if m.get("role") == "user")

    def _maybe_summarize(self, removed_content: list[str]) -> None:
        """Optionally summarize removed content and append to summary."""
        if not removed_content:
            return

        if self.summarizer:
            try:
                new_summary = self.summarizer(removed_content)
                if new_summary:
                    if self._summary:
                        self._summary = f"{self._summary}\n\n{new_summary}"
                    else:
                        self._summary = new_summary
                    self._summary_tokens = self._estimate_tokens(self._summary)
                    return
            except Exception:
                logger.warning("Summarization failed", exc_info=True)

        # Fallback: append a terse note
        user_notes = [c for c in removed_content if c.startswith("[user]")]
        if user_notes:
            note = f"(Earlier: {len(removed_content)} messages including {len(user_notes)} user turns.)"
            if self._summary:
                self._summary = f"{self._summary}\n{note}"
            else:
                self._summary = note
            self._summary_tokens = self._estimate_tokens(self._summary)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _system_message(self) -> dict | None:
        """Return the first system message if present."""
        if self._messages and self._messages[0].get("role") == "system":
            return self._messages[0]
        return None

    def _total_message_tokens(self) -> int:
        """Estimate total tokens in all stored messages."""
        total = 0
        for msg in self._messages:
            total += self._estimate_message_tokens(msg)
        return total

    def _estimate_message_tokens(self, msg: dict) -> int:
        """Estimate tokens for a single message dict."""
        text = msg.get("content", "") or ""
        # Tool calls add token overhead
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            text += json.dumps(tool_calls)
        return self._estimate_tokens(text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars per token)."""
        if not text:
            return 0
        return len(text) // 4 + 1

    def _resolve_limit(self, model: str) -> int:
        """Resolve context-window limit for a model name."""
        # Exact match
        if model in self.MODEL_LIMITS:
            return self.MODEL_LIMITS[model]

        # Prefix match (e.g., "qwen2.5:32b" might be pulled as "qwen2.5:32b-latest")
        for key, limit in self.MODEL_LIMITS.items():
            if model.startswith(key) or key.startswith(model.split(":")[0] if ":" in model else model):
                return limit

        # Wildcard family match
        model_lower = model.lower()
        if "qwen2.5" in model_lower:
            return 32768
        if "llama3.1" in model_lower or "llama3" in model_lower:
            return 8192
        if "hermes3" in model_lower:
            return 8192
        if "mistral" in model_lower or "mixtral" in model_lower:
            return 32768
        if "phi" in model_lower:
            return 16384
        if "gemma" in model_lower:
            return 8192
        if "deepseek" in model_lower:
            return 16384

        return self.MODEL_LIMITS["default"]
