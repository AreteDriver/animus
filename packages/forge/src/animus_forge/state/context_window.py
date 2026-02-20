"""Context window management with token budgeting."""

from __future__ import annotations

import logging
from collections.abc import Callable

from .agent_memory import AgentMemory
from .memory_models import ContextWindowStats, Message, MessageRole

logger = logging.getLogger(__name__)


class ContextWindow:
    """Manages conversation context with token budgeting.

    Provides:
    - Message history with role tracking
    - Token counting and budget management
    - Automatic truncation when exceeding limits
    - Context summarization for long conversations
    - Integration with AgentMemory for persistence
    """

    # Default token limits for common models
    MODEL_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-sonnet-4-20250514": 200000,
        "claude-opus-4-5-20251101": 200000,
    }

    def __init__(
        self,
        max_tokens: int = 128000,
        reserve_tokens: int = 4096,
        token_counter: Callable[[str], int] | None = None,
        summarizer: Callable[[list[Message]], str] | None = None,
        memory: AgentMemory | None = None,
        agent_id: str = "default",
    ):
        """Initialize context window.

        Args:
            max_tokens: Maximum context window size
            reserve_tokens: Tokens to reserve for response
            token_counter: Function to count tokens (default: char/4 estimate)
            summarizer: Function to summarize messages (optional)
            memory: AgentMemory for persistence (optional)
            agent_id: Agent identifier for memory storage
        """
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.token_counter = token_counter or self._estimate_tokens
        self.summarizer = summarizer
        self.memory = memory
        self.agent_id = agent_id

        self._messages: list[Message] = []
        self._system_message: Message | None = None
        self._summary: str | None = None
        self._total_tokens: int = 0

    @classmethod
    def for_model(
        cls,
        model: str,
        reserve_tokens: int = 4096,
        **kwargs,
    ) -> ContextWindow:
        """Create context window with model-specific limits.

        Args:
            model: Model name
            reserve_tokens: Tokens to reserve for response
            **kwargs: Additional arguments for __init__

        Returns:
            Configured ContextWindow
        """
        max_tokens = cls.MODEL_LIMITS.get(model, 128000)
        return cls(max_tokens=max_tokens, reserve_tokens=reserve_tokens, **kwargs)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation).

        Uses ~4 characters per token as a conservative estimate.
        For accurate counts, provide a model-specific token_counter.
        """
        return len(text) // 4 + 1

    @property
    def available_tokens(self) -> int:
        """Get available tokens for new content."""
        system_tokens = self._system_message.tokens if self._system_message else 0
        summary_tokens = self._estimate_tokens(self._summary) if self._summary else 0
        used = system_tokens + summary_tokens + self._total_tokens + self.reserve_tokens
        return max(0, self.max_tokens - used)

    def set_system_message(self, content: str) -> None:
        """Set the system message.

        Args:
            content: System message content
        """
        tokens = self.token_counter(content)
        self._system_message = Message(
            role=MessageRole.SYSTEM,
            content=content,
            tokens=tokens,
        )

    def add_message(
        self,
        role: MessageRole | str,
        content: str,
        name: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict | None = None,
    ) -> Message:
        """Add a message to the context.

        Args:
            role: Message role
            content: Message content
            name: Optional name for tool messages
            tool_call_id: Optional tool call ID
            metadata: Optional metadata

        Returns:
            The added message
        """
        if isinstance(role, str):
            role = MessageRole(role)

        tokens = self.token_counter(content)
        message = Message(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            metadata=metadata or {},
            tokens=tokens,
        )

        self._messages.append(message)
        self._total_tokens += tokens

        # Check if we need to truncate (when over budget)
        if self._is_over_budget():
            self._truncate()

        return message

    def _is_over_budget(self) -> bool:
        """Check if context exceeds token budget."""
        system_tokens = self._system_message.tokens if self._system_message else 0
        summary_tokens = self._estimate_tokens(self._summary) if self._summary else 0
        used = system_tokens + summary_tokens + self._total_tokens + self.reserve_tokens
        return used > self.max_tokens

    def add_user_message(self, content: str, metadata: dict | None = None) -> Message:
        """Add a user message."""
        return self.add_message(MessageRole.USER, content, metadata=metadata)

    def add_assistant_message(self, content: str, metadata: dict | None = None) -> Message:
        """Add an assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, metadata=metadata)

    def add_tool_message(
        self,
        content: str,
        name: str,
        tool_call_id: str | None = None,
    ) -> Message:
        """Add a tool response message."""
        return self.add_message(MessageRole.TOOL, content, name=name, tool_call_id=tool_call_id)

    def get_messages(self, include_system: bool = True) -> list[dict]:
        """Get messages in API format.

        Args:
            include_system: Include system message

        Returns:
            List of message dicts for API call
        """
        messages = []

        # System message first
        if include_system and self._system_message:
            messages.append(self._system_message.to_dict())

        # Add summary as a system message if present
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary:\n{self._summary}",
                }
            )

        # Add all conversation messages
        messages.extend(m.to_dict() for m in self._messages)

        return messages

    def _truncate(self) -> None:
        """Truncate context to fit within limits.

        Strategy:
        1. If summarizer available, summarize older messages
        2. Otherwise, remove oldest messages until within limit
        """
        if not self._messages:
            return

        target_tokens = self.max_tokens - self.reserve_tokens
        system_tokens = self._system_message.tokens if self._system_message else 0
        available = target_tokens - system_tokens

        # Try summarization first
        if self.summarizer and len(self._messages) > 10:
            # Summarize older half of messages
            midpoint = len(self._messages) // 2
            to_summarize = self._messages[:midpoint]

            if to_summarize:
                try:
                    new_summary = self.summarizer(to_summarize)

                    # Combine with existing summary if present
                    if self._summary:
                        self._summary = f"{self._summary}\n\n{new_summary}"
                    else:
                        self._summary = new_summary

                    # Remove summarized messages
                    removed_tokens = sum(m.tokens for m in to_summarize)
                    self._messages = self._messages[midpoint:]
                    self._total_tokens -= removed_tokens

                    logger.info(f"Summarized {midpoint} messages, freed {removed_tokens} tokens")
                    return
                except Exception as e:
                    logger.warning(f"Summarization failed: {e}")

        # Fallback: remove oldest messages
        summary_tokens = self._estimate_tokens(self._summary) if self._summary else 0
        available -= summary_tokens

        while self._total_tokens > available and self._messages:
            removed = self._messages.pop(0)
            self._total_tokens -= removed.tokens
            logger.debug(f"Removed message with {removed.tokens} tokens")

    def clear(self) -> None:
        """Clear all messages (keeps system message)."""
        self._messages = []
        self._summary = None
        self._total_tokens = 0

    def get_stats(self) -> ContextWindowStats:
        """Get context window statistics."""
        system_tokens = self._system_message.tokens if self._system_message else 0
        summary_tokens = self._estimate_tokens(self._summary) if self._summary else 0

        user_count = sum(1 for m in self._messages if m.role == MessageRole.USER)
        assistant_count = sum(1 for m in self._messages if m.role == MessageRole.ASSISTANT)

        total = system_tokens + summary_tokens + self._total_tokens
        utilization = (total / self.max_tokens) * 100 if self.max_tokens > 0 else 0

        return ContextWindowStats(
            total_tokens=total,
            message_count=len(self._messages),
            user_messages=user_count,
            assistant_messages=assistant_count,
            system_tokens=system_tokens,
            available_tokens=self.available_tokens,
            utilization_percent=round(utilization, 1),
        )

    def save_to_memory(
        self,
        workflow_id: str | None = None,
        importance: float = 0.5,
    ) -> list[int]:
        """Save current context to agent memory.

        Args:
            workflow_id: Optional workflow context
            importance: Importance score for memories

        Returns:
            List of memory entry IDs
        """
        if not self.memory:
            return []

        entry_ids = []
        for msg in self._messages:
            content = f"[{msg.role.value}] {msg.content}"
            entry_id = self.memory.store(
                agent_id=self.agent_id,
                content=content,
                memory_type="conversation",
                workflow_id=workflow_id,
                metadata={
                    "role": msg.role.value,
                    "timestamp": msg.timestamp.isoformat(),
                },
                importance=importance,
            )
            entry_ids.append(entry_id)

        return entry_ids

    def load_from_memory(
        self,
        workflow_id: str | None = None,
        limit: int = 20,
    ) -> int:
        """Load recent context from agent memory.

        Args:
            workflow_id: Optional workflow context filter
            limit: Maximum messages to load

        Returns:
            Number of messages loaded
        """
        if not self.memory:
            return 0

        memories = self.memory.recall(
            agent_id=self.agent_id,
            memory_type="conversation",
            workflow_id=workflow_id,
            limit=limit,
        )

        count = 0
        for mem in memories:
            # Parse role from content format "[role] content"
            content = mem.content
            role = MessageRole.USER

            if content.startswith("["):
                end_bracket = content.find("]")
                if end_bracket > 0:
                    role_str = content[1:end_bracket]
                    try:
                        role = MessageRole(role_str)
                    except ValueError:
                        pass  # Graceful degradation: invalid role string falls back to default USER
                    content = content[end_bracket + 2 :]  # Skip "] "

            self.add_message(role, content, metadata=mem.metadata)
            count += 1

        return count

    def to_dict(self) -> dict:
        """Serialize context window to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
            "agent_id": self.agent_id,
            "system_message": self._system_message.to_dict() if self._system_message else None,
            "messages": [
                {
                    **m.to_dict(),
                    "tokens": m.tokens,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata,
                }
                for m in self._messages
            ],
            "summary": self._summary,
            "total_tokens": self._total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> ContextWindow:
        """Deserialize context window from dictionary."""
        window = cls(
            max_tokens=data.get("max_tokens", 128000),
            reserve_tokens=data.get("reserve_tokens", 4096),
            agent_id=data.get("agent_id", "default"),
            **kwargs,
        )

        if data.get("system_message"):
            sys_msg = data["system_message"]
            window.set_system_message(sys_msg["content"])

        window._summary = data.get("summary")

        for msg_data in data.get("messages", []):
            window._messages.append(Message.from_dict(msg_data))

        window._total_tokens = data.get("total_tokens", 0)

        return window
