"""Memory data models and enums."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


@dataclass
class MemoryEntry:
    """A single memory entry."""

    id: int | None = None
    agent_id: str = ""
    workflow_id: str | None = None
    memory_type: str = "conversation"  # conversation, fact, preference, learned
    content: str = ""
    metadata: dict = field(default_factory=dict)
    importance: float = 0.5  # 0.0 to 1.0
    created_at: datetime | None = None
    accessed_at: datetime | None = None
    access_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            agent_id=data.get("agent_id", ""),
            workflow_id=data.get("workflow_id"),
            memory_type=data.get("memory_type", "conversation"),
            content=data.get("content", ""),
            metadata=json.loads(data["metadata"])
            if isinstance(data.get("metadata"), str)
            else data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            accessed_at=datetime.fromisoformat(data["accessed_at"])
            if data.get("accessed_at")
            else None,
            access_count=data.get("access_count", 0),
        )


class MessageRole(Enum):
    """Role of a message in the context window."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in the context window."""

    role: MessageRole
    content: str
    name: str | None = None  # For tool messages
    tool_call_id: str | None = None  # For tool responses
    metadata: dict = field(default_factory=dict)
    tokens: int = 0  # Cached token count
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        """Convert to API message format."""
        msg = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
            tokens=data.get("tokens", 0),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(UTC),
        )


@dataclass
class ContextWindowStats:
    """Statistics about the context window."""

    total_tokens: int = 0
    message_count: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    system_tokens: int = 0
    available_tokens: int = 0
    utilization_percent: float = 0.0
