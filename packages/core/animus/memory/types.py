"""Dataclasses and enums for the Animus memory layer.

No I/O here — only the shape of memories, facts, procedures, and
conversations. Store implementations and the MemoryLayer façade live
in sibling modules.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class MemoryType(Enum):
    """Types of memory in the system."""

    EPISODIC = "episodic"  # What happened (conversations, events, decisions)
    SEMANTIC = "semantic"  # What you know (facts, preferences, entities)
    PROCEDURAL = "procedural"  # How you do things (workflows, patterns)
    ACTIVE = "active"  # Current context (live state)


class MemorySource(Enum):
    """How the memory was acquired."""

    STATED = "stated"  # User explicitly told
    INFERRED = "inferred"  # Derived from context
    LEARNED = "learned"  # Pattern detected over time


class Sensitivity(Enum):
    """Disclosure classification for a memory.

    Used by the per-tier ChromaDB collection split (Stage 2.B) and the
    MCP-egress scope gate. Stricter tiers require explicit opt-in via
    ``MemoryLayer.recall(allowed_tiers=...)``.

    Ordering (least → most sensitive): PUBLIC < PERSONAL < CONFIDENTIAL < SECRET.
    """

    PUBLIC = "public"  # Safe to surface anywhere — public refs, podcast facts
    PERSONAL = "personal"  # Default for own-notes — emails, drafts, decisions
    CONFIDENTIAL = "confidential"  # Client / legal / employer — TIAID notes, Toyota context
    SECRET = "secret"  # Credentials, financial, anything that must never cross boundaries


@dataclass
class Memory:
    """A single memory entry with structured metadata."""

    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime
    updated_at: datetime
    metadata: dict
    # Phase 1 additions
    tags: list[str] = field(default_factory=list)
    source: str = "stated"  # stated | inferred | learned
    confidence: float = 1.0  # 0.0-1.0
    subtype: str | None = None  # e.g., "conversation", "fact", "preference"
    # Context Core versioning fields
    version: int = 1
    parent_id: str | None = None  # previous version's memory ID
    change_summary: str | None = None  # what changed from parent
    provenance: str = "direct"  # "direct" | "sync" | "consolidation" | "import" | "mcp"
    # Stage 2 hardening — disclosure tier (default PUBLIC for backward compat)
    sensitivity: Sensitivity = Sensitivity.PUBLIC

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "source": self.source,
            "confidence": self.confidence,
            "subtype": self.subtype,
            "version": self.version,
            "parent_id": self.parent_id,
            "change_summary": self.change_summary,
            "provenance": self.provenance,
            "sensitivity": self.sensitivity.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Memory:
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            source=data.get("source", "stated"),
            confidence=data.get("confidence", 1.0),
            subtype=data.get("subtype"),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            change_summary=data.get("change_summary"),
            provenance=data.get("provenance", "direct"),
            sensitivity=Sensitivity(data.get("sensitivity", Sensitivity.PUBLIC.value)),
        )

    @classmethod
    def create(
        cls,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        source: str = "stated",
        confidence: float = 1.0,
        subtype: str | None = None,
        version: int = 1,
        parent_id: str | None = None,
        change_summary: str | None = None,
        provenance: str = "direct",
        sensitivity: Sensitivity = Sensitivity.PUBLIC,
    ) -> Memory:
        """Factory method to create a Memory with auto-generated id and timestamps."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            tags=tags or [],
            source=source,
            confidence=confidence,
            subtype=subtype,
            version=version,
            parent_id=parent_id,
            change_summary=change_summary,
            provenance=provenance,
            sensitivity=sensitivity,
        )

    def add_tag(self, tag: str) -> None:
        """Add a tag (normalized to lowercase)."""
        normalized = tag.lower().strip()
        if normalized and normalized not in self.tags:
            self.tags.append(normalized)
            self.updated_at = datetime.now()

    def remove_tag(self, tag: str) -> bool:
        """Remove a tag. Returns True if removed."""
        normalized = tag.lower().strip()
        if normalized in self.tags:
            self.tags.remove(normalized)
            self.updated_at = datetime.now()
            return True
        return False


@dataclass
class SemanticFact:
    """Structured knowledge representation (subject-predicate-object)."""

    subject: str
    predicate: str
    obj: str  # 'object' is reserved
    category: str = "fact"  # fact | preference | entity | relationship
    confidence: float = 1.0
    source: str = "stated"

    def to_content(self) -> str:
        """Convert to natural language content."""
        return f"{self.subject} {self.predicate} {self.obj}"

    def to_metadata(self) -> dict:
        """Convert structured fields to metadata dict."""
        return {
            "fact_subject": self.subject,
            "fact_predicate": self.predicate,
            "fact_object": self.obj,
            "fact_category": self.category,
        }


@dataclass
class Procedure:
    """A learned workflow or pattern."""

    name: str
    trigger: str  # What triggers this procedure
    steps: list[str]
    frequency: int = 0  # Times used
    last_used: datetime | None = None

    def to_content(self) -> str:
        """Convert to natural language content."""
        steps_text = "; ".join(f"{i + 1}. {s}" for i, s in enumerate(self.steps))
        return f"Procedure '{self.name}': When {self.trigger}, do: {steps_text}"

    def to_metadata(self) -> dict:
        """Convert structured fields to metadata dict."""
        return {
            "procedure_name": self.name,
            "procedure_trigger": self.trigger,
            "procedure_steps": json.dumps(self.steps),
            "procedure_frequency": self.frequency,
            "procedure_last_used": self.last_used.isoformat() if self.last_used else None,
        }

    def use(self) -> None:
        """Record usage of this procedure."""
        self.frequency += 1
        self.last_used = datetime.now()


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class Conversation:
    """A conversation session."""

    id: str
    messages: list[Message]
    started_at: datetime
    ended_at: datetime | None = None
    metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the conversation."""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        return msg

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "messages": [m.to_dict() for m in self.messages],
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Conversation:
        return cls(
            id=data["id"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            started_at=datetime.fromisoformat(data["started_at"]),
            ended_at=(datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None),
            metadata=data.get("metadata", {}),
        )

    def to_memory_content(self) -> str:
        """Convert conversation to a string for memory storage."""
        lines = [f"Conversation from {self.started_at.strftime('%Y-%m-%d %H:%M')}:"]
        for msg in self.messages:
            prefix = "User" if msg.role == "user" else "Animus"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

    @classmethod
    def new(cls) -> Conversation:
        """Create a new conversation."""
        return cls(
            id=str(uuid.uuid4()),
            messages=[],
            started_at=datetime.now(),
        )
