"""
Animus Memory Layer

Handles persistence of context, knowledge, and patterns.
Supports both local JSON storage and ChromaDB vector storage.
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from animus.logging import get_logger

logger = get_logger("memory")


class MemoryType(Enum):
    """Types of memory in the system."""

    EPISODIC = "episodic"  # What happened (conversations, events)
    SEMANTIC = "semantic"  # What you know (facts, knowledge)
    PROCEDURAL = "procedural"  # How you do things (workflows, patterns)
    ACTIVE = "active"  # Current context (live state)


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime
    updated_at: datetime
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


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
    def from_dict(cls, data: dict) -> "Message":
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
    def from_dict(cls, data: dict) -> "Conversation":
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
    def new(cls) -> "Conversation":
        """Create a new conversation."""
        return cls(
            id=str(uuid.uuid4()),
            messages=[],
            started_at=datetime.now(),
        )


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    def store(self, memory: Memory) -> None:
        """Store a memory."""
        pass

    @abstractmethod
    def retrieve(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID."""
        pass

    @abstractmethod
    def search(
        self, query: str, memory_type: MemoryType | None = None, limit: int = 10
    ) -> list[Memory]:
        """Search memories by content."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        pass

    @abstractmethod
    def list_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        """List all memories, optionally filtered by type."""
        pass


class LocalMemoryStore(MemoryStore):
    """
    Simple local file-based memory store.

    Uses substring matching for search. Fallback when ChromaDB unavailable.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.memories_file = data_dir / "memories.json"
        self._memories: dict[str, Memory] = {}
        self._load()
        logger.debug(f"LocalMemoryStore initialized at {data_dir}")

    def _load(self):
        """Load memories from disk."""
        if self.memories_file.exists():
            with open(self.memories_file) as f:
                data = json.load(f)
                self._memories = {k: Memory.from_dict(v) for k, v in data.items()}
            logger.info(f"Loaded {len(self._memories)} memories from disk")

    def _save(self):
        """Save memories to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.memories_file, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._memories.items()}, f, indent=2)

    def store(self, memory: Memory) -> None:
        self._memories[memory.id] = memory
        self._save()
        logger.debug(f"Stored memory {memory.id[:8]}")

    def retrieve(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def search(
        self, query: str, memory_type: MemoryType | None = None, limit: int = 10
    ) -> list[Memory]:
        """Simple substring search."""
        results = []
        query_lower = query.lower()

        for memory in self._memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            if query_lower in memory.content.lower():
                results.append(memory)
            if len(results) >= limit:
                break

        logger.debug(f"Search '{query}' found {len(results)} results")
        return results

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save()
            logger.debug(f"Deleted memory {memory_id[:8]}")
            return True
        return False

    def list_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        if memory_type:
            return [m for m in self._memories.values() if m.memory_type == memory_type]
        return list(self._memories.values())


class ChromaMemoryStore(MemoryStore):
    """
    Vector-based memory store using ChromaDB.

    Provides semantic search using embeddings.
    """

    def __init__(self, data_dir: Path, collection_name: str = "animus_memories"):
        self.data_dir = data_dir
        self.collection_name = collection_name
        self._memories: dict[str, Memory] = {}  # Local cache for metadata

        try:
            import chromadb

            self.chroma_dir = data_dir / "chroma"
            self.chroma_dir.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB initialized at {self.chroma_dir} "
                f"with {self.collection.count()} documents"
            )
        except ImportError as e:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb") from e
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

        self._load_metadata()

    def _load_metadata(self):
        """Load memory metadata from ChromaDB."""
        try:
            results = self.collection.get(include=["metadatas", "documents"])
            for i, mem_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                content = results["documents"][i] if results["documents"] else ""

                self._memories[mem_id] = Memory(
                    id=mem_id,
                    content=content,
                    memory_type=MemoryType(metadata.get("memory_type", "semantic")),
                    created_at=datetime.fromisoformat(
                        metadata.get("created_at", datetime.now().isoformat())
                    ),
                    updated_at=datetime.fromisoformat(
                        metadata.get("updated_at", datetime.now().isoformat())
                    ),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ("memory_type", "created_at", "updated_at")
                    },
                )
        except Exception as e:
            logger.warning(f"Failed to load metadata from ChromaDB: {e}")

    def store(self, memory: Memory) -> None:
        """Store memory with embedding."""
        metadata = {
            "memory_type": memory.memory_type.value,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            **{k: str(v) for k, v in memory.metadata.items()},
        }

        self.collection.upsert(
            ids=[memory.id],
            documents=[memory.content],
            metadatas=[metadata],
        )
        self._memories[memory.id] = memory
        logger.debug(f"Stored memory {memory.id[:8]} in ChromaDB")

    def retrieve(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def search(
        self, query: str, memory_type: MemoryType | None = None, limit: int = 10
    ) -> list[Memory]:
        """Semantic search using vector similarity."""
        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type.value}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            memories = []
            for i, mem_id in enumerate(results["ids"][0]):
                if mem_id in self._memories:
                    memories.append(self._memories[mem_id])
                else:
                    # Reconstruct from results
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    content = results["documents"][0][i] if results["documents"] else ""
                    memories.append(
                        Memory(
                            id=mem_id,
                            content=content,
                            memory_type=MemoryType(metadata.get("memory_type", "semantic")),
                            created_at=datetime.fromisoformat(
                                metadata.get("created_at", datetime.now().isoformat())
                            ),
                            updated_at=datetime.fromisoformat(
                                metadata.get("updated_at", datetime.now().isoformat())
                            ),
                            metadata={},
                        )
                    )

            logger.debug(f"Semantic search '{query[:30]}...' found {len(memories)} results")
            return memories

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

    def delete(self, memory_id: str) -> bool:
        try:
            self.collection.delete(ids=[memory_id])
            if memory_id in self._memories:
                del self._memories[memory_id]
            logger.debug(f"Deleted memory {memory_id[:8]} from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    def list_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        if memory_type:
            return [m for m in self._memories.values() if m.memory_type == memory_type]
        return list(self._memories.values())


class MemoryLayer:
    """
    Main memory layer interface.

    Coordinates between different memory types and storage backends.
    """

    def __init__(self, data_dir: Path, backend: str = "chroma"):
        self.data_dir = data_dir
        self.backend_type = backend

        if backend == "chroma":
            try:
                self.store = ChromaMemoryStore(data_dir)
            except ImportError:
                logger.warning("ChromaDB not available, falling back to JSON storage")
                self.store = LocalMemoryStore(data_dir)
        else:
            self.store = LocalMemoryStore(data_dir)

        logger.info(f"MemoryLayer initialized with {type(self.store).__name__}")

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: dict | None = None,
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: The content to remember
            memory_type: Type of memory
            metadata: Optional additional data

        Returns:
            The created Memory object
        """
        now = datetime.now()

        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        self.store.store(memory)
        logger.info(f"Remembered {memory_type.value} memory: {content[:50]}...")
        return memory

    def recall(
        self, query: str, memory_type: MemoryType | None = None, limit: int = 10
    ) -> list[Memory]:
        """
        Retrieve relevant memories.

        Args:
            query: What to search for
            memory_type: Optional filter by type
            limit: Maximum results

        Returns:
            List of relevant memories
        """
        return self.store.search(query, memory_type, limit)

    def forget(self, memory_id: str) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted, False if not found
        """
        return self.store.delete(memory_id)

    def save_conversation(self, conversation: Conversation) -> Memory:
        """
        Save a conversation as an episodic memory.

        Args:
            conversation: The conversation to save

        Returns:
            The created Memory object
        """
        conversation.ended_at = datetime.now()
        content = conversation.to_memory_content()

        return self.remember(
            content=content,
            memory_type=MemoryType.EPISODIC,
            metadata={
                "conversation_id": conversation.id,
                "message_count": len(conversation.messages),
                "duration_seconds": (
                    conversation.ended_at - conversation.started_at
                ).total_seconds(),
            },
        )

    def consolidate(self):
        """
        Consolidate memories - summarize and compress.

        TODO: Implement in Phase 1
        """
        pass
