"""
Animus Memory Layer

Handles persistence of context, knowledge, and patterns.
Supports both local JSON storage and ChromaDB vector storage.

Phase 1: Structured memory with types, tags, confidence, and export/import.
"""

import json
import shutil
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

    EPISODIC = "episodic"  # What happened (conversations, events, decisions)
    SEMANTIC = "semantic"  # What you know (facts, preferences, entities)
    PROCEDURAL = "procedural"  # How you do things (workflows, patterns)
    ACTIVE = "active"  # Current context (live state)


class MemorySource(Enum):
    """How the memory was acquired."""

    STATED = "stated"  # User explicitly told
    INFERRED = "inferred"  # Derived from context
    LEARNED = "learned"  # Pattern detected over time


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
            tags=data.get("tags", []),
            source=data.get("source", "stated"),
            confidence=data.get("confidence", 1.0),
            subtype=data.get("subtype"),
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
    ) -> "Memory":
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
    def update(self, memory: Memory) -> bool:
        """Update an existing memory."""
        pass

    @abstractmethod
    def retrieve(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories with filters."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        pass

    @abstractmethod
    def list_all(self, memory_type: MemoryType | None = None) -> list[Memory]:
        """List all memories, optionally filtered by type."""
        pass

    @abstractmethod
    def get_all_tags(self) -> dict[str, int]:
        """Get all tags with their counts."""
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

    def update(self, memory: Memory) -> bool:
        if memory.id in self._memories:
            self._memories[memory.id] = memory
            self._save()
            logger.debug(f"Updated memory {memory.id[:8]}")
            return True
        return False

    def retrieve(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]:
        """Substring search with filters."""
        results = []
        query_lower = query.lower()

        for memory in self._memories.values():
            # Apply filters
            if memory_type and memory.memory_type != memory_type:
                continue
            if tags and not all(t in memory.tags for t in tags):
                continue
            if source and memory.source != source:
                continue
            if memory.confidence < min_confidence:
                continue
            # Content match
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

    def get_all_tags(self) -> dict[str, int]:
        """Get all tags with counts."""
        tag_counts: dict[str, int] = {}
        for memory in self._memories.values():
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts


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

                # Parse tags from JSON string
                tags_json = metadata.get("tags", "[]")
                try:
                    tags = json.loads(tags_json) if isinstance(tags_json, str) else []
                except json.JSONDecodeError:
                    tags = []

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
                        if k
                        not in (
                            "memory_type",
                            "created_at",
                            "updated_at",
                            "tags",
                            "source",
                            "confidence",
                            "subtype",
                        )
                    },
                    tags=tags,
                    source=metadata.get("source", "stated"),
                    confidence=float(metadata.get("confidence", 1.0)),
                    subtype=metadata.get("subtype"),
                )
        except Exception as e:
            logger.warning(f"Failed to load metadata from ChromaDB: {e}")

    def _build_chroma_metadata(self, memory: Memory) -> dict:
        """Build ChromaDB-compatible metadata dict."""
        metadata = {
            "memory_type": memory.memory_type.value,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "tags": json.dumps(memory.tags),  # Store as JSON string
            "source": memory.source,
            "confidence": memory.confidence,
        }
        if memory.subtype:
            metadata["subtype"] = memory.subtype
        # Add custom metadata (convert to strings)
        for k, v in memory.metadata.items():
            metadata[k] = str(v)
        return metadata

    def store(self, memory: Memory) -> None:
        """Store memory with embedding."""
        metadata = self._build_chroma_metadata(memory)

        self.collection.upsert(
            ids=[memory.id],
            documents=[memory.content],
            metadatas=[metadata],
        )
        self._memories[memory.id] = memory
        logger.debug(f"Stored memory {memory.id[:8]} in ChromaDB")

    def update(self, memory: Memory) -> bool:
        """Update an existing memory."""
        if memory.id in self._memories:
            self.store(memory)  # Upsert handles update
            return True
        return False

    def retrieve(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]:
        """Semantic search with filters."""
        # Build where clause for ChromaDB
        where_conditions = []
        if memory_type:
            where_conditions.append({"memory_type": memory_type.value})
        if source:
            where_conditions.append({"source": source})
        if min_confidence > 0:
            where_conditions.append({"confidence": {"$gte": min_confidence}})

        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit * 2 if tags else limit,  # Over-fetch if filtering tags
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            memories = []
            for i, mem_id in enumerate(results["ids"][0]):
                memory = self._memories.get(mem_id)
                if not memory:
                    # Reconstruct from results
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    content = results["documents"][0][i] if results["documents"] else ""
                    tags_json = metadata.get("tags", "[]")
                    try:
                        mem_tags = json.loads(tags_json) if isinstance(tags_json, str) else []
                    except json.JSONDecodeError:
                        mem_tags = []

                    memory = Memory(
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
                        tags=mem_tags,
                        source=metadata.get("source", "stated"),
                        confidence=float(metadata.get("confidence", 1.0)),
                        subtype=metadata.get("subtype"),
                    )

                # Apply tag filter (ChromaDB can't filter JSON arrays)
                if tags and not all(t in memory.tags for t in tags):
                    continue

                memories.append(memory)
                if len(memories) >= limit:
                    break

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

    def get_all_tags(self) -> dict[str, int]:
        """Get all tags with counts."""
        tag_counts: dict[str, int] = {}
        for memory in self._memories.values():
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return tag_counts


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
        tags: list[str] | None = None,
        source: str = "stated",
        confidence: float = 1.0,
        subtype: str | None = None,
    ) -> Memory:
        """
        Store a new memory.

        Args:
            content: The content to remember
            memory_type: Type of memory
            metadata: Optional additional data
            tags: Optional list of tags
            source: How the memory was acquired (stated/inferred/learned)
            confidence: Confidence level 0.0-1.0
            subtype: Optional subtype (e.g., "fact", "preference")

        Returns:
            The created Memory object
        """
        now = datetime.now()
        normalized_tags = [t.lower().strip() for t in (tags or []) if t.strip()]

        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            tags=normalized_tags,
            source=source,
            confidence=confidence,
            subtype=subtype,
        )

        self.store.store(memory)
        logger.info(f"Remembered {memory_type.value} memory: {content[:50]}...")
        return memory

    def remember_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        category: str = "fact",
        confidence: float = 1.0,
        source: str = "stated",
        tags: list[str] | None = None,
    ) -> Memory:
        """
        Store a structured semantic fact.

        Args:
            subject: The subject of the fact
            predicate: The relationship/verb
            obj: The object of the fact
            category: fact | preference | entity | relationship
            confidence: Confidence level
            source: How acquired
            tags: Optional tags

        Returns:
            The created Memory object
        """
        fact = SemanticFact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            category=category,
            confidence=confidence,
            source=source,
        )

        return self.remember(
            content=fact.to_content(),
            memory_type=MemoryType.SEMANTIC,
            metadata=fact.to_metadata(),
            tags=tags,
            source=source,
            confidence=confidence,
            subtype=category,
        )

    def remember_procedure(
        self,
        name: str,
        trigger: str,
        steps: list[str],
        tags: list[str] | None = None,
    ) -> Memory:
        """
        Store a procedural memory (workflow/pattern).

        Args:
            name: Name of the procedure
            trigger: What triggers this procedure
            steps: List of steps to execute
            tags: Optional tags

        Returns:
            The created Memory object
        """
        procedure = Procedure(name=name, trigger=trigger, steps=steps)

        return self.remember(
            content=procedure.to_content(),
            memory_type=MemoryType.PROCEDURAL,
            metadata=procedure.to_metadata(),
            tags=tags,
            source="stated",
            confidence=1.0,
            subtype="workflow",
        )

    def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]:
        """
        Retrieve relevant memories with optional filters.

        Args:
            query: What to search for
            memory_type: Optional filter by type
            tags: Optional filter by tags (all must match)
            source: Optional filter by source
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of relevant memories
        """
        return self.store.search(query, memory_type, tags, source, min_confidence, limit)

    def recall_by_tags(self, tags: list[str], limit: int = 10) -> list[Memory]:
        """Retrieve memories that have all specified tags."""
        all_memories = self.store.list_all()
        matching = [m for m in all_memories if all(t in m.tags for t in tags)]
        return matching[:limit]

    def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID or partial ID."""
        # Try exact match first
        memory = self.store.retrieve(memory_id)
        if memory:
            return memory
        # Try partial match
        for mem in self.store.list_all():
            if mem.id.startswith(memory_id):
                return mem
        return None

    def update_memory(self, memory: Memory) -> bool:
        """Update an existing memory."""
        memory.updated_at = datetime.now()
        return self.store.update(memory)

    def add_tag(self, memory_id: str, tag: str) -> bool:
        """Add a tag to a memory."""
        memory = self.get_memory(memory_id)
        if memory:
            memory.add_tag(tag)
            return self.update_memory(memory)
        return False

    def remove_tag(self, memory_id: str, tag: str) -> bool:
        """Remove a tag from a memory."""
        memory = self.get_memory(memory_id)
        if memory:
            if memory.remove_tag(tag):
                return self.update_memory(memory)
        return False

    def get_all_tags(self) -> dict[str, int]:
        """Get all tags with their usage counts."""
        return self.store.get_all_tags()

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        # Try partial match
        memory = self.get_memory(memory_id)
        if memory:
            return self.store.delete(memory.id)
        return False

    def save_conversation(self, conversation: Conversation) -> Memory:
        """Save a conversation as an episodic memory."""
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
            subtype="conversation",
        )

    # Export/Import functionality

    def export_memories(self, format: str = "json") -> str:
        """
        Export all memories to string format.

        Args:
            format: "json" or "jsonl"

        Returns:
            Exported data as string
        """
        memories = self.store.list_all()
        data = [m.to_dict() for m in memories]

        if format == "jsonl":
            return "\n".join(json.dumps(m) for m in data)
        else:
            return json.dumps(data, indent=2)

    def import_memories(self, data: str, format: str = "json") -> int:
        """
        Import memories from string data.

        Args:
            data: The data to import
            format: "json" or "jsonl"

        Returns:
            Number of memories imported
        """
        if format == "jsonl":
            items = [json.loads(line) for line in data.strip().split("\n") if line.strip()]
        else:
            items = json.loads(data)

        count = 0
        for item in items:
            memory = Memory.from_dict(item)
            self.store.store(memory)
            count += 1

        logger.info(f"Imported {count} memories")
        return count

    def backup(self, backup_path: Path) -> None:
        """
        Create a full backup of the data directory.

        Args:
            backup_path: Path for the backup archive
        """
        backup_path = Path(backup_path)
        if backup_path.suffix != ".zip":
            backup_path = backup_path.with_suffix(".zip")

        shutil.make_archive(str(backup_path.with_suffix("")), "zip", self.data_dir)
        logger.info(f"Created backup at {backup_path}")

    def get_statistics(self) -> dict:
        """Get memory statistics."""
        all_memories = self.store.list_all()
        tags = self.get_all_tags()

        by_type = {}
        by_source = {}
        by_subtype = {}
        total_confidence = 0.0

        for mem in all_memories:
            by_type[mem.memory_type.value] = by_type.get(mem.memory_type.value, 0) + 1
            by_source[mem.source] = by_source.get(mem.source, 0) + 1
            if mem.subtype:
                by_subtype[mem.subtype] = by_subtype.get(mem.subtype, 0) + 1
            total_confidence += mem.confidence

        return {
            "total": len(all_memories),
            "by_type": by_type,
            "by_source": by_source,
            "by_subtype": by_subtype,
            "avg_confidence": total_confidence / len(all_memories) if all_memories else 0,
            "unique_tags": len(tags),
            "top_tags": sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10],
        }

    def consolidate(self):
        """
        Consolidate memories - summarize and compress.

        TODO: Implement summarization logic
        """
        pass
