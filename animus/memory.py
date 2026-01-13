"""
Animus Memory Layer

Handles persistence of context, knowledge, and patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import json


class MemoryType(Enum):
    """Types of memory in the system."""
    EPISODIC = "episodic"      # What happened (conversations, events)
    SEMANTIC = "semantic"       # What you know (facts, knowledge)
    PROCEDURAL = "procedural"   # How you do things (workflows, patterns)
    ACTIVE = "active"           # Current context (live state)


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


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def store(self, memory: Memory) -> None:
        """Store a memory."""
        pass
    
    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        pass
    
    @abstractmethod
    def search(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 10) -> list[Memory]:
        """Search memories by content."""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        pass
    
    @abstractmethod
    def list_all(self, memory_type: Optional[MemoryType] = None) -> list[Memory]:
        """List all memories, optionally filtered by type."""
        pass


class LocalMemoryStore(MemoryStore):
    """
    Simple local file-based memory store.
    
    For Phase 0 - will be replaced/augmented with vector DB in Phase 1.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.memories_file = data_dir / "memories.json"
        self._memories: dict[str, Memory] = {}
        self._load()
    
    def _load(self):
        """Load memories from disk."""
        if self.memories_file.exists():
            with open(self.memories_file) as f:
                data = json.load(f)
                self._memories = {
                    k: Memory.from_dict(v) for k, v in data.items()
                }
    
    def _save(self):
        """Save memories to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.memories_file, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._memories.items()},
                f,
                indent=2
            )
    
    def store(self, memory: Memory) -> None:
        self._memories[memory.id] = memory
        self._save()
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        return self._memories.get(memory_id)
    
    def search(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 10) -> list[Memory]:
        """Simple substring search - will be replaced with semantic search."""
        results = []
        query_lower = query.lower()
        
        for memory in self._memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            if query_lower in memory.content.lower():
                results.append(memory)
            if len(results) >= limit:
                break
                
        return results
    
    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save()
            return True
        return False
    
    def list_all(self, memory_type: Optional[MemoryType] = None) -> list[Memory]:
        if memory_type:
            return [m for m in self._memories.values() if m.memory_type == memory_type]
        return list(self._memories.values())


class MemoryLayer:
    """
    Main memory layer interface.
    
    Coordinates between different memory types and storage backends.
    """
    
    def __init__(self, data_dir: Path):
        self.store = LocalMemoryStore(data_dir)
        
    def remember(self, content: str, memory_type: MemoryType = MemoryType.SEMANTIC, metadata: Optional[dict] = None) -> Memory:
        """
        Store a new memory.
        
        Args:
            content: The content to remember
            memory_type: Type of memory
            metadata: Optional additional data
            
        Returns:
            The created Memory object
        """
        import uuid
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
        return memory
    
    def recall(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 10) -> list[Memory]:
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
    
    def consolidate(self):
        """
        Consolidate memories - summarize and compress.
        
        TODO: Implement in Phase 1
        """
        pass
