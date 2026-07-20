"""Tests for Memory dataclass and related types.

Covers:
- Memory.create() factory
- Memory.to_dict() / from_dict() round-trip
- Memory.add_tag() / remove_tag()
- SemanticFact.to_content() / to_metadata()
- Procedure.to_content() / to_metadata() / use()
- Conversation.add_message() / to_memory_content() / to_dict() / from_dict()
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from animus_kernel.memory.types import (
    Conversation,
    Memory,
    MemoryTier,
    MemoryType,
    Message,
    Procedure,
    SemanticFact,
    Sensitivity,
)


class TestMemory:
    def test_create_generates_id_and_timestamps(self):
        before = datetime.now()
        mem = Memory.create(content="hello")
        after = datetime.now()

        assert mem.id
        assert isinstance(mem.id, str)
        assert before <= mem.created_at <= after
        assert mem.created_at == mem.updated_at
        assert mem.memory_type == MemoryType.SEMANTIC
        assert mem.tags == []
        assert mem.source == "stated"
        assert mem.confidence == 1.0
        assert mem.sensitivity == Sensitivity.PUBLIC
        assert mem.tier == MemoryTier.WARM
        assert mem.access_count == 0
        assert mem.last_accessed is None

    def test_create_with_all_fields(self):
        mem = Memory.create(
            content="test",
            memory_type=MemoryType.EPISODIC,
            metadata={"key": "value"},
            tags=["a", "b"],
            source="learned",
            confidence=0.75,
            subtype="conversation",
            version=3,
            parent_id="parent-123",
            change_summary="updated",
            provenance="consolidation",
            sensitivity=Sensitivity.CONFIDENTIAL,
            tier=MemoryTier.HOT,
            access_count=5,
            last_accessed=datetime(2024, 1, 1, 12, 0),
        )
        assert mem.content == "test"
        assert mem.memory_type == MemoryType.EPISODIC
        assert mem.metadata == {"key": "value"}
        assert mem.tags == ["a", "b"]
        assert mem.source == "learned"
        assert mem.confidence == 0.75
        assert mem.subtype == "conversation"
        assert mem.version == 3
        assert mem.parent_id == "parent-123"
        assert mem.change_summary == "updated"
        assert mem.provenance == "consolidation"
        assert mem.sensitivity == Sensitivity.CONFIDENTIAL
        assert mem.tier == MemoryTier.HOT
        assert mem.access_count == 5
        assert mem.last_accessed == datetime(2024, 1, 1, 12, 0)

    def test_to_dict_roundtrip(self):
        original = Memory.create(
            content="roundtrip",
            tags=["tag1"],
            sensitivity=Sensitivity.PERSONAL,
            tier=MemoryTier.COLD,
            last_accessed=datetime(2024, 6, 15, 10, 30),
        )
        d = original.to_dict()
        restored = Memory.from_dict(d)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.memory_type == original.memory_type
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at
        assert restored.metadata == original.metadata
        assert restored.tags == original.tags
        assert restored.source == original.source
        assert restored.confidence == original.confidence
        assert restored.sensitivity == original.sensitivity
        assert restored.tier == original.tier
        assert restored.access_count == original.access_count
        assert restored.last_accessed == original.last_accessed

    def test_from_dict_defaults(self):
        now = datetime.now().isoformat()
        mem = Memory.from_dict({
            "id": "123",
            "content": "minimal",
            "memory_type": "semantic",
            "created_at": now,
            "updated_at": now,
            "metadata": {},
        })
        assert mem.tags == []
        assert mem.source == "stated"
        assert mem.confidence == 1.0
        assert mem.version == 1
        assert mem.parent_id is None
        assert mem.sensitivity == Sensitivity.PUBLIC
        assert mem.tier == MemoryTier.WARM
        assert mem.access_count == 0
        assert mem.last_accessed is None

    def test_add_tag_normalizes_and_dedupes(self):
        mem = Memory.create(content="tags")
        mem.add_tag("Hello")
        mem.add_tag("  hello  ")
        mem.add_tag("world")
        assert mem.tags == ["hello", "world"]

    def test_add_tag_empty_ignored(self):
        mem = Memory.create(content="tags")
        mem.add_tag("")
        mem.add_tag("   ")
        assert mem.tags == []

    def test_remove_tag(self):
        mem = Memory.create(content="tags", tags=["a", "b"])
        removed = mem.remove_tag("A")
        assert removed is True
        assert mem.tags == ["b"]
        removed_again = mem.remove_tag("a")
        assert removed_again is False

    def test_to_dict_serializes_enums(self):
        mem = Memory.create(content="enum", sensitivity=Sensitivity.SECRET, tier=MemoryTier.HOT)
        d = mem.to_dict()
        assert d["sensitivity"] == "secret"
        assert d["tier"] == "hot"
        assert d["memory_type"] == "semantic"

    def test_to_dict_with_none_last_accessed(self):
        mem = Memory.create(content="none")
        d = mem.to_dict()
        assert d["last_accessed"] is None


class TestSemanticFact:
    def test_to_content(self):
        fact = SemanticFact(subject="Alice", predicate="likes", obj="tea")
        assert fact.to_content() == "Alice likes tea"

    def test_to_metadata(self):
        fact = SemanticFact(
            subject="Alice",
            predicate="likes",
            obj="tea",
            category="preference",
            confidence=0.9,
            source="inferred",
        )
        meta = fact.to_metadata()
        assert meta["fact_subject"] == "Alice"
        assert meta["fact_predicate"] == "likes"
        assert meta["fact_object"] == "tea"
        assert meta["fact_category"] == "preference"


class TestProcedure:
    def test_to_content(self):
        proc = Procedure(
            name="Deploy",
            trigger="push to main",
            steps=["run tests", "build image", "deploy"],
        )
        content = proc.to_content()
        assert "Deploy" in content
        assert "push to main" in content
        assert "1. run tests" in content
        assert "2. build image" in content
        assert "3. deploy" in content

    def test_to_metadata(self):
        proc = Procedure(name="Test", trigger="commit", steps=["step1"])
        meta = proc.to_metadata()
        assert meta["procedure_name"] == "Test"
        assert json.loads(meta["procedure_steps"]) == ["step1"]
        assert meta["procedure_last_used"] is None

    def test_use_increments_frequency(self):
        proc = Procedure(name="X", trigger="Y", steps=["z"])
        assert proc.frequency == 0
        proc.use()
        assert proc.frequency == 1
        assert proc.last_used is not None


class TestConversation:
    def test_add_message(self):
        conv = Conversation.new()
        msg = conv.add_message("user", "hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert len(conv.messages) == 1

    def test_to_memory_content(self):
        conv = Conversation.new()
        conv.add_message("user", "hi")
        conv.add_message("assistant", "hello there")
        conv.ended_at = datetime.now()
        content = conv.to_memory_content()
        assert "User: hi" in content
        assert "Animus: hello there" in content

    def test_to_dict_roundtrip(self):
        conv = Conversation.new()
        conv.add_message("user", "q")
        d = conv.to_dict()
        restored = Conversation.from_dict(d)
        assert restored.id == conv.id
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "q"

    def test_message_to_dict(self):
        msg = Message(role="user", content="test", timestamp=datetime(2024, 1, 1))
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "test"
        assert d["timestamp"] == "2024-01-01T00:00:00"


class TestMemoryTypeAndSensitivity:
    def test_memory_type_values(self):
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.ACTIVE.value == "active"

    def test_sensitivity_values(self):
        assert Sensitivity.PUBLIC.value == "public"
        assert Sensitivity.PERSONAL.value == "personal"
        assert Sensitivity.CONFIDENTIAL.value == "confidential"
        assert Sensitivity.SECRET.value == "secret"

    def test_memory_tier_values(self):
        assert MemoryTier.HOT.value == "hot"
        assert MemoryTier.WARM.value == "warm"
        assert MemoryTier.COLD.value == "cold"
