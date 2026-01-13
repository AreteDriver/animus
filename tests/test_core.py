"""
Tests for Animus core functionality.

Phase 0: Basic config, cognitive, memory, conversation tests
Phase 1: Tags, facts, procedures, export/import, statistics tests
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.cognitive import CognitiveLayer, ModelConfig, ModelProvider, ReasoningMode
from animus.config import AnimusConfig
from animus.memory import (
    Conversation,
    LocalMemoryStore,
    Memory,
    MemoryLayer,
    MemorySource,
    MemoryType,
    Message,
    Procedure,
    SemanticFact,
)


class TestAnimusConfig:
    """Tests for AnimusConfig."""

    def test_default_config(self):
        config = AnimusConfig()
        assert config.log_level == "INFO"
        assert config.data_dir == Path.home() / ".animus"

    def test_custom_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir), log_level="DEBUG")
            assert config.log_level == "DEBUG"
            assert config.data_dir == Path(tmpdir)

    def test_config_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir), log_level="DEBUG")
            config.model.provider = "anthropic"
            config.save()

            loaded = AnimusConfig.load(config.config_file)
            assert loaded.log_level == "DEBUG"
            assert loaded.model.provider == "anthropic"

    def test_paths(self):
        config = AnimusConfig()
        assert config.config_file == config.data_dir / "config.yaml"
        assert config.log_file == config.data_dir / "animus.log"
        assert config.chroma_dir == config.data_dir / "chroma"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_ollama_config(self):
        config = ModelConfig.ollama("llama3:8b")
        assert config.provider == ModelProvider.OLLAMA
        assert config.model_name == "llama3:8b"
        assert config.base_url == "http://localhost:11434"

    def test_anthropic_config(self):
        config = ModelConfig.anthropic("claude-3-haiku-20240307")
        assert config.provider == ModelProvider.ANTHROPIC
        assert config.model_name == "claude-3-haiku-20240307"


class TestCognitiveLayer:
    """Tests for CognitiveLayer."""

    def test_initialization(self):
        config = ModelConfig.ollama()
        layer = CognitiveLayer(primary_config=config)
        assert layer.primary_config.provider == ModelProvider.OLLAMA

    def test_build_system_prompt(self):
        layer = CognitiveLayer()
        prompt = layer._build_system_prompt(None, ReasoningMode.QUICK)
        assert "Animus" in prompt
        assert "personal AI assistant" in prompt

    def test_build_system_prompt_with_context(self):
        layer = CognitiveLayer()
        prompt = layer._build_system_prompt("User likes Python", ReasoningMode.QUICK)
        assert "User likes Python" in prompt

    def test_build_system_prompt_deep_mode(self):
        layer = CognitiveLayer()
        prompt = layer._build_system_prompt(None, ReasoningMode.DEEP)
        assert "think through this carefully" in prompt

    @patch("animus.cognitive.OllamaModel")
    def test_think_calls_model(self, mock_model_class):
        mock_model = MagicMock()
        mock_model.generate.return_value = "Test response"
        mock_model_class.return_value = mock_model

        layer = CognitiveLayer()
        layer.primary = mock_model

        result = layer.think("Hello")
        assert result == "Test response"
        mock_model.generate.assert_called_once()


class TestMemoryLayer:
    """Tests for memory functionality."""

    def test_remember_and_recall_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")

            # Store a memory
            mem = memory.remember("Python is a programming language", MemoryType.SEMANTIC)
            assert mem.id is not None
            assert mem.content == "Python is a programming language"

            # Recall it (substring search)
            results = memory.recall("Python")
            assert len(results) >= 1
            assert any("Python" in r.content for r in results)

    def test_forget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")

            # Store and then forget
            mem = memory.remember("Temporary thought")
            assert memory.forget(mem.id)

            # Should not find it
            results = memory.recall("Temporary")
            assert len(results) == 0

    def test_memory_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")

            # Store different types
            memory.remember("Meeting at 3pm", MemoryType.EPISODIC)
            memory.remember("User prefers dark mode", MemoryType.SEMANTIC)

            # List all
            all_mems = memory.store.list_all()
            assert len(all_mems) == 2

            # Filter by type
            episodic = memory.store.list_all(MemoryType.EPISODIC)
            assert len(episodic) == 1
            assert episodic[0].memory_type == MemoryType.EPISODIC


class TestLocalMemoryStore:
    """Tests for LocalMemoryStore."""

    def test_persistence(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and store
            store1 = LocalMemoryStore(Path(tmpdir))
            mem = Memory(
                id="test-id",
                content="Test content",
                memory_type=MemoryType.SEMANTIC,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                metadata={},
            )
            store1.store(mem)

            # Load in new instance
            store2 = LocalMemoryStore(Path(tmpdir))
            retrieved = store2.retrieve("test-id")
            assert retrieved is not None
            assert retrieved.content == "Test content"


class TestConversation:
    """Tests for Conversation."""

    def test_new_conversation(self):
        conv = Conversation.new()
        assert conv.id is not None
        assert len(conv.messages) == 0
        assert conv.started_at is not None

    def test_add_message(self):
        conv = Conversation.new()
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")

        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

    def test_to_memory_content(self):
        conv = Conversation.new()
        conv.add_message("user", "What is Python?")
        conv.add_message("assistant", "Python is a programming language.")

        content = conv.to_memory_content()
        assert "User: What is Python?" in content
        assert "Animus: Python is a programming language." in content

    def test_serialization(self):
        conv = Conversation.new()
        conv.add_message("user", "Test message")

        data = conv.to_dict()
        restored = Conversation.from_dict(data)

        assert restored.id == conv.id
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Test message"


class TestMessage:
    """Tests for Message."""

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_message_serialization(self):
        msg = Message(role="assistant", content="Hi!")
        data = msg.to_dict()
        restored = Message.from_dict(data)

        assert restored.role == msg.role
        assert restored.content == msg.content


# =============================================================================
# Phase 1 Tests: Memory Architecture
# =============================================================================


class TestMemoryPhase1Fields:
    """Tests for Phase 1 Memory fields (tags, source, confidence, subtype)."""

    def test_memory_default_fields(self):
        """Memory should have sensible defaults for Phase 1 fields."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        assert mem.tags == []
        assert mem.source == "stated"
        assert mem.confidence == 1.0
        assert mem.subtype is None

    def test_memory_with_all_fields(self):
        """Memory should accept all Phase 1 fields."""
        now = datetime.now()
        mem = Memory(
            id="test",
            content="User prefers Python",
            memory_type=MemoryType.SEMANTIC,
            created_at=now,
            updated_at=now,
            metadata={"key": "value"},
            tags=["python", "preference"],
            source="inferred",
            confidence=0.85,
            subtype="preference",
        )
        assert mem.tags == ["python", "preference"]
        assert mem.source == "inferred"
        assert mem.confidence == 0.85
        assert mem.subtype == "preference"

    def test_memory_to_dict_includes_phase1_fields(self):
        """to_dict should include all Phase 1 fields."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["tag1", "tag2"],
            source="learned",
            confidence=0.9,
            subtype="fact",
        )
        data = mem.to_dict()
        assert data["tags"] == ["tag1", "tag2"]
        assert data["source"] == "learned"
        assert data["confidence"] == 0.9
        assert data["subtype"] == "fact"

    def test_memory_from_dict_includes_phase1_fields(self):
        """from_dict should restore all Phase 1 fields."""
        data = {
            "id": "test",
            "content": "Test",
            "memory_type": "semantic",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
            "tags": ["important", "work"],
            "source": "inferred",
            "confidence": 0.75,
            "subtype": "entity",
        }
        mem = Memory.from_dict(data)
        assert mem.tags == ["important", "work"]
        assert mem.source == "inferred"
        assert mem.confidence == 0.75
        assert mem.subtype == "entity"

    def test_memory_from_dict_handles_missing_phase1_fields(self):
        """from_dict should handle legacy data without Phase 1 fields."""
        data = {
            "id": "test",
            "content": "Test",
            "memory_type": "semantic",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
            # No Phase 1 fields
        }
        mem = Memory.from_dict(data)
        assert mem.tags == []
        assert mem.source == "stated"
        assert mem.confidence == 1.0
        assert mem.subtype is None


class TestMemoryTags:
    """Tests for Memory tag methods."""

    def test_add_tag(self):
        """add_tag should add normalized tag."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        mem.add_tag("Python")
        assert "python" in mem.tags
        assert "Python" not in mem.tags  # Should be normalized

    def test_add_tag_no_duplicates(self):
        """add_tag should not add duplicate tags."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["python"],
        )
        mem.add_tag("python")
        mem.add_tag("PYTHON")
        assert mem.tags.count("python") == 1

    def test_add_tag_strips_whitespace(self):
        """add_tag should strip whitespace."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        mem.add_tag("  python  ")
        assert "python" in mem.tags

    def test_add_tag_ignores_empty(self):
        """add_tag should ignore empty tags."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
        )
        mem.add_tag("")
        mem.add_tag("   ")
        assert mem.tags == []

    def test_remove_tag(self):
        """remove_tag should remove tag and return True."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["python", "work"],
        )
        result = mem.remove_tag("python")
        assert result is True
        assert "python" not in mem.tags
        assert "work" in mem.tags

    def test_remove_tag_case_insensitive(self):
        """remove_tag should be case-insensitive."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["python"],
        )
        result = mem.remove_tag("PYTHON")
        assert result is True
        assert "python" not in mem.tags

    def test_remove_tag_not_found(self):
        """remove_tag should return False if tag not found."""
        mem = Memory(
            id="test",
            content="Test",
            memory_type=MemoryType.SEMANTIC,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={},
            tags=["python"],
        )
        result = mem.remove_tag("rust")
        assert result is False
        assert mem.tags == ["python"]


class TestSemanticFact:
    """Tests for SemanticFact dataclass."""

    def test_semantic_fact_creation(self):
        """SemanticFact should store subject-predicate-object."""
        fact = SemanticFact(
            subject="User",
            predicate="prefers",
            obj="dark mode",
        )
        assert fact.subject == "User"
        assert fact.predicate == "prefers"
        assert fact.obj == "dark mode"

    def test_semantic_fact_defaults(self):
        """SemanticFact should have sensible defaults."""
        fact = SemanticFact(subject="Python", predicate="is", obj="a programming language")
        assert fact.category == "fact"
        assert fact.confidence == 1.0
        assert fact.source == "stated"

    def test_semantic_fact_to_content(self):
        """to_content should generate natural language."""
        fact = SemanticFact(subject="User", predicate="prefers", obj="dark mode")
        content = fact.to_content()
        assert content == "User prefers dark mode"

    def test_semantic_fact_to_metadata(self):
        """to_metadata should include structured fields."""
        fact = SemanticFact(
            subject="Python",
            predicate="is",
            obj="versatile",
            category="preference",
        )
        metadata = fact.to_metadata()
        assert metadata["fact_subject"] == "Python"
        assert metadata["fact_predicate"] == "is"
        assert metadata["fact_object"] == "versatile"
        assert metadata["fact_category"] == "preference"


class TestProcedure:
    """Tests for Procedure dataclass."""

    def test_procedure_creation(self):
        """Procedure should store workflow data."""
        proc = Procedure(
            name="morning-standup",
            trigger="9am weekdays",
            steps=["Open Slack", "Check messages", "Join standup"],
        )
        assert proc.name == "morning-standup"
        assert proc.trigger == "9am weekdays"
        assert len(proc.steps) == 3

    def test_procedure_defaults(self):
        """Procedure should have sensible defaults."""
        proc = Procedure(name="test", trigger="always", steps=["step1"])
        assert proc.frequency == 0
        assert proc.last_used is None

    def test_procedure_to_content(self):
        """to_content should generate descriptive text."""
        proc = Procedure(
            name="deploy",
            trigger="release ready",
            steps=["Run tests", "Build image", "Push to registry"],
        )
        content = proc.to_content()
        assert "Procedure 'deploy'" in content
        assert "When release ready" in content
        assert "Run tests" in content

    def test_procedure_to_metadata(self):
        """to_metadata should include structured fields."""
        proc = Procedure(
            name="backup",
            trigger="midnight",
            steps=["Dump DB", "Compress", "Upload"],
        )
        metadata = proc.to_metadata()
        assert metadata["procedure_name"] == "backup"
        assert metadata["procedure_trigger"] == "midnight"
        assert "Dump DB" in metadata["procedure_steps"]

    def test_procedure_use(self):
        """use() should increment frequency and set last_used."""
        proc = Procedure(name="test", trigger="always", steps=["step1"])
        assert proc.frequency == 0
        assert proc.last_used is None

        proc.use()
        assert proc.frequency == 1
        assert proc.last_used is not None

        proc.use()
        assert proc.frequency == 2


class TestMemoryLayerPhase1:
    """Tests for MemoryLayer Phase 1 features."""

    def test_remember_with_tags(self):
        """remember() should accept and store tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            mem = memory.remember(
                "Python is great",
                tags=["python", "programming"],
            )
            assert "python" in mem.tags
            assert "programming" in mem.tags

    def test_remember_with_source_and_confidence(self):
        """remember() should accept source and confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            mem = memory.remember(
                "User seems to prefer CLI tools",
                source="inferred",
                confidence=0.7,
            )
            assert mem.source == "inferred"
            assert mem.confidence == 0.7

    def test_remember_fact(self):
        """remember_fact() should store structured fact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            mem = memory.remember_fact(
                subject="User",
                predicate="works at",
                obj="TechCorp",
                category="entity",
                tags=["work", "employer"],
            )
            assert mem.memory_type == MemoryType.SEMANTIC
            assert "User works at TechCorp" in mem.content
            assert mem.subtype == "entity"
            assert "work" in mem.tags
            assert mem.metadata["fact_subject"] == "User"

    def test_remember_procedure(self):
        """remember_procedure() should store workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            mem = memory.remember_procedure(
                name="deploy-app",
                trigger="release tagged",
                steps=["Build", "Test", "Deploy"],
                tags=["devops", "automation"],
            )
            assert mem.memory_type == MemoryType.PROCEDURAL
            assert "deploy-app" in mem.content
            assert mem.subtype == "workflow"
            assert "devops" in mem.tags

    def test_recall_with_tags_filter(self):
        """recall() should filter by tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("Python basics", tags=["python", "basics"])
            memory.remember("Rust basics", tags=["rust", "basics"])
            memory.remember("Python advanced", tags=["python", "advanced"])

            # Search with tag filter
            results = memory.recall("basics", tags=["python"])
            assert len(results) == 1
            assert "Python basics" in results[0].content

    def test_recall_with_source_filter(self):
        """recall() should filter by source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("User said they like Python", source="stated")
            memory.remember("User probably prefers CLI", source="inferred")

            results = memory.recall("User", source="inferred")
            assert len(results) == 1
            assert "CLI" in results[0].content

    def test_recall_with_min_confidence(self):
        """recall() should filter by minimum confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("High confidence fact", confidence=0.9)
            memory.remember("Low confidence guess", confidence=0.3)

            results = memory.recall("confidence", min_confidence=0.5)
            assert len(results) == 1
            assert "High" in results[0].content

    def test_recall_by_tags(self):
        """recall_by_tags() should find memories with all specified tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("Python web dev", tags=["python", "web"])
            memory.remember("Python CLI tool", tags=["python", "cli"])
            memory.remember("Rust web dev", tags=["rust", "web"])

            results = memory.recall_by_tags(["python", "web"])
            assert len(results) == 1
            assert "Python web dev" in results[0].content

    def test_get_memory_partial_id(self):
        """get_memory() should match partial IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            mem = memory.remember("Test content")
            partial_id = mem.id[:8]

            found = memory.get_memory(partial_id)
            assert found is not None
            assert found.id == mem.id

    def test_add_tag_via_layer(self):
        """MemoryLayer.add_tag() should add tag to memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            mem = memory.remember("Python stuff")

            result = memory.add_tag(mem.id, "programming")
            assert result is True

            updated = memory.get_memory(mem.id)
            assert "programming" in updated.tags

    def test_remove_tag_via_layer(self):
        """MemoryLayer.remove_tag() should remove tag from memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            mem = memory.remember("Python stuff", tags=["python", "work"])

            result = memory.remove_tag(mem.id, "work")
            assert result is True

            updated = memory.get_memory(mem.id)
            assert "work" not in updated.tags
            assert "python" in updated.tags

    def test_get_all_tags(self):
        """get_all_tags() should return tag counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("Item 1", tags=["python", "work"])
            memory.remember("Item 2", tags=["python", "personal"])
            memory.remember("Item 3", tags=["rust", "work"])

            tags = memory.get_all_tags()
            assert tags["python"] == 2
            assert tags["work"] == 2
            assert tags["rust"] == 1
            assert tags["personal"] == 1


class TestMemoryExportImport:
    """Tests for export/import functionality."""

    def test_export_json(self):
        """export_memories() should export to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("Memory 1", tags=["tag1"])
            memory.remember("Memory 2", tags=["tag2"])

            exported = memory.export_memories(format="json")
            data = json.loads(exported)
            assert len(data) == 2
            assert any("Memory 1" in m["content"] for m in data)

    def test_export_jsonl(self):
        """export_memories() should export to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("Memory 1")
            memory.remember("Memory 2")

            exported = memory.export_memories(format="jsonl")
            lines = exported.strip().split("\n")
            assert len(lines) == 2
            # Each line should be valid JSON
            for line in lines:
                json.loads(line)

    def test_import_json(self):
        """import_memories() should import from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source memory
            source = MemoryLayer(Path(tmpdir) / "source", backend="json")
            source.remember("Imported memory 1", tags=["imported"])
            source.remember("Imported memory 2", tags=["imported"])
            exported = source.export_memories(format="json")

            # Import to destination
            dest = MemoryLayer(Path(tmpdir) / "dest", backend="json")
            count = dest.import_memories(exported, format="json")
            assert count == 2

            # Verify import
            results = dest.recall("Imported")
            assert len(results) == 2

    def test_import_jsonl(self):
        """import_memories() should import from JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = MemoryLayer(Path(tmpdir) / "source", backend="json")
            source.remember("Line 1")
            source.remember("Line 2")
            exported = source.export_memories(format="jsonl")

            dest = MemoryLayer(Path(tmpdir) / "dest", backend="json")
            count = dest.import_memories(exported, format="jsonl")
            assert count == 2

    def test_export_preserves_phase1_fields(self):
        """Export should preserve all Phase 1 fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember(
                "Test",
                tags=["tag1", "tag2"],
                source="inferred",
                confidence=0.8,
                subtype="preference",
            )

            exported = memory.export_memories()
            data = json.loads(exported)[0]
            assert data["tags"] == ["tag1", "tag2"]
            assert data["source"] == "inferred"
            assert data["confidence"] == 0.8
            assert data["subtype"] == "preference"

    def test_backup_creates_zip(self):
        """backup() should create a zip file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "data", backend="json")
            memory.remember("Test memory")

            backup_path = Path(tmpdir) / "backup"
            memory.backup(backup_path)

            assert (Path(tmpdir) / "backup.zip").exists()


class TestMemoryStatistics:
    """Tests for get_statistics()."""

    def test_statistics_basic(self):
        """get_statistics() should return basic counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("Episodic 1", MemoryType.EPISODIC)
            memory.remember("Semantic 1", MemoryType.SEMANTIC)
            memory.remember("Semantic 2", MemoryType.SEMANTIC)

            stats = memory.get_statistics()
            assert stats["total"] == 3
            assert stats["by_type"]["episodic"] == 1
            assert stats["by_type"]["semantic"] == 2

    def test_statistics_by_source(self):
        """get_statistics() should count by source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("Stated 1", source="stated")
            memory.remember("Stated 2", source="stated")
            memory.remember("Inferred", source="inferred")

            stats = memory.get_statistics()
            assert stats["by_source"]["stated"] == 2
            assert stats["by_source"]["inferred"] == 1

    def test_statistics_avg_confidence(self):
        """get_statistics() should calculate average confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("High", confidence=1.0)
            memory.remember("Low", confidence=0.5)

            stats = memory.get_statistics()
            assert stats["avg_confidence"] == 0.75

    def test_statistics_top_tags(self):
        """get_statistics() should return top tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember("A", tags=["python"])
            memory.remember("B", tags=["python", "work"])
            memory.remember("C", tags=["python", "work"])
            memory.remember("D", tags=["rust"])

            stats = memory.get_statistics()
            assert stats["unique_tags"] == 3
            # Top tags should be sorted by count
            top = stats["top_tags"]
            assert top[0][0] == "python"
            assert top[0][1] == 3
            assert top[1][0] == "work"
            assert top[1][1] == 2


class TestMemorySourceEnum:
    """Tests for MemorySource enum."""

    def test_memory_source_values(self):
        """MemorySource should have expected values."""
        assert MemorySource.STATED.value == "stated"
        assert MemorySource.INFERRED.value == "inferred"
        assert MemorySource.LEARNED.value == "learned"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
