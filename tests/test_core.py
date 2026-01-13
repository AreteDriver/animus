"""
Tests for Animus core functionality.
"""

import tempfile
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
    MemoryType,
    Message,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
