"""
Tests for Animus core functionality.
"""

import pytest
from pathlib import Path
import tempfile

from animus.core import Animus, AnimusConfig, create_animus
from animus.memory import MemoryLayer, MemoryType, Memory


class TestAnimusConfig:
    """Tests for AnimusConfig."""
    
    def test_default_config(self):
        config = AnimusConfig()
        assert config.instance_name == "animus_primary"
        assert config.log_level == "INFO"
        
    def test_custom_config(self):
        config = AnimusConfig(
            instance_name="test_instance",
            log_level="DEBUG"
        )
        assert config.instance_name == "test_instance"
        assert config.log_level == "DEBUG"


class TestAnimus:
    """Tests for main Animus class."""
    
    def test_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            animus = Animus(config)
            assert animus.config.instance_name == "animus_primary"
            
    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            animus = Animus(config)
            assert "animus_primary" in repr(animus)
            
    def test_chat_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            animus = Animus(config)
            response = animus.chat("Hello")
            assert "Hello" in response  # Placeholder echoes input


class TestMemoryLayer:
    """Tests for memory functionality."""
    
    def test_remember_and_recall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir))
            
            # Store a memory
            mem = memory.remember("Python is a programming language", MemoryType.SEMANTIC)
            assert mem.id is not None
            assert mem.content == "Python is a programming language"
            
            # Recall it
            results = memory.recall("Python")
            assert len(results) >= 1
            assert any("Python" in r.content for r in results)
            
    def test_forget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir))
            
            # Store and then forget
            mem = memory.remember("Temporary thought")
            assert memory.forget(mem.id)
            
            # Should not find it
            results = memory.recall("Temporary")
            assert len(results) == 0
            
    def test_memory_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir))
            
            # Store different types
            memory.remember("Meeting at 3pm", MemoryType.EPISODIC)
            memory.remember("User prefers dark mode", MemoryType.SEMANTIC)
            memory.remember("Always format code with black", MemoryType.PROCEDURAL)
            
            # Filter by type
            episodic = memory.recall("meeting", MemoryType.EPISODIC)
            assert len(episodic) >= 1
            assert all(m.memory_type == MemoryType.EPISODIC for m in episodic)


class TestCreateAnimus:
    """Tests for the create_animus factory function."""
    
    def test_create_without_config(self):
        animus = create_animus()
        assert isinstance(animus, Animus)
        
    def test_create_with_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a config file
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("""
core:
  instance_name: "test_from_file"
  log_level: "DEBUG"
""")
            
            animus = create_animus(config_path)
            assert animus.config.instance_name == "test_from_file"
            assert animus.config.log_level == "DEBUG"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
