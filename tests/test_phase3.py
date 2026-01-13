"""
Tests for Phase 3: Multi-Interface (Voice + API)

API Server, Voice I/O, Configuration
"""

import tempfile
from pathlib import Path

import pytest

from animus.config import AnimusConfig, APIConfig, VoiceConfig

# =============================================================================
# API Config Tests
# =============================================================================


class TestAPIConfig:
    """Tests for APIConfig dataclass."""

    def test_api_config_defaults(self):
        config = APIConfig()
        assert config.enabled is False
        assert config.host == "127.0.0.1"
        assert config.port == 8420
        assert config.api_key is None

    def test_api_config_custom(self):
        config = APIConfig(enabled=True, host="0.0.0.0", port=9000, api_key="secret")
        assert config.enabled is True
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.api_key == "secret"


class TestVoiceConfig:
    """Tests for VoiceConfig dataclass."""

    def test_voice_config_defaults(self):
        config = VoiceConfig()
        assert config.input_enabled is False
        assert config.output_enabled is False
        assert config.whisper_model == "base"
        assert config.tts_engine == "pyttsx3"
        assert config.tts_rate == 150

    def test_voice_config_custom(self):
        config = VoiceConfig(
            input_enabled=True,
            output_enabled=True,
            whisper_model="small",
            tts_engine="edge-tts",
            tts_rate=200,
        )
        assert config.input_enabled is True
        assert config.output_enabled is True
        assert config.whisper_model == "small"
        assert config.tts_engine == "edge-tts"
        assert config.tts_rate == 200


class TestAnimusConfigPhase3:
    """Tests for AnimusConfig with Phase 3 additions."""

    def test_config_has_api_section(self):
        config = AnimusConfig()
        assert hasattr(config, "api")
        assert isinstance(config.api, APIConfig)

    def test_config_has_voice_section(self):
        config = AnimusConfig()
        assert hasattr(config, "voice")
        assert isinstance(config.voice, VoiceConfig)

    def test_config_to_dict_includes_api(self):
        config = AnimusConfig()
        data = config.to_dict()
        assert "api" in data
        assert data["api"]["enabled"] is False
        assert data["api"]["port"] == 8420

    def test_config_to_dict_includes_voice(self):
        config = AnimusConfig()
        data = config.to_dict()
        assert "voice" in data
        assert data["voice"]["whisper_model"] == "base"
        assert data["voice"]["tts_engine"] == "pyttsx3"

    def test_config_save_load_preserves_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            config.api.enabled = True
            config.api.port = 9999
            config.save()

            loaded = AnimusConfig.load(config.config_file)
            assert loaded.api.enabled is True
            assert loaded.api.port == 9999

    def test_config_save_load_preserves_voice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig(data_dir=Path(tmpdir))
            config.voice.whisper_model = "large"
            config.voice.tts_rate = 200
            config.save()

            loaded = AnimusConfig.load(config.config_file)
            assert loaded.voice.whisper_model == "large"
            assert loaded.voice.tts_rate == 200


# =============================================================================
# API Model Tests (Pydantic models, always available)
# =============================================================================


class TestAPIModels:
    """Tests for API request/response Pydantic models."""

    def test_chat_request_model(self):
        from animus.api import ChatRequest

        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.mode == "auto"
        assert req.conversation_id is None

    def test_chat_response_model(self):
        from animus.api import ChatResponse

        resp = ChatResponse(
            response="Hi there",
            conversation_id="test-id",
            mode_used="quick",
        )
        assert resp.response == "Hi there"
        assert resp.conversation_id == "test-id"
        assert resp.mode_used == "quick"

    def test_memory_create_model(self):
        from animus.api import MemoryCreate

        req = MemoryCreate(content="Test memory")
        assert req.content == "Test memory"
        assert req.memory_type == "semantic"
        assert req.tags == []
        assert req.confidence == 1.0

    def test_memory_response_model(self):
        from animus.api import MemoryResponse

        resp = MemoryResponse(
            id="test-id",
            content="Test content",
            memory_type="semantic",
            tags=["tag1"],
            source="stated",
            confidence=0.9,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        assert resp.id == "test-id"
        assert resp.confidence == 0.9

    def test_task_create_model(self):
        from animus.api import TaskCreate

        req = TaskCreate(description="Do something")
        assert req.description == "Do something"
        assert req.tags == []
        assert req.priority == 0

    def test_task_response_model(self):
        from animus.api import TaskResponse

        resp = TaskResponse(
            id="test-id",
            description="Do something",
            status="pending",
            tags=["work"],
            priority=1,
            created_at="2024-01-01T00:00:00",
        )
        assert resp.id == "test-id"
        assert resp.status == "pending"

    def test_decision_request_model(self):
        from animus.api import DecisionRequest

        req = DecisionRequest(question="Which option?")
        assert req.question == "Which option?"
        assert req.options is None
        assert req.criteria is None

    def test_status_response_model(self):
        from animus.api import StatusResponse

        resp = StatusResponse(
            status="running",
            version="0.3.0",
            memory_count=10,
            task_count=5,
            model_provider="ollama",
            model_name="llama3:8b",
        )
        assert resp.status == "running"
        assert resp.memory_count == 10


# =============================================================================
# API Server Class Tests
# =============================================================================


class TestAPIServerClass:
    """Tests for APIServer class (without actually starting server)."""

    def test_api_server_import_error_when_fastapi_missing(self, monkeypatch):
        """Test that APIServer raises ImportError without FastAPI."""
        # Import the module
        import animus.api as api_module

        # Save original value
        original = api_module.FASTAPI_AVAILABLE

        try:
            # Pretend FastAPI isn't available
            monkeypatch.setattr(api_module, "FASTAPI_AVAILABLE", False)

            # Trying to create APIServer should raise ImportError
            with pytest.raises(ImportError) as exc_info:
                api_module.APIServer(
                    memory=None,
                    cognitive=None,
                    tools=None,
                    tasks=None,
                    decisions=None,
                )

            assert "FastAPI not installed" in str(exc_info.value)
        finally:
            # Restore original value
            monkeypatch.setattr(api_module, "FASTAPI_AVAILABLE", original)


# =============================================================================
# Voice Module Tests (without actual audio)
# =============================================================================


class TestVoiceModuleStructure:
    """Tests for voice module structure (no actual audio required)."""

    def test_voice_input_class_exists(self):
        from animus.voice import VoiceInput

        assert VoiceInput is not None

    def test_voice_output_class_exists(self):
        from animus.voice import VoiceOutput

        assert VoiceOutput is not None

    def test_voice_interface_class_exists(self):
        from animus.voice import VoiceInterface

        assert VoiceInterface is not None

    def test_voice_input_init(self):
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        assert vi.model_name == "base"
        assert vi._model is None  # Lazy loading

    def test_voice_output_init(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3", rate=150)
        assert vo.engine_name == "pyttsx3"
        assert vo.rate == 150

    def test_voice_input_is_listening_default(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        assert vi.is_listening is False

    def test_voice_output_is_speaking_default(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        assert vo.is_speaking is False


# =============================================================================
# Version Tests
# =============================================================================


class TestVersion:
    """Test version is at least Phase 3."""

    def test_version_is_at_least_0_3_0(self):
        from animus import __version__

        # Version should be at least 0.3.0 (Phase 3 introduced voice and API)
        major, minor, patch = map(int, __version__.split("."))
        assert (major, minor) >= (0, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
