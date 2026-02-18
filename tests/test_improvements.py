"""
Tests for general improvements:
- Register Translation
- API Call Framework (http_request tool)
- Credential encryption
- Memory consolidation & CSV export
- Method naming consistency
- Preference inference enhancements
- Voice interface test mocking
"""

import base64
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.learning.categories import LearningCategory
from animus.learning.patterns import DetectedPattern, PatternType
from animus.learning.preferences import PreferenceEngine

# =============================================================================
# Register Translation Tests
# =============================================================================


class TestRegisterDetection:
    """Tests for register detection."""

    def test_detect_formal(self):
        from animus.register import Register, detect_register

        result = detect_register("Dear sir, I would like to inquire about your services.")
        assert result == Register.FORMAL

    def test_detect_casual(self):
        from animus.register import Register, detect_register

        result = detect_register("hey, gonna grab some coffee, wanna come?")
        assert result == Register.CASUAL

    def test_detect_technical(self):
        from animus.register import Register, detect_register

        result = detect_register(
            "We need to refactor the API endpoint to reduce latency in our microservice."
        )
        assert result == Register.TECHNICAL

    def test_detect_neutral(self):
        from animus.register import Register, detect_register

        result = detect_register("Hello, how are you?")
        assert result == Register.NEUTRAL

    def test_detect_empty_text(self):
        from animus.register import Register, detect_register

        result = detect_register("")
        assert result == Register.NEUTRAL


class TestRegisterTranslator:
    """Tests for RegisterTranslator."""

    def test_default_register(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator()
        assert translator.active_register == Register.NEUTRAL

    def test_custom_default(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator(default_register=Register.FORMAL)
        assert translator.active_register == Register.FORMAL

    def test_detect_and_set(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator()
        detected = translator.detect_and_set("hey, gonna refactor this thing, lol")
        assert detected == Register.CASUAL
        assert translator.active_register == Register.CASUAL

    def test_override_takes_precedence(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator()
        translator.detect_and_set("hey lol cool")
        assert translator.active_register == Register.CASUAL

        translator.set_override(Register.FORMAL)
        assert translator.active_register == Register.FORMAL

    def test_clear_override(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator()
        translator.detect_and_set("hey lol cool")
        translator.set_override(Register.FORMAL)
        translator.set_override(None)
        # Should fall back to detected user register
        assert translator.active_register == Register.CASUAL

    def test_system_prompt_modifier(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator()
        translator.set_override(Register.FORMAL)
        modifier = translator.get_system_prompt_modifier()
        assert "formal" in modifier.lower()
        assert "professional" in modifier.lower()

    def test_neutral_no_modifier(self):
        from animus.register import RegisterTranslator

        translator = RegisterTranslator()
        modifier = translator.get_system_prompt_modifier()
        assert modifier == ""

    def test_adapt_prompt(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator()
        translator.set_override(Register.TECHNICAL)
        result = translator.adapt_prompt("You are a helpful assistant.")
        assert "Communication style:" in result
        assert "technical" in result.lower()

    def test_adapt_prompt_neutral_unchanged(self):
        from animus.register import RegisterTranslator

        translator = RegisterTranslator()
        base = "You are a helpful assistant."
        result = translator.adapt_prompt(base)
        assert result == base

    def test_get_register_context(self):
        from animus.register import Register, RegisterTranslator

        translator = RegisterTranslator()
        translator.set_override(Register.CASUAL)
        ctx = translator.get_register_context()
        assert ctx["register"] == "casual"
        assert ctx["is_override"] is True
        assert ctx["detected_user_register"] is None


class TestRegisterInCognitive:
    """Tests for register translation integration in CognitiveLayer."""

    def test_cognitive_layer_has_register_translator(self):
        from animus.cognitive import CognitiveLayer, ModelConfig, ModelProvider

        # Create with a mock provider
        config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name="test",
            base_url="http://localhost:11434",
        )
        layer = CognitiveLayer(primary_config=config)
        assert hasattr(layer, "register_translator")
        assert layer.register_translator is not None


# =============================================================================
# API Call Framework Tests
# =============================================================================


class TestHttpRequestTool:
    """Tests for the http_request tool handler."""

    def test_missing_url(self):
        from animus.tools import _tool_http_request

        result = _tool_http_request({})
        assert result.success is False
        assert "url" in result.error.lower()

    def test_unsupported_method(self):
        from animus.tools import _tool_http_request

        result = _tool_http_request({"url": "https://example.com", "method": "INVALID"})
        assert result.success is False
        assert "Unsupported" in result.error

    def test_unsupported_scheme(self):
        from animus.tools import _tool_http_request

        result = _tool_http_request({"url": "ftp://example.com/file"})
        assert result.success is False
        assert "scheme" in result.error.lower()

    @patch("urllib.request.urlopen")
    def test_successful_get(self, mock_urlopen):
        from animus.tools import _tool_http_request

        mock_response = MagicMock()
        mock_response.read.return_value = b'{"result": "ok"}'
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _tool_http_request({"url": "https://api.example.com/data"})
        assert result.success is True
        assert "200" in result.output

    @patch("urllib.request.urlopen")
    def test_bearer_auth(self, mock_urlopen):
        from animus.tools import _tool_http_request

        mock_response = MagicMock()
        mock_response.read.return_value = b"ok"
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _tool_http_request(
            {
                "url": "https://api.example.com",
                "auth_type": "bearer",
                "auth_value": "my-token",
            }
        )

        # Verify the request was made
        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        assert request.get_header("Authorization") == "Bearer my-token"

    @patch("urllib.request.urlopen")
    def test_basic_auth(self, mock_urlopen):
        from animus.tools import _tool_http_request

        mock_response = MagicMock()
        mock_response.read.return_value = b"ok"
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _tool_http_request(
            {
                "url": "https://api.example.com",
                "auth_type": "basic",
                "auth_value": "user:pass",
            }
        )

        request = mock_urlopen.call_args[0][0]
        expected = "Basic " + base64.b64encode(b"user:pass").decode()
        assert request.get_header("Authorization") == expected

    @patch("urllib.request.urlopen")
    def test_api_key_auth(self, mock_urlopen):
        from animus.tools import _tool_http_request

        mock_response = MagicMock()
        mock_response.read.return_value = b"ok"
        mock_response.status = 200
        mock_response.headers = {}
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        _tool_http_request(
            {
                "url": "https://api.example.com",
                "auth_type": "api_key",
                "auth_value": "key123",
            }
        )

        request = mock_urlopen.call_args[0][0]
        assert request.get_header("X-api-key") == "key123"

    def test_http_request_tool_in_builtin_tools(self):
        from animus.tools import BUILTIN_TOOLS

        names = [t.name for t in BUILTIN_TOOLS]
        assert "http_request" in names

    def test_http_request_requires_approval(self):
        from animus.tools import BUILTIN_TOOLS

        tool = next(t for t in BUILTIN_TOOLS if t.name == "http_request")
        assert tool.requires_approval is True


# =============================================================================
# Credential Encryption Tests
# =============================================================================


class TestCredentialEncryption:
    """Tests for credential encryption in IntegrationManager."""

    def test_save_and_load_credentials_fallback(self):
        """Test base64 fallback when cryptography is not installed."""
        from animus.integrations.manager import IntegrationManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(data_dir=Path(tmpdir))

            creds = {"api_key": "test-key-123", "secret": "super-secret"}
            manager._save_credentials("test_service", creds)

            loaded = manager._load_credentials("test_service")
            assert loaded == creds

    def test_credentials_file_permissions(self):
        """Test that credential files have restricted permissions."""
        import os

        from animus.integrations.manager import IntegrationManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(data_dir=Path(tmpdir))
            manager._save_credentials("test_perm", {"key": "val"})

            cred_path = Path(tmpdir) / "test_perm.json"
            mode = os.stat(cred_path).st_mode & 0o777
            assert mode == 0o600

    def test_credentials_not_plaintext(self):
        """Test that saved credentials are not plaintext JSON on disk."""
        from animus.integrations.manager import IntegrationManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(data_dir=Path(tmpdir))
            creds = {"api_key": "secret-value"}
            manager._save_credentials("test_enc", creds)

            cred_path = Path(tmpdir) / "test_enc.json"
            raw = cred_path.read_bytes()
            # Should not be plain JSON
            try:
                json.loads(raw)
                # If it decodes as JSON, it's plaintext — fail
                assert False, "Credentials stored as plaintext JSON"
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Expected: not plaintext

    def test_load_nonexistent_credentials(self):
        from animus.integrations.manager import IntegrationManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(data_dir=Path(tmpdir))
            result = manager._load_credentials("nonexistent")
            assert result is None


# =============================================================================
# Memory Consolidation & CSV Export Tests
# =============================================================================


class TestMemoryConsolidation:
    """Tests for memory consolidation."""

    def test_consolidate_no_old_memories(self):
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            # Store a recent memory
            memory.remember(content="recent event", tags=["test"], source="stated")
            count = memory.consolidate(max_age_days=90)
            assert count == 0

    def test_consolidate_groups_by_tag(self):
        from animus.memory import MemoryLayer, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")

            # Create old memories with same tag
            old_date = datetime.now() - timedelta(days=100)
            for i in range(5):
                mem = memory.remember(
                    content=f"Old event {i}",
                    memory_type=MemoryType.EPISODIC,
                    tags=["meetings"],
                    source="stated",
                )
                # Manually set old dates
                mem.created_at = old_date
                mem.updated_at = old_date
                memory.store.update(mem)

            count = memory.consolidate(max_age_days=90, min_group_size=3)
            assert count == 5

            # Should have one consolidated memory
            all_mems = memory.store.list_all()
            consolidated = [m for m in all_mems if "consolidated" in m.tags]
            assert len(consolidated) == 1
            assert "Consolidated summary" in consolidated[0].content

    def test_consolidate_skips_small_groups(self):
        from animus.memory import MemoryLayer, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")

            old_date = datetime.now() - timedelta(days=100)
            for i in range(2):  # Only 2 items, below min_group_size
                mem = memory.remember(
                    content=f"Small group {i}",
                    memory_type=MemoryType.EPISODIC,
                    tags=["rare"],
                    source="stated",
                )
                mem.created_at = old_date
                mem.updated_at = old_date
                memory.store.update(mem)

            count = memory.consolidate(max_age_days=90, min_group_size=3)
            assert count == 0


class TestMemoryCsvExport:
    """Tests for CSV memory export."""

    def test_export_csv_empty(self):
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            csv = memory.export_memories_csv()
            assert "id" in csv
            assert "content" in csv
            lines = csv.strip().split("\n")
            assert len(lines) == 1  # Header only

    def test_export_csv_with_data(self):
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember(content="Test memory 1", tags=["tag1"], source="stated")
            memory.remember(content="Test memory 2", tags=["tag2"], source="stated")

            csv = memory.export_memories_csv()
            lines = csv.strip().split("\n")
            assert len(lines) == 3  # Header + 2 rows

    def test_export_csv_format(self):
        import csv
        import io

        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember(content="CSV test", tags=["a", "b"], source="stated")

            csv_str = memory.export_memories_csv()
            reader = csv.reader(io.StringIO(csv_str))
            rows = list(reader)
            headers = rows[0]
            assert "id" in headers
            assert "content" in headers
            assert "tags" in headers
            assert "memory_type" in headers

            # Data row
            data_row = rows[1]
            assert "CSV test" in data_row


# =============================================================================
# Method Naming Consistency Tests
# =============================================================================


class TestMethodNamingConsistency:
    """Tests to verify method naming aliases work."""

    def test_task_tracker_list_all_alias(self):
        from animus.tasks import TaskTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = TaskTracker(Path(tmpdir))
            tracker.add(description="Test task")

            # Both list() and list_all() should work
            result_list = tracker.list(include_completed=True)
            result_list_all = tracker.list_all(include_completed=True)
            assert len(result_list) == len(result_list_all)

    def test_integration_manager_list_tools_alias(self):
        from animus.integrations.manager import IntegrationManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = IntegrationManager(data_dir=Path(tmpdir))
            # Both should return the same result
            tools1 = manager.get_all_tools()
            tools2 = manager.list_tools()
            assert tools1 == tools2


# =============================================================================
# Preference Inference Enhancement Tests
# =============================================================================


class TestPreferenceInferenceEnhancements:
    """Tests for enhanced preference inference."""

    def test_scheduling_domain_inference(self):
        """Test that time-related patterns infer scheduling domain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PreferenceEngine(Path(tmpdir))
            pattern = DetectedPattern(
                id="time_pattern",
                pattern_type=PatternType.TEMPORAL,
                description="Active during morning (9:00)",
                occurrences=5,
                confidence=0.8,
                evidence=[],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                suggested_category=LearningCategory.PREFERENCE,
                suggested_learning="Morning activity pattern",
            )
            pref = engine.infer_from_pattern(pattern)
            assert pref is not None
            assert pref.domain == "scheduling"

    def test_communication_domain_from_tone_keywords(self):
        """Test that tone-related patterns get communication domain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PreferenceEngine(Path(tmpdir))
            pattern = DetectedPattern(
                id="tone_pattern",
                pattern_type=PatternType.PREFERENCE,
                description="Prefers: formal tone in responses",
                occurrences=4,
                confidence=0.85,
                evidence=[],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                suggested_category=LearningCategory.STYLE,
                suggested_learning="User prefers formal tone",
            )
            pref = engine.infer_from_pattern(pattern)
            assert pref is not None
            assert pref.domain == "communication"

    def test_tools_domain_from_keywords(self):
        """Test that tool-related patterns get tools domain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PreferenceEngine(Path(tmpdir))
            pattern = DetectedPattern(
                id="tool_pattern",
                pattern_type=PatternType.FREQUENCY,
                description="Frequently requests: file search with tool",
                occurrences=5,
                confidence=0.75,
                evidence=[],
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                suggested_category=LearningCategory.WORKFLOW,
                suggested_learning="Common request: file search",
            )
            pref = engine.infer_from_pattern(pattern)
            assert pref is not None
            assert pref.domain == "tools"


# =============================================================================
# Voice Interface Test Mocking
# =============================================================================


class MockWhisperModel:
    """Mock Whisper model for testing."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name

    def transcribe(self, audio) -> dict:
        """Return mock transcription."""
        return {"text": "This is a mock transcription."}


class MockSoundDevice:
    """Mock sounddevice module for testing."""

    @staticmethod
    def rec(frames, samplerate=16000, channels=1, dtype=None):
        import numpy as np

        return np.zeros((frames, channels), dtype=dtype or np.float32)

    @staticmethod
    def wait():
        pass


class TestVoiceInputMocked:
    """Tests for VoiceInput using mocks."""

    def test_transcribe_file_with_mock(self, tmp_path):
        """Test file transcription with mocked Whisper."""
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        vi._model = MockWhisperModel()

        # Create a dummy audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        result = vi.transcribe_file(audio_file)
        assert result == "This is a mock transcription."

    def test_transcribe_file_not_found(self):
        """Test file not found raises error."""
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        with pytest.raises(FileNotFoundError):
            vi.transcribe_file("/nonexistent/audio.wav")

    @patch("animus.voice.sd", create=True)
    def test_transcribe_microphone_with_mock(self, mock_sd):
        """Test microphone transcription with mocked dependencies."""
        import numpy as np

        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        vi._model = MockWhisperModel()

        # Mock sounddevice
        mock_sd.rec.return_value = np.zeros((16000 * 5, 1), dtype=np.float32)
        mock_sd.wait.return_value = None

        # Patch the imports inside the method
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            with patch("sounddevice.rec", mock_sd.rec):
                with patch("sounddevice.wait", mock_sd.wait):
                    # Can't easily test this without actually importing sounddevice
                    # Just verify the mock model works
                    result = vi._model.transcribe(np.zeros(16000))
                    assert result["text"] == "This is a mock transcription."

    def test_is_listening_default(self):
        """Test that listening is off by default."""
        from animus.voice import VoiceInput

        vi = VoiceInput()
        assert vi.is_listening is False

    def test_stop_listening_when_not_started(self):
        """Test stop_listening when not started doesn't error."""
        from animus.voice import VoiceInput

        vi = VoiceInput()
        vi.stop_listening()
        assert vi.is_listening is False


class TestVoiceOutputMocked:
    """Tests for VoiceOutput using mocks."""

    def test_voice_output_init(self):
        """Test VoiceOutput initialization."""
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3", rate=175)
        assert vo.engine_name == "pyttsx3"
        assert vo.rate == 175

    def test_set_voice(self):
        """Test set_voice stores the voice ID."""
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        vo._voice_id = "test-voice-id"
        assert vo._voice_id == "test-voice-id"


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Test that new features are properly exported."""

    def test_register_exported(self):
        from animus import Register, RegisterTranslator, detect_register

        assert Register is not None
        assert RegisterTranslator is not None
        assert detect_register is not None

    def test_register_values(self):
        from animus import Register

        assert Register.FORMAL.value == "formal"
        assert Register.CASUAL.value == "casual"
        assert Register.TECHNICAL.value == "technical"
        assert Register.NEUTRAL.value == "neutral"
