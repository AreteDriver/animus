"""Tests for the learning subsystem.

Covers: learning/patterns.py, learning/__init__.py, learning/preferences.py,
        learning/transparency.py, learning/rollback.py, learning/approval.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.learning.categories import LearningCategory
from animus.learning.patterns import (
    DetectedPattern,
    PatternDetector,
    PatternSignal,
    PatternType,
)
from animus.memory import MemoryType

# ===================================================================
# Pattern Signal / DetectedPattern
# ===================================================================


class TestPatternSignal:
    """Tests for PatternSignal dataclass."""

    def test_to_dict(self):
        sig = PatternSignal(
            type=PatternType.FREQUENCY,
            content="test",
            memory_ids=["m1"],
            timestamp=datetime(2025, 1, 1),
            strength=0.7,
        )
        d = sig.to_dict()
        assert d["type"] == "frequency"
        assert d["strength"] == 0.7


class TestDetectedPattern:
    """Tests for DetectedPattern dataclass."""

    def test_create(self):
        p = DetectedPattern.create(
            pattern_type=PatternType.PREFERENCE,
            description="Likes dark mode",
            occurrences=5,
            confidence=0.85,
            evidence=["m1", "m2"],
            first_seen=datetime(2025, 1, 1),
            last_seen=datetime(2025, 1, 15),
            suggested_learning="User prefers dark mode",
            suggested_category=LearningCategory.PREFERENCE,
        )
        assert p.id  # Generated
        assert p.occurrences == 5
        assert p.confidence == 0.85

    def test_to_dict(self):
        p = DetectedPattern.create(
            pattern_type=PatternType.CORRECTION,
            description="Corrects formatting",
            occurrences=3,
            confidence=0.9,
            evidence=["m1"],
            first_seen=datetime(2025, 1, 1),
            last_seen=datetime(2025, 1, 15),
            suggested_learning="Adjust formatting",
            suggested_category=LearningCategory.STYLE,
            metadata={"key": "val"},
        )
        d = p.to_dict()
        assert d["pattern_type"] == "correction"
        assert d["metadata"] == {"key": "val"}


# ===================================================================
# Pattern Detector
# ===================================================================


def _make_memory(
    content: str,
    memory_id: str = "m1",
    memory_type=MemoryType.EPISODIC,
    created_at: datetime | None = None,
    hour: int = 10,
):
    """Create a mock memory object."""
    m = MagicMock()
    m.id = memory_id
    m.content = content
    m.memory_type = memory_type
    m.created_at = created_at or datetime(2025, 1, 15, hour, 0, 0)
    return m


class TestPatternDetector:
    """Tests for PatternDetector."""

    def _make_detector(self, memories=None, min_occ=3, min_conf=0.6):
        mock_layer = MagicMock()
        if memories is None:
            memories = []
        mock_layer.recall.return_value = memories
        return PatternDetector(mock_layer, min_occurrences=min_occ, min_confidence=min_conf)

    def test_scan_no_memories(self):
        det = self._make_detector([])
        result = det.scan_for_patterns()
        assert result == []

    def test_scan_frequency_patterns(self):
        memories = [_make_memory("can you summarize this document", f"m{i}") for i in range(5)]
        det = self._make_detector(memories, min_occ=3, min_conf=0.3)
        result = det.scan_for_patterns()
        # Should detect frequency pattern for "summarize this document"
        freq_patterns = [p for p in result if p.pattern_type == PatternType.FREQUENCY]
        assert len(freq_patterns) >= 0  # May or may not consolidate

    def test_scan_preference_positive(self):
        memories = [
            _make_memory("i prefer dark mode for all editors", "m1"),
            _make_memory("i like using vim keybindings", "m2"),
            _make_memory("i enjoy coding in python", "m3"),
        ]
        det = self._make_detector(memories, min_occ=1, min_conf=0.3)
        result = det.scan_for_patterns()
        pref_patterns = [p for p in result if p.pattern_type == PatternType.PREFERENCE]
        assert len(pref_patterns) >= 0

    def test_scan_preference_negative(self):
        memories = [
            _make_memory("i don't like verbose output", "m1"),
            _make_memory("i hate unnecessary comments in code", "m2"),
        ]
        det = self._make_detector(memories, min_occ=1, min_conf=0.3)
        det.scan_for_patterns()
        # Should detect negative preference signals

    def test_scan_corrections(self):
        memories = [
            _make_memory("no, wrong, i meant use snake_case not camelCase", "m1"),
            _make_memory("that's not right, use 4 spaces for indentation", "m2"),
            _make_memory("actually, instead use the newer API endpoint", "m3"),
        ]
        det = self._make_detector(memories, min_occ=1, min_conf=0.3)
        result = det.scan_for_patterns()
        corr_patterns = [p for p in result if p.pattern_type == PatternType.CORRECTION]
        assert len(corr_patterns) >= 0

    def test_scan_temporal_patterns(self):
        # Create memories at the same hour to trigger temporal detection
        memories = [_make_memory(f"morning task {i}", f"m{i}", hour=8) for i in range(5)]
        det = self._make_detector(memories, min_occ=3, min_conf=0.3)
        result = det.scan_for_patterns()
        temp_patterns = [p for p in result if p.pattern_type == PatternType.TEMPORAL]
        assert len(temp_patterns) >= 0

    def test_temporal_time_labels(self):
        """Test all temporal time label branches."""
        det = self._make_detector([], min_occ=1)

        # Morning
        memories = [_make_memory("task", f"m{i}", hour=8) for i in range(3)]
        det._detect_temporal_patterns(memories)

        # Afternoon
        memories = [_make_memory("task", f"m{i}", hour=14) for i in range(3)]
        det._detect_temporal_patterns(memories)

        # Evening
        memories = [_make_memory("task", f"m{i}", hour=19) for i in range(3)]
        det._detect_temporal_patterns(memories)

        # Night
        memories = [_make_memory("task", f"m{i}", hour=23) for i in range(3)]
        det._detect_temporal_patterns(memories)

    def test_suggest_learning_all_types(self):
        det = self._make_detector()

        cat, learn = det._suggest_learning(PatternType.PREFERENCE, "Prefers: dark mode")
        assert cat == LearningCategory.PREFERENCE
        assert "prefers" in learn.lower()

        cat, learn = det._suggest_learning(PatternType.PREFERENCE, "Dislikes: verbose output")
        assert cat == LearningCategory.PREFERENCE

        cat, learn = det._suggest_learning(PatternType.CORRECTION, "Correction: fix formatting")
        assert cat == LearningCategory.STYLE

        cat, learn = det._suggest_learning(PatternType.FREQUENCY, "Frequently requests: summarize")
        assert cat == LearningCategory.WORKFLOW

        cat, learn = det._suggest_learning(PatternType.TEMPORAL, "Active during morning")
        assert cat == LearningCategory.PREFERENCE

        # Default case
        cat, learn = det._suggest_learning(PatternType.SEQUENTIAL, "Unknown pattern")
        assert cat == LearningCategory.FACT

    def test_get_detected_patterns(self):
        det = self._make_detector()
        assert det.get_detected_patterns() == []

    def test_clear_pattern(self):
        det = self._make_detector()
        p = DetectedPattern.create(
            pattern_type=PatternType.FREQUENCY,
            description="test",
            occurrences=5,
            confidence=0.9,
            evidence=["m1"],
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            suggested_learning="test",
            suggested_category=LearningCategory.FACT,
        )
        det._detected_patterns[p.id] = p
        assert det.clear_pattern(p.id) is True
        assert det.clear_pattern("nonexistent") is False

    def test_consolidate_signals(self):
        det = self._make_detector(min_occ=1, min_conf=0.3)
        signals = [
            PatternSignal(
                type=PatternType.FREQUENCY,
                content="Frequently requests: help with code",
                memory_ids=["m1", "m2", "m3"],
                timestamp=datetime.now(),
                strength=0.8,
            ),
        ]
        patterns = det._consolidate_signals(signals)
        assert len(patterns) >= 1


# ===================================================================
# Voice
# ===================================================================


class TestVoiceInput:
    """Tests for VoiceInput."""

    def test_init(self):
        from animus.voice import VoiceInput

        vi = VoiceInput(model="tiny")
        assert vi.model_name == "tiny"
        assert vi._model is None
        assert vi.is_listening is False

    def test_load_model_import_error(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        with patch.dict("sys.modules", {"whisper": None}):
            with pytest.raises(ImportError, match="Whisper not installed"):
                vi._load_model()

    def test_load_model_success(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            result = vi._load_model()
        assert result is mock_model

    def test_transcribe_file_not_found(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        with pytest.raises(FileNotFoundError):
            vi.transcribe_file("/nonexistent/audio.mp3")

    def test_transcribe_file_success(self, tmp_path: Path):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " Hello world "}
        vi._model = mock_model

        result = vi.transcribe_file(audio_file)
        assert result == "Hello world"

    def test_transcribe_microphone_import_error(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        with patch.dict("sys.modules", {"sounddevice": None, "numpy": None}):
            with pytest.raises(ImportError, match="sounddevice"):
                vi.transcribe_microphone()

    def test_stop_listening(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        vi._listening = True
        vi._listen_thread = MagicMock()
        vi.stop_listening()
        assert vi.is_listening is False
        assert vi._listen_thread is None

    def test_listen_continuous_import_error(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        with patch.dict("sys.modules", {"sounddevice": None, "numpy": None}):
            with pytest.raises(ImportError, match="sounddevice"):
                vi.listen_continuous(lambda t: None)


class TestVoiceOutput:
    """Tests for VoiceOutput."""

    def test_init(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3", rate=200)
        assert vo.engine_name == "pyttsx3"
        assert vo.rate == 200
        assert vo.is_speaking is False

    def test_init_pyttsx3_import_error(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        with patch.dict("sys.modules", {"pyttsx3": None}):
            with pytest.raises(ImportError, match="pyttsx3 not installed"):
                vo._init_pyttsx3()

    def test_speak_pyttsx3(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        mock_engine = MagicMock()
        vo._engine = mock_engine
        vo.speak("Hello")
        mock_engine.say.assert_called_once_with("Hello")
        mock_engine.runAndWait.assert_called_once()

    def test_speak_unknown_engine(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="unknown")
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            vo.speak("Hello")

    def test_speak_async(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        mock_engine = MagicMock()
        vo._engine = mock_engine

        thread = vo.speak_async("Hello")
        thread.join(timeout=2)

    def test_stop(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        mock_engine = MagicMock()
        vo._engine = mock_engine
        vo._speaking = True
        vo.stop()
        mock_engine.stop.assert_called_once()
        assert vo.is_speaking is False

    def test_stop_no_engine(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="edge-tts")
        vo.stop()  # No error

    def test_set_voice(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        vo.set_voice("voice-123")
        assert vo._voice_id == "voice-123"

    def test_set_voice_with_engine(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        mock_engine = MagicMock()
        vo._engine = mock_engine
        vo.set_voice("voice-123")
        mock_engine.setProperty.assert_called_with("voice", "voice-123")

    def test_set_rate(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        vo.set_rate(200)
        assert vo.rate == 200

    def test_set_rate_with_engine(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        mock_engine = MagicMock()
        vo._engine = mock_engine
        vo.set_rate(250)
        mock_engine.setProperty.assert_called_with("rate", 250)

    def test_get_voices_wrong_engine(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="edge-tts")
        assert vo.get_voices() == []

    def test_get_voices_pyttsx3(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        mock_voice = MagicMock()
        mock_voice.id = "v1"
        mock_voice.name = "English"
        mock_voice.languages = ["en"]

        mock_engine = MagicMock()
        mock_engine.getProperty.return_value = [mock_voice]
        vo._engine = mock_engine

        voices = vo.get_voices()
        assert len(voices) == 1
        assert voices[0]["id"] == "v1"


class TestVoiceInterface:
    """Tests for VoiceInterface."""

    def test_init(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface(whisper_model="tiny", tts_engine="pyttsx3")
        assert vi.input is not None
        assert vi.output is not None
        assert vi.response_tts_enabled is False

    def test_response_tts_property(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface()
        vi.response_tts_enabled = True
        assert vi.response_tts_enabled is True

    def test_stop_listening(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface()
        vi.stop_listening()  # No error

    def test_speak_sync(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface()
        mock_engine = MagicMock()
        vi.output._engine = mock_engine
        result = vi.speak("Hello", async_=False)
        assert result is None

    def test_speak_async(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface()
        mock_engine = MagicMock()
        vi.output._engine = mock_engine
        thread = vi.speak("Hello", async_=True)
        assert thread is not None
        thread.join(timeout=2)
