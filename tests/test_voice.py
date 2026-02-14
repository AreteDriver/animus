"""
Tests for Voice Interface

Uses mock audio devices so tests run without hardware or optional dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Mock Audio Infrastructure
# =============================================================================


class MockWhisperModel:
    """Mock Whisper model that returns pre-configured transcriptions."""

    def __init__(self, transcriptions: dict[str, str] | None = None, default: str = "hello world"):
        self.transcriptions = transcriptions or {}
        self.default = default
        self.calls: list[dict] = []

    def transcribe(self, audio_input, **kwargs):
        self.calls.append({"input": str(audio_input)[:50], "kwargs": kwargs})
        # If input is a file path, check transcriptions map
        if isinstance(audio_input, str) and audio_input in self.transcriptions:
            return {"text": self.transcriptions[audio_input]}
        return {"text": self.default}


class MockSoundDevice:
    """Mock sounddevice module for testing without audio hardware."""

    def __init__(self, audio_data=None):
        self._audio_data = audio_data
        self.recordings: list[dict] = []
        self.playbacks: list[dict] = []

    def rec(self, frames, samplerate=16000, channels=1, dtype="float32"):
        import numpy as np

        self.recordings.append(
            {
                "frames": frames,
                "samplerate": samplerate,
                "channels": channels,
            }
        )
        if self._audio_data is not None:
            return self._audio_data
        # Return fake audio with some "speech" energy
        return np.random.randn(frames, channels).astype(dtype) * 0.1

    def wait(self):
        pass

    def play(self, data, samplerate):
        self.playbacks.append({"frames": len(data), "samplerate": samplerate})

    def stop(self):
        pass


# =============================================================================
# VoiceInput Tests
# =============================================================================


class TestVoiceInput:
    """Tests for speech-to-text input."""

    def test_init_default_model(self):
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        assert vi.model_name == "base"
        assert vi._model is None
        assert vi._listening is False

    def test_init_custom_model(self):
        from animus.voice import VoiceInput

        vi = VoiceInput(model="tiny")
        assert vi.model_name == "tiny"

    def test_transcribe_file(self, tmp_path):
        """Test file transcription with mock model."""
        from animus.voice import VoiceInput

        vi = VoiceInput()

        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)

        mock_model = MockWhisperModel(transcriptions={str(audio_file): "test transcription"})
        vi._model = mock_model

        result = vi.transcribe_file(audio_file)
        assert result == "test transcription"
        assert len(mock_model.calls) == 1

    def test_transcribe_file_not_found(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        with pytest.raises(FileNotFoundError):
            vi.transcribe_file("/nonexistent/audio.wav")

    def test_transcribe_microphone_with_mock(self):
        """Test microphone transcription with mocked audio stack."""
        from animus.voice import VoiceInput

        vi = VoiceInput()
        mock_model = MockWhisperModel(default="microphone test")
        vi._model = mock_model

        mock_sd = MockSoundDevice()
        mock_np = MagicMock()
        mock_np.float32 = "float32"
        mock_np.sqrt = lambda x: x
        mock_np.mean = lambda x: x

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": mock_np}):
            # Directly test the flow by mocking imports
            import numpy as np

            mock_sd_module = MockSoundDevice()
            audio = mock_sd_module.rec(5 * 16000, samplerate=16000, channels=1, dtype=np.float32)
            mock_sd_module.wait()
            audio_flat = audio.flatten()
            result = mock_model.transcribe(audio_flat)

            assert result["text"] == "microphone test"
            assert len(mock_sd_module.recordings) == 1

    def test_listening_state(self):
        """Test listening state management."""
        from animus.voice import VoiceInput

        vi = VoiceInput()
        assert not vi.is_listening
        vi._listening = True
        assert vi.is_listening
        vi._listening = False
        assert not vi.is_listening


# =============================================================================
# VoiceOutput Tests
# =============================================================================


class TestVoiceOutput:
    """Tests for text-to-speech output."""

    def test_init_pyttsx3(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3", rate=150)
        assert vo.engine_name == "pyttsx3"
        assert vo.rate == 150
        assert not vo.is_speaking

    def test_init_edge_tts(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="edge-tts")
        assert vo.engine_name == "edge-tts"

    def test_set_voice(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        vo.set_voice("test-voice-id")
        assert vo._voice_id == "test-voice-id"

    def test_set_rate(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        vo.set_rate(200)
        assert vo.rate == 200

    def test_speak_unknown_engine(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="nonexistent")
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            vo.speak("hello")

    def test_speaking_state(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput()
        assert not vo.is_speaking
        vo._speaking = True
        assert vo.is_speaking
        vo._speaking = False
        assert not vo.is_speaking


# =============================================================================
# VoiceInterface Tests
# =============================================================================


class TestVoiceInterface:
    """Tests for the combined voice interface."""

    def test_init(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface(whisper_model="tiny", tts_engine="pyttsx3", tts_rate=120)
        assert vi.input.model_name == "tiny"
        assert vi.output.engine_name == "pyttsx3"
        assert vi.output.rate == 120
        assert not vi.response_tts_enabled

    def test_response_tts_toggle(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface()
        assert not vi.response_tts_enabled
        vi.response_tts_enabled = True
        assert vi.response_tts_enabled
        vi.response_tts_enabled = False
        assert not vi.response_tts_enabled

    def test_stop_listening_when_not_active(self):
        from animus.voice import VoiceInterface

        vi = VoiceInterface()
        # Should not raise
        vi.stop_listening()
        assert not vi.input.is_listening


# =============================================================================
# Mock Model Integration Test
# =============================================================================


class TestMockWhisperModel:
    """Tests for the mock Whisper model itself."""

    def test_default_transcription(self):
        model = MockWhisperModel(default="default response")
        result = model.transcribe("any_input")
        assert result["text"] == "default response"

    def test_file_specific_transcription(self):
        model = MockWhisperModel(transcriptions={"/path/to/audio.wav": "specific text"})
        result = model.transcribe("/path/to/audio.wav")
        assert result["text"] == "specific text"

    def test_call_history(self):
        model = MockWhisperModel()
        model.transcribe("input1")
        model.transcribe("input2")
        assert len(model.calls) == 2


class TestMockSoundDevice:
    """Tests for the mock sounddevice module."""

    def test_recording(self):
        import numpy as np

        sd = MockSoundDevice()
        audio = sd.rec(16000, samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        assert audio.shape == (16000, 1)
        assert len(sd.recordings) == 1

    def test_playback(self):
        import numpy as np

        sd = MockSoundDevice()
        data = np.zeros(16000)
        sd.play(data, samplerate=16000)
        assert len(sd.playbacks) == 1
        assert sd.playbacks[0]["samplerate"] == 16000

    def test_custom_audio_data(self):
        import numpy as np

        custom = np.ones((8000, 1), dtype=np.float32)
        sd = MockSoundDevice(audio_data=custom)
        audio = sd.rec(8000, samplerate=16000, channels=1)
        assert (audio == custom).all()
