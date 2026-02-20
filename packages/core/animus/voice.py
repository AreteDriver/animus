"""
Animus Voice Interface

Speech-to-text (Whisper) and text-to-speech capabilities.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from animus.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("voice")


class VoiceInput:
    """
    Speech-to-text using OpenAI Whisper.

    Supports file transcription and microphone input.
    """

    def __init__(self, model: str = "base"):
        """
        Initialize Whisper model.

        Args:
            model: Whisper model size - tiny, base, small, medium, large
                   Larger models are more accurate but slower.
        """
        self.model_name = model
        self._model = None
        self._listening = False
        self._listen_thread: threading.Thread | None = None
        logger.debug(f"VoiceInput initialized with model: {model}")

    def _load_model(self) -> None:
        """Lazy-load Whisper model."""
        if self._model is None:
            try:
                import whisper

                logger.info(f"Loading Whisper model: {self.model_name}")
                self._model = whisper.load_model(self.model_name)
                logger.info("Whisper model loaded")
            except ImportError:
                raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
        return self._model

    def transcribe_file(self, audio_path: Path | str) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (mp3, wav, etc.)

        Returns:
            Transcribed text
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()
        logger.debug(f"Transcribing file: {audio_path}")

        result = model.transcribe(str(audio_path))
        text = result["text"].strip()

        logger.debug(f"Transcription: {text[:50]}...")
        return text

    def transcribe_microphone(self, duration: int = 5, sample_rate: int = 16000) -> str:
        """
        Record from microphone and transcribe.

        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate (16000 recommended for Whisper)

        Returns:
            Transcribed text
        """
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice/numpy not installed. Install with: pip install sounddevice numpy"
            )

        logger.info(f"Recording for {duration} seconds...")

        # Record audio
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()

        logger.debug("Recording complete, transcribing...")

        # Transcribe directly from numpy array
        model = self._load_model()
        audio_flat = audio.flatten()
        result = model.transcribe(audio_flat)

        text = result["text"].strip()
        logger.debug(f"Transcription: {text[:50]}...")
        return text

    def listen_continuous(
        self,
        callback: Callable[[str], None],
        chunk_duration: int = 3,
        silence_threshold: float = 0.01,
        sample_rate: int = 16000,
    ) -> None:
        """
        Continuous listening with callback on detected speech.

        This runs in a background thread. Call stop_listening() to stop.

        Args:
            callback: Function to call with transcribed text
            chunk_duration: Duration of each audio chunk in seconds
            silence_threshold: RMS threshold below which is considered silence
            sample_rate: Audio sample rate
        """
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            raise ImportError(
                "sounddevice/numpy not installed. Install with: pip install sounddevice numpy"
            )

        def listen_loop():
            model = self._load_model()
            logger.info("Continuous listening started")

            while self._listening:
                # Record chunk
                audio = sd.rec(
                    int(chunk_duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype=np.float32,
                )
                sd.wait()

                if not self._listening:
                    break

                # Check if chunk contains speech (not just silence)
                audio_flat = audio.flatten()
                rms = np.sqrt(np.mean(audio_flat**2))

                if rms > silence_threshold:
                    logger.debug(f"Speech detected (RMS: {rms:.4f})")
                    result = model.transcribe(audio_flat)
                    text = result["text"].strip()

                    if text:
                        callback(text)
                else:
                    logger.debug(f"Silence (RMS: {rms:.4f})")

            logger.info("Continuous listening stopped")

        self._listening = True
        self._listen_thread = threading.Thread(target=listen_loop, daemon=True)
        self._listen_thread.start()

    def stop_listening(self) -> None:
        """Stop continuous listening."""
        self._listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=5)
            self._listen_thread = None

    @property
    def is_listening(self) -> bool:
        """Check if continuous listening is active."""
        return self._listening


class VoiceOutput:
    """
    Text-to-speech output.

    Supports pyttsx3 (offline) and edge-tts (higher quality, requires internet).
    """

    def __init__(self, engine: str = "pyttsx3", rate: int = 150):
        """
        Initialize TTS engine.

        Args:
            engine: TTS engine to use - "pyttsx3" (offline) or "edge-tts" (online)
            rate: Speech rate in words per minute
        """
        self.engine_name = engine
        self.rate = rate
        self._engine = None
        self._voice_id: str | None = None
        self._speaking = False
        logger.debug(f"VoiceOutput initialized with engine: {engine}")

    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        if self._engine is None:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                logger.debug("pyttsx3 engine initialized")
            except ImportError:
                raise ImportError("pyttsx3 not installed. Install with: pip install pyttsx3")
        return self._engine

    def speak(self, text: str) -> None:
        """
        Speak text aloud (blocking).

        Args:
            text: Text to speak
        """
        if self.engine_name == "pyttsx3":
            self._speak_pyttsx3(text)
        elif self.engine_name == "edge-tts":
            self._speak_edge_tts(text)
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_name}")

    def _speak_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3."""
        engine = self._init_pyttsx3()
        self._speaking = True
        logger.debug(f"Speaking: {text[:50]}...")

        engine.say(text)
        engine.runAndWait()

        self._speaking = False
        logger.debug("Speaking complete")

    def _speak_edge_tts(self, text: str) -> None:
        """Speak using edge-tts."""
        try:
            import asyncio
            import tempfile

            import edge_tts
        except ImportError:
            raise ImportError("edge-tts not installed. Install with: pip install edge-tts")

        async def _speak():
            voice = self._voice_id or "en-US-AriaNeural"
            communicate = edge_tts.Communicate(text, voice)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name

            await communicate.save(temp_path)

            # Play the audio file
            try:
                import sounddevice as sd
                import soundfile as sf

                data, samplerate = sf.read(temp_path)
                sd.play(data, samplerate)
                sd.wait()
            except ImportError:
                # Fallback to system command
                import subprocess

                subprocess.run(["mpv", "--no-video", temp_path], capture_output=True)
            finally:
                Path(temp_path).unlink(missing_ok=True)

        self._speaking = True
        logger.debug(f"Speaking (edge-tts): {text[:50]}...")

        asyncio.run(_speak())

        self._speaking = False
        logger.debug("Speaking complete")

    def speak_async(self, text: str) -> threading.Thread:
        """
        Speak text in background thread.

        Args:
            text: Text to speak

        Returns:
            Thread handle
        """
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """Stop current speech (pyttsx3 only)."""
        if self.engine_name == "pyttsx3" and self._engine:
            self._engine.stop()
            self._speaking = False

    def set_voice(self, voice_id: str) -> None:
        """
        Set voice by ID.

        For pyttsx3: Get available voices with get_voices()
        For edge-tts: Use voice names like "en-US-AriaNeural", "en-US-GuyNeural"

        Args:
            voice_id: Voice identifier
        """
        self._voice_id = voice_id
        if self.engine_name == "pyttsx3" and self._engine:
            self._engine.setProperty("voice", voice_id)
        logger.debug(f"Voice set to: {voice_id}")

    def set_rate(self, rate: int) -> None:
        """
        Set speech rate.

        Args:
            rate: Words per minute (default: 150)
        """
        self.rate = rate
        if self.engine_name == "pyttsx3" and self._engine:
            self._engine.setProperty("rate", rate)
        logger.debug(f"Rate set to: {rate}")

    def get_voices(self) -> list[dict]:
        """
        Get available voices (pyttsx3 only).

        Returns:
            List of voice info dicts with id, name, languages
        """
        if self.engine_name != "pyttsx3":
            return []

        engine = self._init_pyttsx3()
        voices = engine.getProperty("voices")

        return [
            {
                "id": v.id,
                "name": v.name,
                "languages": v.languages,
            }
            for v in voices
        ]

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._speaking


class VoiceInterface:
    """
    Combined voice input/output interface.

    Convenience class that manages both VoiceInput and VoiceOutput.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        tts_engine: str = "pyttsx3",
        tts_rate: int = 150,
    ):
        """
        Initialize voice interface.

        Args:
            whisper_model: Whisper model size
            tts_engine: TTS engine name
            tts_rate: TTS speech rate
        """
        self.input = VoiceInput(model=whisper_model)
        self.output = VoiceOutput(engine=tts_engine, rate=tts_rate)
        self._response_tts_enabled = False
        logger.info("VoiceInterface initialized")

    def listen(self, duration: int = 5) -> str:
        """
        Listen for speech and return transcription.

        Args:
            duration: Recording duration in seconds

        Returns:
            Transcribed text
        """
        return self.input.transcribe_microphone(duration=duration)

    def speak(self, text: str, async_: bool = False) -> threading.Thread | None:
        """
        Speak text aloud.

        Args:
            text: Text to speak
            async_: If True, speak in background thread

        Returns:
            Thread handle if async_, else None
        """
        if async_:
            return self.output.speak_async(text)
        else:
            self.output.speak(text)
            return None

    def start_listening(self, callback: Callable[[str], None]) -> None:
        """
        Start continuous listening.

        Args:
            callback: Function to call with transcribed text
        """
        self.input.listen_continuous(callback)

    def stop_listening(self) -> None:
        """Stop continuous listening."""
        self.input.stop_listening()

    @property
    def response_tts_enabled(self) -> bool:
        """Whether TTS is enabled for responses."""
        return self._response_tts_enabled

    @response_tts_enabled.setter
    def response_tts_enabled(self, value: bool) -> None:
        """Enable/disable TTS for responses."""
        self._response_tts_enabled = value
        logger.debug(f"Response TTS {'enabled' if value else 'disabled'}")
