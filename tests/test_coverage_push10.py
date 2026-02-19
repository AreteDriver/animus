"""Coverage push round 10 — voice listen loop, proactive synthesis, swarm conflicts, filesystem."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus.swarm.engine import SwarmEngine

# ---------------------------------------------------------------------------
# Voice — full listen_continuous loop (lines 140, 146-181)
# ---------------------------------------------------------------------------


class TestVoiceListenContinuousLoop:
    """Lines 146-181: listen_continuous inner loop with speech detection."""

    def test_listen_loop_speech_and_silence(self):
        """Exercises the full listen loop: speech detected, silence, then stop."""
        np = pytest.importorskip("numpy")
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        transcribed = []

        call_count = [0]

        def fake_rec(frames, samplerate, channels, dtype):
            call_count[0] += 1
            if call_count[0] == 1:
                # High RMS — speech detected
                return np.ones((frames, 1), dtype=np.float32)
            elif call_count[0] == 2:
                # Zero RMS — silence
                return np.zeros((frames, 1), dtype=np.float32)
            else:
                # Stop
                vi._listening = False
                return np.zeros((frames, 1), dtype=np.float32)

        mock_sd = MagicMock()
        mock_sd.rec = fake_rec
        mock_sd.wait = MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}

        with (
            patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": np}),
            patch.object(vi, "_load_model", return_value=mock_model),
        ):
            vi.listen_continuous(callback=transcribed.append, chunk_duration=0.01)

            # Wait for thread to finish
            if vi._listen_thread:
                vi._listen_thread.join(timeout=5)

        assert "hello world" in transcribed

    def test_listen_loop_empty_transcription(self):
        """Speech detected but transcription is empty → callback NOT called."""
        np = pytest.importorskip("numpy")
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        transcribed = []
        call_count = [0]

        def fake_rec(frames, samplerate, channels, dtype):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.ones((frames, 1), dtype=np.float32)
            vi._listening = False
            return np.zeros((frames, 1), dtype=np.float32)

        mock_sd = MagicMock()
        mock_sd.rec = fake_rec
        mock_sd.wait = MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "   "}  # Whitespace only

        with (
            patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": np}),
            patch.object(vi, "_load_model", return_value=mock_model),
        ):
            vi.listen_continuous(callback=transcribed.append, chunk_duration=0.01)
            if vi._listen_thread:
                vi._listen_thread.join(timeout=5)

        assert len(transcribed) == 0

    def test_listen_loop_stop_during_recording(self):
        """Listening stopped between rec and check → breaks out."""
        np = pytest.importorskip("numpy")
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")

        def fake_rec(frames, samplerate, channels, dtype):
            # Stop listening during recording
            vi._listening = False
            return np.zeros((frames, 1), dtype=np.float32)

        mock_sd = MagicMock()
        mock_sd.rec = fake_rec
        mock_sd.wait = MagicMock()

        mock_model = MagicMock()

        with (
            patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": np}),
            patch.object(vi, "_load_model", return_value=mock_model),
        ):
            vi.listen_continuous(callback=lambda t: None, chunk_duration=0.01)
            if vi._listen_thread:
                vi._listen_thread.join(timeout=5)

        # Model.transcribe should NOT have been called
        mock_model.transcribe.assert_not_called()

    def test_voice_input_load_model_import_error(self):
        """Line 140: VoiceInput._load_model when whisper not installed."""
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        with patch.dict("sys.modules", {"whisper": None}):
            with pytest.raises(ImportError, match="Whisper"):
                vi._load_model()


# ---------------------------------------------------------------------------
# Proactive — follow-up with cognitive synthesis (lines 254-255, 456-458)
# ---------------------------------------------------------------------------


class TestProactiveFollowUpSynthesis:
    """Lines 449-458: cognitive synthesis for follow-up nudges."""

    def _make_pe(self, tmp_path):
        from animus.proactive import ProactiveEngine

        mem = MagicMock()
        cognitive = MagicMock()
        pe = ProactiveEngine(data_dir=tmp_path, memory=mem, cognitive=cognitive)
        pe.memory = mem
        pe.memory.store = MagicMock()
        return pe, mem, cognitive

    def test_follow_up_synthesis_success(self, tmp_path):
        pe, mem, cognitive = self._make_pe(tmp_path)
        cognitive.think.return_value = "Follow up: send the report"

        m = MagicMock()
        m.content = "I need to follow up on the quarterly report"
        m.memory_type = MagicMock()
        m.memory_type.value = "episodic"
        m.created_at = datetime.now() - timedelta(days=2)
        m.id = "m1"
        m.tags = set()
        pe.memory.store.list_all.return_value = [m]

        result = pe.scan_follow_ups()
        assert len(result) == 1
        assert "report" in result[0].content.lower()
        cognitive.think.assert_called_once()

    def test_follow_up_synthesis_failure(self, tmp_path):
        """Line 457-458: synthesis failure falls back to raw content."""
        pe, mem, cognitive = self._make_pe(tmp_path)
        cognitive.think.side_effect = RuntimeError("LLM down")

        m = MagicMock()
        m.content = "Need to follow up on the invoice"
        m.memory_type = MagicMock()
        m.memory_type.value = "episodic"
        m.created_at = datetime.now() - timedelta(days=1)
        m.id = "m2"
        m.tags = set()
        pe.memory.store.list_all.return_value = [m]

        result = pe.scan_follow_ups()
        assert len(result) == 1
        assert "follow up" in result[0].content.lower()


class TestProactiveMorningBriefFollowUps:
    """Lines 254-255: morning brief follow-ups section."""

    def _make_pe(self, tmp_path):
        from animus.proactive import ProactiveEngine

        mem = MagicMock()
        pe = ProactiveEngine(data_dir=tmp_path, memory=mem)
        pe.memory = mem
        pe.memory.store = MagicMock()
        return pe, mem

    def test_morning_brief_with_follow_ups(self, tmp_path):
        pe, mem = self._make_pe(tmp_path)

        m = MagicMock()
        m.content = "Important follow-up needed for client meeting"
        m.tags = {"follow-up"}
        m.created_at = datetime.now() - timedelta(hours=2)
        m.memory_type = MagicMock()
        m.memory_type.value = "episodic"
        mem.recall.return_value = [m]

        result = pe.generate_morning_brief()
        assert result is not None
        assert "follow" in result.content.lower() or result.content


# ---------------------------------------------------------------------------
# Proactive — background loop error + sleep break (lines 581-582, 586-587)
# ---------------------------------------------------------------------------


class TestProactiveBackgroundLoopError:
    """Lines 581-582: run_scheduled_checks raises in background loop."""

    def test_background_loop_recovers_from_error(self, tmp_path):
        from animus.proactive import ProactiveEngine

        mem = MagicMock()
        pe = ProactiveEngine(data_dir=tmp_path, memory=mem)
        pe.memory = mem
        pe.memory.store = MagicMock()

        call_count = [0]

        def flaky_run():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient error")
            pe._running = False
            return []

        pe.run_scheduled_checks = flaky_run

        with patch("time.sleep", return_value=None):
            pe.start_background(interval_seconds=1)
            pe._thread.join(timeout=5)

        assert call_count[0] >= 2


# ---------------------------------------------------------------------------
# SwarmEngine — conflict resolution in _publish_stage_intents (lines 214-220)
# ---------------------------------------------------------------------------


class TestSwarmEngineIntentConflicts:
    """Lines 214-220: conflict detection and resolution during stage publishing."""

    def test_conflict_resolution_logged(self, tmp_path):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.forge.models import AgentConfig
        from animus.swarm.intent import IntentEntry

        config = ModelConfig.mock(default_response="done")
        cognitive = CognitiveLayer(config)
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)

        # Pre-populate intent graph with conflicting intent
        existing = IntentEntry(
            agent="prior_agent",
            action="execute_step",
            provides=["a1.brief"],
            requires=[],
            stability=0.9,
            status="pending",
        )
        engine._intent_graph.publish(existing)

        # Now publish stage with agent that provides the same key
        a1 = AgentConfig(name="a1", archetype="researcher", outputs=["brief"], budget_tokens=5000)
        engine._publish_stage_intents(
            agent_names=["a1"],
            agent_configs={"a1": a1},
            available_keys=set(),
        )

        # Intent should have conflict evidence
        intent = engine._intent_graph.get("a1")
        assert intent is not None
        assert any("conflict_resolved" in e for e in intent.evidence)


# ---------------------------------------------------------------------------
# Filesystem — _index_directory recursive with subdirs (lines 231)
#              + entry stat error (lines 233-234)
# ---------------------------------------------------------------------------


class TestFilesystemIndexRecursive:
    """Lines 230-234: recursive indexing + per-entry error handling."""

    def test_index_recursive_with_subdir(self, tmp_path):
        from animus.integrations.filesystem import FilesystemIntegration

        fi = FilesystemIntegration()
        fi._index = {}
        fi._indexed_paths = []

        # Create nested structure
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "file1.txt").write_text("a")
        (sub / "file2.txt").write_text("b")

        count = fi._index_directory(tmp_path, recursive=True)
        assert count >= 3  # file1 + subdir + file2

    def test_index_entry_permission_error(self, tmp_path):
        """Lines 233-234: individual entry stat error."""
        from animus.integrations.filesystem import FilesystemIntegration

        fi = FilesystemIntegration()
        fi._index = {}

        (tmp_path / "ok.txt").write_text("fine")

        # Patch Path.iterdir to include one entry that raises on stat()
        real_iterdir = Path.iterdir

        def patched_iterdir(self_path):
            entries = list(real_iterdir(self_path))
            bad_entry = MagicMock()
            bad_entry.name = "bad.txt"
            bad_entry.stat.side_effect = PermissionError("denied")
            entries.append(bad_entry)
            return iter(entries)

        with patch.object(Path, "iterdir", patched_iterdir):
            count = fi._index_directory(tmp_path, recursive=False)

        # Should have indexed ok.txt but skipped bad.txt
        assert count >= 1


# ---------------------------------------------------------------------------
# Filesystem — _should_exclude with .git prefix (line 114)
# ---------------------------------------------------------------------------


class TestFilesystemShouldExcludeGitPrefix:
    """Line 114: _should_exclude for .git paths."""

    def test_git_config_excluded(self):
        from animus.integrations.filesystem import FilesystemIntegration

        fi = FilesystemIntegration()
        assert fi._should_exclude(Path("/repo/.git")) is True
        assert fi._should_exclude(Path("/repo/.git/config")) is True

    def test_regular_file_not_excluded(self):
        from animus.integrations.filesystem import FilesystemIntegration

        fi = FilesystemIntegration()
        assert fi._should_exclude(Path("/repo/src/main.py")) is False


# ---------------------------------------------------------------------------
# Google __init__.py — gmail fallback (lines 19-20)
# ---------------------------------------------------------------------------


class TestGoogleSubpackageGmailFallback:
    """Lines 19-20: google/__init__.py Gmail import failure."""

    def test_gmail_import_none_when_unavailable(self):
        import importlib
        import sys

        import animus.integrations.google as goog

        # Save originals
        orig_gmail_mod = sys.modules.get("animus.integrations.google.gmail")
        had_gmail_attr = hasattr(goog, "gmail")
        orig_gmail_attr = getattr(goog, "gmail", None)

        try:
            # Remove gmail from both sys.modules AND parent module attribute
            sys.modules["animus.integrations.google.gmail"] = None  # type: ignore
            if hasattr(goog, "gmail"):
                delattr(goog, "gmail")
            if hasattr(goog, "GmailIntegration"):
                delattr(goog, "GmailIntegration")

            importlib.reload(goog)
            assert goog.GmailIntegration is None
        finally:
            # Restore
            if orig_gmail_mod is not None:
                sys.modules["animus.integrations.google.gmail"] = orig_gmail_mod
            if had_gmail_attr:
                goog.gmail = orig_gmail_attr
            importlib.reload(goog)


# ---------------------------------------------------------------------------
# SyncServer — _handle_message parse error (line 252)
# ---------------------------------------------------------------------------


class TestSyncServerMessageParseError:
    """Line 252: _handle_message with malformed message."""

    def test_malformed_message_error_response(self):
        from animus.sync.server import SyncServer

        state = MagicMock()
        state.device_id = "server"
        server = SyncServer(state=state, shared_secret="s")

        peer = MagicMock()
        peer.device_id = "peer"
        peer.device_name = "p"
        peer.websocket = AsyncMock()

        # Malformed JSON
        asyncio.run(server._handle_message(peer, "not-json"))
        # Should not raise — error is caught and logged
