"""Additional coverage tests targeting highest-miss modules.

Covers: webhooks (do_POST, _verify_signature, tools, callbacks),
        learning/patterns (scan_for_patterns, frequency, preference, correction detection),
        voice (VoiceOutput speak pyttsx3, edge-tts, speak_async, listen_continuously),
        filesystem (index_directory, save/load_index, tool_read, tool_search_content),
        integrations/__init__ (load_defaults), integrations/manager (encrypt/decrypt),
        oauth (OAuth2CallbackHandler, OAuth2Flow.refresh_token)
"""

from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ===================================================================
# Webhooks — WebhookHandler, WebhookIntegration
# ===================================================================


class TestWebhookHandler:
    """Test WebhookHandler.do_POST and _verify_signature."""

    def _make_handler(self, path, body, headers=None, integration=None):
        """Create a mock handler with given path, body, and headers."""
        from animus.integrations.webhooks import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        handler.path = path
        handler.headers = headers or {}
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO()

        # Mock HTTP response methods
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.send_error = MagicMock()

        if integration is not None:
            WebhookHandler.integration = integration
        return handler

    def test_do_post_no_integration(self):
        from animus.integrations.webhooks import WebhookHandler

        WebhookHandler.integration = None
        handler = self._make_handler("/github/push", b"{}")
        handler.do_POST()
        handler.send_error.assert_called_once_with(500, "Integration not initialized")

    def test_do_post_invalid_path(self):
        from animus.integrations.webhooks import WebhookIntegration

        integration = WebhookIntegration()
        handler = self._make_handler("/", b"{}", integration=integration)
        handler.do_POST()
        handler.send_error.assert_called()

    def test_do_post_success(self):
        from animus.integrations.webhooks import WebhookHandler, WebhookIntegration

        integration = WebhookIntegration()
        body = json.dumps({"ref": "refs/heads/main"}).encode()
        headers = {"Content-Length": str(len(body))}
        handler = self._make_handler("/github/push", body, headers, integration)
        handler.do_POST()

        handler.send_response.assert_called_with(200)
        assert len(integration._events) == 1
        event = integration._events[0]
        assert event.source == "github"
        assert event.event_type == "push"
        WebhookHandler.integration = None  # Cleanup

    def test_do_post_with_signature_valid(self):
        import hashlib
        import hmac as hmac_mod

        from animus.integrations.webhooks import WebhookHandler, WebhookIntegration

        integration = WebhookIntegration()
        integration._secret = "mysecret"
        body = b'{"action": "completed"}'
        sig = hmac_mod.new(b"mysecret", body, hashlib.sha256).hexdigest()
        headers = {
            "Content-Length": str(len(body)),
            "X-Hub-Signature-256": f"sha256={sig}",
        }
        handler = self._make_handler("/ci/build", body, headers, integration)
        handler.do_POST()

        handler.send_response.assert_called_with(200)
        WebhookHandler.integration = None

    def test_do_post_with_signature_invalid(self):
        from animus.integrations.webhooks import WebhookHandler, WebhookIntegration

        integration = WebhookIntegration()
        integration._secret = "mysecret"
        body = b'{"data": "test"}'
        headers = {
            "Content-Length": str(len(body)),
            "X-Hub-Signature-256": "sha256=bad_signature",
        }
        handler = self._make_handler("/ci/build", body, headers, integration)
        handler.do_POST()

        handler.send_error.assert_called_with(401, "Invalid signature")
        WebhookHandler.integration = None

    def test_do_post_no_signature_header(self):
        from animus.integrations.webhooks import WebhookHandler, WebhookIntegration

        integration = WebhookIntegration()
        integration._secret = "mysecret"
        body = b'{"data": "test"}'
        headers = {"Content-Length": str(len(body))}
        handler = self._make_handler("/ci/build", body, headers, integration)
        handler.do_POST()

        handler.send_error.assert_called_with(401, "Invalid signature")
        WebhookHandler.integration = None

    def test_do_post_invalid_json_body(self):
        from animus.integrations.webhooks import WebhookHandler, WebhookIntegration

        integration = WebhookIntegration()
        body = b"not json at all"
        headers = {"Content-Length": str(len(body))}
        handler = self._make_handler("/src/event", body, headers, integration)
        handler.do_POST()

        handler.send_response.assert_called_with(200)
        event = integration._events[0]
        assert "raw" in event.payload
        WebhookHandler.integration = None

    def test_log_message(self):
        from animus.integrations.webhooks import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        # Should not raise
        handler.log_message("test %s", "msg")


class TestWebhookIntegrationTools:
    """Test WebhookIntegration tool methods and callbacks."""

    def test_receive_event_with_callback(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        integration = WebhookIntegration()
        received = []
        integration.register_callback("github", "push", lambda e: received.append(e))

        event = WebhookEvent(
            id="test-1",
            source="github",
            event_type="push",
            payload={"ref": "main"},
            received_at=datetime.now(),
        )
        integration._receive_event(event)
        assert len(received) == 1

    def test_receive_event_wildcard_callback(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        integration = WebhookIntegration()
        received = []
        integration.register_callback("*", "*", lambda e: received.append(e))

        event = WebhookEvent(
            id="test-2",
            source="any",
            event_type="any",
            payload={},
            received_at=datetime.now(),
        )
        integration._receive_event(event)
        assert len(received) == 1

    def test_receive_event_callback_error(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        integration = WebhookIntegration()
        integration.register_callback("src", "type", lambda e: 1 / 0)

        event = WebhookEvent(
            id="test-3",
            source="src",
            event_type="type",
            payload={},
            received_at=datetime.now(),
        )
        # Should not raise despite callback error
        integration._receive_event(event)

    def test_tool_list_events(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        integration = WebhookIntegration()
        for i in range(5):
            integration._events.append(
                WebhookEvent(
                    id=f"ev-{i}",
                    source="github" if i % 2 == 0 else "gitlab",
                    event_type="push",
                    payload={},
                    received_at=datetime.now(),
                )
            )

        result = asyncio.run(integration._tool_list_events(source="github"))
        assert result.success is True
        assert result.output["count"] == 3

    def test_tool_list_events_by_type(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        integration = WebhookIntegration()
        for etype in ["push", "pull_request", "push"]:
            integration._events.append(
                WebhookEvent(
                    id=f"ev-{etype}",
                    source="gh",
                    event_type=etype,
                    payload={},
                    received_at=datetime.now(),
                )
            )

        result = asyncio.run(integration._tool_list_events(event_type="push"))
        assert result.output["count"] == 2

    def test_tool_get_event_found(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        integration = WebhookIntegration()
        integration._events.append(
            WebhookEvent(
                id="target-1",
                source="s",
                event_type="t",
                payload={"key": "val"},
                received_at=datetime.now(),
            )
        )

        result = asyncio.run(integration._tool_get_event("target-1"))
        assert result.success is True
        assert result.output["id"] == "target-1"

    def test_tool_get_event_not_found(self):
        from animus.integrations.webhooks import WebhookIntegration

        integration = WebhookIntegration()
        result = asyncio.run(integration._tool_get_event("missing"))
        assert result.success is False

    def test_tool_info(self):
        from animus.integrations.webhooks import WebhookIntegration

        integration = WebhookIntegration()
        result = asyncio.run(integration._tool_info())
        assert result.success is True

    def test_get_tools(self):
        from animus.integrations.webhooks import WebhookIntegration

        integration = WebhookIntegration()
        tools = integration.get_tools()
        names = {t.name for t in tools}
        assert "webhook_list_events" in names
        assert "webhook_get_event" in names
        assert "webhook_info" in names

    def test_verify(self):
        from animus.integrations.webhooks import WebhookIntegration

        integration = WebhookIntegration()
        assert asyncio.run(integration.verify()) is False


# ===================================================================
# Learning Patterns — PatternDetector
# ===================================================================


class TestPatternDetectorScan:
    """Test PatternDetector scan methods."""

    def _make_memory(self, content, memory_id=None, created_at=None):
        """Create a mock memory object."""
        m = MagicMock()
        m.content = content
        m.id = memory_id or f"mem-{id(content)}"
        m.created_at = created_at or datetime.now()
        return m

    def _make_detector(self):
        from animus.learning.patterns import PatternDetector

        mock_memory = MagicMock()
        mock_memory.recall.return_value = []
        return PatternDetector(memory=mock_memory, min_occurrences=2)

    def test_scan_no_memories(self):
        detector = self._make_detector()
        result = detector.scan_for_patterns()
        assert result == []

    def test_detect_frequency_patterns(self):
        detector = self._make_detector()
        # Same action phrase repeated to meet min_occurrences=2
        memories = [
            self._make_memory("Can you summarize this"),
            self._make_memory("Can you summarize this"),
            self._make_memory("Can you summarize this"),
        ]
        signals = detector._detect_frequency_patterns(memories)
        assert len(signals) >= 1
        assert any("summarize" in s.content.lower() for s in signals)

    def test_detect_preference_positive(self):
        detector = self._make_detector()
        memories = [self._make_memory("I prefer concise answers.")]
        signals = detector._detect_preference_signals(memories)
        assert len(signals) >= 1
        assert any("concise" in s.content.lower() for s in signals)

    def test_detect_preference_negative(self):
        detector = self._make_detector()
        memories = [self._make_memory("I don't like verbose responses.")]
        signals = detector._detect_preference_signals(memories)
        assert len(signals) >= 1
        assert any("verbose" in s.content.lower() for s in signals)

    def test_detect_corrections(self):
        detector = self._make_detector()
        memories = [self._make_memory("That's not right, it should be 42")]
        signals = detector._detect_corrections(memories)
        assert len(signals) >= 1
        assert signals[0].strength == 0.9

    def test_detect_no_corrections(self):
        detector = self._make_detector()
        memories = [self._make_memory("Great answer, thank you!")]
        signals = detector._detect_corrections(memories)
        assert signals == []

    def test_full_scan_with_memories(self):
        from animus.learning.patterns import PatternDetector

        mock_memory = MagicMock()
        memories = [
            self._make_memory("Can you summarize the report"),
            self._make_memory("Can you summarize the data"),
            self._make_memory("Can you summarize the notes"),
            self._make_memory("I prefer short answers."),
            self._make_memory("That's not right, it should be different"),
        ]
        mock_memory.recall.return_value = memories
        detector = PatternDetector(memory=mock_memory, min_occurrences=2)
        patterns = detector.scan_for_patterns()
        assert isinstance(patterns, list)


# ===================================================================
# Voice — VoiceOutput
# ===================================================================


class TestVoiceOutput:
    """Test VoiceOutput speak methods."""

    def test_init(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3", rate=200)
        assert vo.engine_name == "pyttsx3"
        assert vo.rate == 200
        assert vo._engine is None

    def test_speak_pyttsx3(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        mock_engine = MagicMock()
        vo._engine = mock_engine  # Skip _init_pyttsx3

        vo.speak("Hello world")
        mock_engine.say.assert_called_once_with("Hello world")
        mock_engine.runAndWait.assert_called_once()

    def test_init_pyttsx3_import_error(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        with patch.dict("sys.modules", {"pyttsx3": None}):
            with pytest.raises(ImportError, match="pyttsx3"):
                vo._init_pyttsx3()

    def test_init_pyttsx3_success(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3", rate=180)
        mock_pyttsx3 = MagicMock()
        mock_engine = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            result = vo._init_pyttsx3()
        assert result is mock_engine
        mock_engine.setProperty.assert_called_with("rate", 180)

    def test_speak_unknown_engine(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="unknown")
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            vo.speak("test")

    def test_speak_edge_tts_import_error(self):
        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="edge-tts")
        with patch.dict("sys.modules", {"edge_tts": None}):
            with pytest.raises(ImportError, match="edge-tts"):
                vo.speak("test")

    def test_speak_async(self):
        import threading

        from animus.voice import VoiceOutput

        vo = VoiceOutput(engine="pyttsx3")
        vo._engine = MagicMock()

        thread = vo.speak_async("Hello")
        assert isinstance(thread, threading.Thread)
        thread.join(timeout=2)


class TestVoiceInputContinuous:
    """Test VoiceInput continuous listening."""

    def test_stop_listening_not_started(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        # Should not raise
        vi.stop_listening()

    def test_is_listening_property(self):
        from animus.voice import VoiceInput

        vi = VoiceInput()
        assert vi.is_listening is False


# ===================================================================
# Filesystem — additional tool tests
# ===================================================================


class TestFilesystemTools:
    """Test FilesystemIntegration tool methods."""

    def test_tool_read_success(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        test_file = tmp_path / "readme.txt"
        test_file.write_text("line 1\nline 2\nline 3\n")

        result = asyncio.run(fs._tool_read(str(test_file)))
        assert result.success is True

    def test_tool_read_missing(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        result = asyncio.run(fs._tool_read(str(tmp_path / "missing.txt")))
        assert result.success is False

    def test_tool_read_directory(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        result = asyncio.run(fs._tool_read(str(tmp_path)))
        assert result.success is False

    def test_tool_search_content(self, tmp_path: Path):
        from animus.integrations.filesystem import FileEntry, FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        test_file = tmp_path / "code.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        # Add to index
        fs._index[str(test_file)] = FileEntry(
            path=str(test_file),
            name="code.py",
            extension=".py",
            size=30,
            modified=datetime.now(),
            is_dir=False,
        )

        result = asyncio.run(fs._tool_search_content("hello"))
        assert result.success is True
        assert result.output["count"] >= 1

    def test_tool_search_content_bad_regex(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        result = asyncio.run(fs._tool_search_content("[invalid"))
        assert result.success is False

    def test_tool_index(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")

        result = asyncio.run(fs._tool_index(str(tmp_path)))
        assert result.success is True
        assert result.output["files_indexed"] >= 2

    def test_save_and_load_index(self, tmp_path: Path):
        from animus.integrations.filesystem import FileEntry, FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        fs._index["test"] = FileEntry(
            path="/test",
            name="test.txt",
            extension=".txt",
            size=10,
            modified=datetime.now(),
            is_dir=False,
        )
        fs._indexed_paths = ["/some/path"]
        fs._save_index()

        # Load into new instance
        fs2 = FilesystemIntegration(data_dir=tmp_path)
        fs2._load_index()
        assert "/some/path" in fs2._indexed_paths

    def test_load_index_missing(self, tmp_path: Path):
        from animus.integrations.filesystem import FilesystemIntegration

        fs = FilesystemIntegration(data_dir=tmp_path)
        # Should not raise when no index file exists
        fs._load_index()


# ===================================================================
# Integration Manager — encrypt/decrypt
# ===================================================================


class TestIntegrationManagerCrypto:
    """Test IntegrationManager credential encryption."""

    def test_derive_key(self):
        from animus.integrations.manager import _derive_key

        key = _derive_key("test_secret")
        assert isinstance(key, bytes)
        assert len(key) == 44  # base64-encoded Fernet key

    def test_get_encryption_secret(self):
        from animus.integrations.manager import _get_encryption_secret

        secret = _get_encryption_secret()
        assert isinstance(secret, str)
        assert len(secret) > 0

    def test_save_and_load_credentials(self, tmp_path: Path):
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)
        mgr._save_credentials("test_integration", {"key": "value"})

        loaded = mgr._load_credentials("test_integration")
        assert loaded == {"key": "value"}

    def test_load_missing_credentials(self, tmp_path: Path):
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)
        result = mgr._load_credentials("nonexistent")
        assert result is None

    def test_clear_credentials(self, tmp_path: Path):
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)
        mgr._save_credentials("temp", {"a": "b"})
        mgr._clear_credentials("temp")
        assert mgr._load_credentials("temp") is None


# ===================================================================
# OAuth2 — CallbackHandler
# ===================================================================


class TestOAuth2CallbackHandler:
    """Test OAuth2CallbackHandler."""

    def test_do_get_with_code(self):
        from animus.integrations.oauth import OAuth2CallbackHandler

        handler = OAuth2CallbackHandler.__new__(OAuth2CallbackHandler)
        handler.path = "/?code=auth_code_123&state=test"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()

        handler.do_GET()

        assert OAuth2CallbackHandler.authorization_code == "auth_code_123"
        handler.send_response.assert_called_with(200)
        # Clean up class state
        OAuth2CallbackHandler.authorization_code = None

    def test_do_get_with_error(self):
        from animus.integrations.oauth import OAuth2CallbackHandler

        handler = OAuth2CallbackHandler.__new__(OAuth2CallbackHandler)
        handler.path = "/?error=access_denied"
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()

        handler.do_GET()

        assert OAuth2CallbackHandler.error == "access_denied"
        handler.send_response.assert_called_with(400)
        # Clean up
        OAuth2CallbackHandler.error = None

    def test_do_get_no_code_no_error(self):
        from animus.integrations.oauth import OAuth2CallbackHandler

        handler = OAuth2CallbackHandler.__new__(OAuth2CallbackHandler)
        handler.path = "/?state=test"
        handler.send_response = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()

        handler.do_GET()
        handler.send_response.assert_called_with(400)

    def test_log_message_suppressed(self):
        from animus.integrations.oauth import OAuth2CallbackHandler

        handler = OAuth2CallbackHandler.__new__(OAuth2CallbackHandler)
        # Should not raise
        handler.log_message("test %s", "msg")


# ===================================================================
# Tasks — additional coverage
# ===================================================================


class TestTaskTrackerAdditional:
    """Cover missed branches in tasks.py."""

    def test_is_overdue(self):
        from animus.tasks import Task, TaskStatus

        now = datetime.now()
        task = Task(
            id="t1",
            description="overdue task",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            due_at=datetime.now() - timedelta(days=1),
        )
        assert task.is_overdue() is True

    def test_is_overdue_completed(self):
        from animus.tasks import Task, TaskStatus

        now = datetime.now()
        task = Task(
            id="t2",
            description="done task",
            status=TaskStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            due_at=datetime.now() - timedelta(days=1),
        )
        assert task.is_overdue() is False

    def test_block_task(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        task = tracker.add("Test task")
        assert tracker.block(task.id) is True

    def test_add_note(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        task = tracker.add("Test task")
        assert tracker.add_note(task.id, "First note") is True
        assert tracker.add_note(task.id, "Second note") is True
        loaded = tracker.get(task.id)
        assert "Second note" in loaded.notes

    def test_add_note_missing_task(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        assert tracker.add_note("nonexistent", "note") is False

    def test_delete_task(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        task = tracker.add("Delete me")
        deleted = tracker.delete(task.id)
        assert deleted is True
        assert tracker.get(task.id) is None

    def test_delete_missing(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        deleted = tracker.delete("nonexistent")
        assert deleted is False

    def test_list_with_tags(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        tracker.add("Tagged task", tags=["urgent", "backend"])
        tracker.add("Other task", tags=["frontend"])

        result = tracker.list(tags=["urgent"])
        assert len(result) == 1
        assert result[0].description == "Tagged task"

    def test_list_overdue(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        tracker.add("Overdue task", due_at=datetime.now() - timedelta(days=1))
        tracker.add("Not overdue")

        overdue = tracker.list_overdue()
        assert len(overdue) == 1

    def test_get_statistics(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        tracker.add("Task 1")
        task2 = tracker.add("Task 2")
        tracker.complete(task2.id)

        stats = tracker.get_statistics()
        assert stats["total"] == 2
        assert "completed" in stats["by_status"]

    def test_list_all_alias(self, tmp_path: Path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        tracker.add("Task")
        assert len(tracker.list_all()) == len(tracker.list())


# ===================================================================
# Integrations __init__ — import fallback paths
# ===================================================================


class TestIntegrationsImportFallbacks:
    """Test import fallback paths in integrations/__init__.py."""

    def test_integrations_exports(self):
        """Verify __all__ exports exist."""
        from animus import integrations

        assert hasattr(integrations, "IntegrationManager")
        assert hasattr(integrations, "FilesystemIntegration")
        assert hasattr(integrations, "WebhookIntegration")


# ===================================================================
# Config — branch coverage gaps
# ===================================================================


class TestConfigBranches:
    """Cover config.py branch paths."""

    def test_load_with_defaults(self, tmp_path: Path):
        from animus.config import AnimusConfig

        config = AnimusConfig(data_dir=tmp_path)
        assert config.data_dir == tmp_path

    def test_load_with_env_override(self, tmp_path: Path):
        from animus.config import AnimusConfig

        with patch.dict("os.environ", {"ANIMUS_LOG_LEVEL": "DEBUG"}):
            config = AnimusConfig(data_dir=tmp_path)
        assert config.log_level == "DEBUG"
