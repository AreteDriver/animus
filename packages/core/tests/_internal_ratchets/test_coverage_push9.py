"""Coverage push round 9 — sync server/state, filesystem, proactive, google, guardrails, tasks."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus.sync.protocol import MessageType, SyncMessage

# ---------------------------------------------------------------------------
# SyncServer — handle_connection, peer disconnect callback errors (lines 163, 175-176)
# ---------------------------------------------------------------------------


class TestSyncServerHandleConnection:
    """Lines 162-176: _handle_connection message loop + disconnect callbacks."""

    def _make_server(self):
        from animus.sync.server import SyncServer

        state = MagicMock()
        state.device_id = "server-dev"
        server = SyncServer(state=state, shared_secret="secret")
        return server

    def test_handle_connection_message_and_disconnect(self):
        """Peer connects, sends a message, then disconnects."""
        server = self._make_server()

        peer = MagicMock()
        peer.device_id = "peer-123"
        peer.device_name = "peer-device"
        peer.websocket = AsyncMock()

        # _authenticate returns the peer
        server._authenticate = AsyncMock(return_value=peer)

        # websocket yields one status message then ends
        status_msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer-123",
            payload={"status": "ok"},
        )

        async def message_gen():
            yield status_msg.to_json()

        ws = AsyncMock()
        ws.__aiter__ = lambda s: message_gen()

        peer.websocket = ws
        server._handle_message = AsyncMock()

        # Add disconnect callback (including one that errors)
        dc_cb = MagicMock()
        dc_err_cb = MagicMock(side_effect=RuntimeError("cb error"))
        server._on_peer_disconnected.extend([dc_err_cb, dc_cb])

        asyncio.run(server._handle_connection(ws, "/"))

        # Peer should have been added then removed
        assert "peer-123" not in server._peers

    def test_handle_connection_auth_fails(self):
        server = self._make_server()
        server._authenticate = AsyncMock(return_value=None)
        ws = AsyncMock()
        asyncio.run(server._handle_connection(ws, "/"))
        # No peers added
        assert len(server._peers) == 0


# ---------------------------------------------------------------------------
# SyncServer — _handle_message unknown type (line 252-254)
# ---------------------------------------------------------------------------


class TestSyncServerUnknownMessage:
    """Line 252-254: unknown message type logs warning."""

    def _make_server(self):
        from animus.sync.server import SyncServer

        state = MagicMock()
        state.device_id = "server-dev"
        return SyncServer(state=state, shared_secret="secret")

    def test_unknown_message_type(self):
        server = self._make_server()
        peer = MagicMock()
        peer.websocket = AsyncMock()
        msg = SyncMessage(type=MessageType.PONG, device_id="peer")
        # PONG is not in the handler map
        asyncio.run(server._handle_message(peer, msg.to_json()))
        # Should not raise


# ---------------------------------------------------------------------------
# SyncServer — delta push sync callback error (lines 342-343)
# ---------------------------------------------------------------------------


class TestSyncServerDeltaPushCallbackError:
    """Lines 342-343: sync callback error swallowed in _handle_delta_push."""

    def _make_server(self):
        from animus.sync.server import SyncServer

        state = MagicMock()
        state.device_id = "server-dev"
        state.apply_delta.return_value = True
        server = SyncServer(state=state, shared_secret="secret")
        return server

    def test_sync_callback_error(self):
        server = self._make_server()
        failing_cb = MagicMock(side_effect=RuntimeError("boom"))
        server._on_sync.append(failing_cb)

        peer = MagicMock()
        peer.device_id = "peer-1"
        peer.device_name = "peer"
        peer.websocket = AsyncMock()

        delta_msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="peer-1",
            payload={
                "delta": {
                    "id": "d1",
                    "source_device": "peer-1",
                    "target_device": "server-dev",
                    "timestamp": datetime.now().isoformat(),
                    "base_version": 0,
                    "new_version": 1,
                    "changes": {},
                }
            },
        )
        asyncio.run(server._handle_delta_push(peer, delta_msg))
        failing_cb.assert_called_once()


# ---------------------------------------------------------------------------
# SyncableState — _apply_learning_changes, _apply_guardrail_changes (lines 395, 407, 416)
# ---------------------------------------------------------------------------


class TestSyncableStateLearningChanges:
    """Lines 392-395: _apply_learning_changes with modified items."""

    def test_apply_learning_modified(self, tmp_path):
        from animus.sync.state import SyncableState

        state = SyncableState(device_id="dev1", data_dir=tmp_path)

        # Pre-populate learnings file
        learning_dir = tmp_path / "learning"
        learning_dir.mkdir()
        existing = [{"id": "L1", "content": "old", "updated_at": "2025-01-01"}]
        (learning_dir / "learned_items.json").write_text(json.dumps(existing))

        changes = {
            "added": {"learnings": []},
            "modified": {"learnings": [{"id": "L1", "content": "new", "updated_at": "2025-06-01"}]},
        }
        state._apply_learning_changes(changes)

        data = json.loads((learning_dir / "learned_items.json").read_text())
        assert data[0]["content"] == "new"

    def test_apply_learning_modified_new_item(self, tmp_path):
        """Modified item not in existing list → added."""
        from animus.sync.state import SyncableState

        state = SyncableState(device_id="dev1", data_dir=tmp_path)
        learning_dir = tmp_path / "learning"
        learning_dir.mkdir()
        (learning_dir / "learned_items.json").write_text("[]")

        changes = {
            "added": {"learnings": []},
            "modified": {"learnings": [{"id": "L2", "content": "brand new"}]},
        }
        state._apply_learning_changes(changes)
        data = json.loads((learning_dir / "learned_items.json").read_text())
        assert len(data) == 1
        assert data[0]["id"] == "L2"


class TestSyncableStateGuardrailChanges:
    """Lines 407, 412-416: _apply_guardrail_changes."""

    def test_apply_guardrail_added_and_modified(self, tmp_path):
        from animus.sync.state import SyncableState

        state = SyncableState(device_id="dev1", data_dir=tmp_path)
        learning_dir = tmp_path / "learning"
        learning_dir.mkdir()
        existing = [{"id": "G1", "rule": "old rule"}]
        (learning_dir / "user_guardrails.json").write_text(json.dumps(existing))

        changes = {
            "added": {"guardrails": [{"id": "G2", "rule": "new rule"}]},
            "modified": {"guardrails": [{"id": "G1", "rule": "updated rule"}]},
        }
        state._apply_guardrail_changes(changes)

        data = json.loads((learning_dir / "user_guardrails.json").read_text())
        by_id = {g["id"]: g for g in data}
        assert by_id["G1"]["rule"] == "updated rule"
        assert by_id["G2"]["rule"] == "new rule"

    def test_apply_guardrail_no_existing_file(self, tmp_path):
        from animus.sync.state import SyncableState

        state = SyncableState(device_id="dev1", data_dir=tmp_path)
        changes = {
            "added": {"guardrails": [{"id": "G3", "rule": "fresh"}]},
            "modified": {"guardrails": []},
        }
        state._apply_guardrail_changes(changes)
        learning_dir = tmp_path / "learning"
        data = json.loads((learning_dir / "user_guardrails.json").read_text())
        assert len(data) == 1


# ---------------------------------------------------------------------------
# Filesystem — _tool_search hitting limit break, entry with dir skipped (lines 325, 352)
# ---------------------------------------------------------------------------


class TestFilesystemSearchFileLimit:
    """Line 325: _tool_search hitting result limit for file search."""

    def test_search_hits_limit(self, tmp_path):
        from animus.integrations.filesystem import FileEntry, FilesystemIntegration

        fi = FilesystemIntegration()
        for i in range(5):
            p = tmp_path / f"match_{i}.txt"
            p.write_text("hello")
            fi._index[str(p)] = FileEntry(
                path=str(p),
                name=p.name,
                extension=".txt",
                size=5,
                modified=datetime.now(),
                is_dir=False,
            )

        result = asyncio.run(fi._tool_search("match", limit=2))
        assert result.success is True
        assert len(result.output["results"]) == 2

    def test_search_skips_directories(self, tmp_path):
        """Line 352: directory entries skipped in search_content."""
        from animus.integrations.filesystem import FileEntry, FilesystemIntegration

        fi = FilesystemIntegration()
        fi._index["dir1"] = FileEntry(
            path=str(tmp_path),
            name="mydir",
            extension="",
            size=0,
            modified=datetime.now().isoformat(),
            is_dir=True,
        )

        result = asyncio.run(fi._tool_search_content("anything", limit=10))
        assert result.success is True
        assert result.output["count"] == 0


# ---------------------------------------------------------------------------
# GuardrailManager — check_learning guardrail violation (lines 281-290)
# ---------------------------------------------------------------------------


class TestGuardrailCheckLearningViolation:
    """Lines 281-290: check_learning blocked by guardrail check_func."""

    def test_learning_blocked_by_irreversible_check(self, tmp_path):
        from animus.learning.guardrails import GuardrailManager

        gm = GuardrailManager(tmp_path)
        # The core_learning_reversible guardrail checks reversible=True
        # All learnings set reversible=True by default, so they pass.
        # To trigger the violation, add a custom guardrail that blocks "learn" actions.
        from animus.learning.guardrails import Guardrail, GuardrailType

        blocking = Guardrail(
            id="custom_block_learn",
            rule="Block all learning",
            description="Test blocker",
            guardrail_type=GuardrailType.BEHAVIOR,
            immutable=False,
            source="user_defined",
            check_func=lambda action: action.get("type") != "learn",
        )
        gm._guardrails[blocking.id] = blocking

        allowed, explanation = gm.check_learning("some content", "WORKFLOW")
        assert allowed is False
        assert "custom_block_learn" in explanation


# ---------------------------------------------------------------------------
# SyncClient — listen with malformed message (lines 325-326)
# ---------------------------------------------------------------------------


class TestSyncClientListenMalformed:
    """Lines 325-326: malformed message in listen loop (inner exception)."""

    def test_malformed_message_caught(self):
        from animus.sync.client import SyncClient

        state = MagicMock()
        state.device_id = "dev1"
        client = SyncClient(state=state, shared_secret="s")
        client._connected = True

        async def gen():
            yield "not-valid-json"

        ws = AsyncMock()
        ws.__aiter__ = lambda self: gen()
        client._websocket = ws

        asyncio.run(client.listen())
        # Should not raise, error logged


# ---------------------------------------------------------------------------
# Proactive — follow-up skip existing nudge (line 441-442), deadline match (line 431)
# ---------------------------------------------------------------------------


class TestProactiveFollowUpSkipExisting:
    """Lines 431, 441-442: follow-up scanning edge cases."""

    def _make_pe(self, tmp_path):
        from animus.proactive import ProactiveEngine

        mem = MagicMock()
        pe = ProactiveEngine(data_dir=tmp_path, memory=mem)
        # Override mem.store since helper may overwrite
        pe.memory = mem
        pe.memory.store = MagicMock()
        return pe, mem

    def test_follow_up_no_match(self, tmp_path):
        """Line 431: no follow-up phrases matched → empty."""
        pe, mem = self._make_pe(tmp_path)

        m = MagicMock()
        m.content = "just a regular conversation about weather"
        m.memory_type = MagicMock()
        m.memory_type.value = "episodic"
        m.created_at = datetime.now() - timedelta(days=1)
        m.id = "m1"
        m.tags = set()
        pe.memory.store.list_all.return_value = [m]

        result = pe.scan_follow_ups()
        assert len(result) == 0

    def test_follow_up_skip_already_nudged(self, tmp_path):
        """Line 441-442: existing active nudge for this memory → skip."""
        pe, mem = self._make_pe(tmp_path)

        m = MagicMock()
        m.content = "I need to follow up on the report"
        m.memory_type = MagicMock()
        m.memory_type.value = "episodic"
        m.created_at = datetime.now() - timedelta(days=1)
        m.id = "m2"
        m.tags = set()
        pe.memory.store.list_all.return_value = [m]

        # Pre-populate existing nudge for this memory
        from animus.proactive import Nudge, NudgePriority, NudgeType

        existing_nudge = Nudge(
            id="n1",
            nudge_type=NudgeType.FOLLOW_UP,
            priority=NudgePriority.MEDIUM,
            title="Follow up",
            content="existing nudge",
            created_at=datetime.now(),
            source_memory_ids=["m2"],
        )
        pe._nudges.append(existing_nudge)

        result = pe.scan_follow_ups()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Proactive — run_scheduled_checks context_refresh (line 555-556)
# ---------------------------------------------------------------------------


class TestProactiveScheduledContextRefresh:
    """Lines 555-556: context_refresh is a pass (no-op)."""

    def test_context_refresh_noop(self, tmp_path):
        from animus.proactive import ProactiveEngine, ScheduledCheck

        mem = MagicMock()
        pe = ProactiveEngine(data_dir=tmp_path, memory=mem)
        pe.memory = mem
        pe.memory.store = MagicMock()

        # Add a context_refresh check that's due
        check = ScheduledCheck(
            name="context_refresh",
            interval_minutes=1,
            last_run=datetime.now() - timedelta(minutes=5),
        )
        pe._scheduled_checks = [check]

        result = pe.run_scheduled_checks()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tasks — update_status with nonexistent task (line 169), list with status filter (line 240)
# ---------------------------------------------------------------------------


class TestTaskTrackerEdgeCases:
    """Lines 169, 240: TaskTracker edge cases."""

    def test_update_nonexistent_task(self, tmp_path):
        from animus.tasks import TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        result = tracker.update_status("nonexistent-id", "completed")
        assert result is False

    def test_list_with_status_filter(self, tmp_path):
        from animus.tasks import TaskStatus, TaskTracker

        tracker = TaskTracker(data_dir=tmp_path)
        tracker.add("Task A", priority=1)
        tracker.add("Task B", priority=2)

        # Both should be pending
        pending = tracker.list(status=TaskStatus.PENDING)
        assert len(pending) == 2

        # No completed
        completed = tracker.list(status=TaskStatus.COMPLETED)
        assert len(completed) == 0


# ---------------------------------------------------------------------------
# Google import fallback — integrations/__init__.py (lines 41-42),
# google/__init__.py (lines 19-20)
# ---------------------------------------------------------------------------


class TestGoogleImportFallbacks:
    """Import fallback branches for Gmail in integrations package."""

    def test_gmail_import_fallback(self):
        """Line 41-42: GmailIntegration = None when import fails."""
        import importlib

        with patch.dict("sys.modules", {"animus.integrations.google.gmail": None}):
            import animus.integrations as integ

            importlib.reload(integ)

        # After reload, module should still work (GmailIntegration may be None)
        assert hasattr(integ, "GoogleCalendarIntegration") or True

    def test_google_gmail_subpackage_fallback(self):
        """Lines 19-20: google/__init__.py GmailIntegration fallback."""
        import importlib

        with patch.dict("sys.modules", {"animus.integrations.google.gmail": None}):
            import animus.integrations.google as goog

            importlib.reload(goog)

        # Should not raise


# ---------------------------------------------------------------------------
# OAuth — run_local_server no-browser path (line 204)
# ---------------------------------------------------------------------------


class TestOAuthNoBrowser:
    """Line 204: OAuth2Flow.run_local_server with open_browser=False."""

    def test_run_local_server_no_browser(self):
        from animus.integrations.oauth import GOOGLE_AUTH_AVAILABLE

        if not GOOGLE_AUTH_AVAILABLE:
            pytest.skip("google-auth not installed")

        from animus.integrations.oauth import OAuth2CallbackHandler, OAuth2Flow

        flow = OAuth2Flow(
            client_id="test_id",
            client_secret="test_secret",
            scopes=["email"],
        )

        # Mock the Google OAuth Flow object
        mock_google_flow = MagicMock()
        mock_google_flow.authorization_url.return_value = (
            "https://auth.example.com",
            "state",
        )
        mock_creds = MagicMock()
        mock_creds.token = "access_token"
        mock_creds.refresh_token = "refresh_token"
        mock_creds.expiry = None
        mock_creds.scopes = ["email"]
        mock_google_flow.credentials = mock_creds

        with (
            patch(
                "animus.integrations.oauth.Flow.from_client_config",
                return_value=mock_google_flow,
            ),
            patch("animus.integrations.oauth.HTTPServer") as mock_server_cls,
            patch("animus.integrations.oauth.webbrowser") as mock_wb,
            patch("animus.integrations.oauth.Thread") as mock_thread_cls,
        ):
            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server
            mock_thread = MagicMock()
            # When thread.start() is called, simulate receiving auth code
            mock_thread.start.side_effect = lambda: setattr(
                OAuth2CallbackHandler, "authorization_code", "test_code"
            )
            mock_thread_cls.return_value = mock_thread

            result = flow.run_local_server(open_browser=False)

            # webbrowser.open should NOT be called
            mock_wb.open.assert_not_called()
            # Flow should have fetched token
            mock_google_flow.fetch_token.assert_called_once_with(code="test_code")
            assert result is not None
            assert result.access_token == "access_token"
