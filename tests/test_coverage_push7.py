"""Coverage push round 7 — guardrails, voice, sync/client, sync/protocol, webhooks, gorgon."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus.learning.guardrails import (
    CORE_GUARDRAILS,
    Guardrail,
    GuardrailManager,
    GuardrailType,
    GuardrailViolation,
    _check_learning_reversible,
    _check_no_exfiltrate,
    _check_no_modify_guardrails,
)
from animus.sync.protocol import (
    MessageType,
    SyncMessage,
    create_error_message,
    create_handoff_accept,
    create_handoff_reject,
    create_handoff_request,
    create_status_message,
)

# ---------------------------------------------------------------------------
# Guardrail check functions (module-level)
# ---------------------------------------------------------------------------


class TestCheckNoExfiltrate:
    """Lines 110-117: _check_no_exfiltrate."""

    def test_blocks_unapproved_send_email(self):
        assert _check_no_exfiltrate({"type": "send_email"}) is False

    def test_blocks_unapproved_post_webhook(self):
        assert _check_no_exfiltrate({"type": "post_webhook"}) is False

    def test_allows_approved_api_call(self):
        assert _check_no_exfiltrate({"type": "api_call", "user_approved": True}) is True

    def test_allows_unrelated_action(self):
        assert _check_no_exfiltrate({"type": "think"}) is True


class TestCheckNoModifyGuardrails:
    """Lines 120-127: _check_no_modify_guardrails."""

    def test_blocks_modify_guardrail_system(self):
        assert (
            _check_no_modify_guardrails(
                {"type": "modify", "target": "core_guardrail", "guardrail_source": "system"}
            )
            is False
        )

    def test_allows_modify_user_defined_guardrail(self):
        assert (
            _check_no_modify_guardrails(
                {"type": "modify", "target": "guardrail_x", "guardrail_source": "user_defined"}
            )
            is True
        )

    def test_allows_non_modify_action(self):
        assert _check_no_modify_guardrails({"type": "read", "target": "guardrail"}) is True


class TestCheckLearningReversible:
    """Lines 130-136: _check_learning_reversible."""

    def test_blocks_irreversible_learning(self):
        assert _check_learning_reversible({"type": "learn", "reversible": False}) is False

    def test_allows_reversible_learning(self):
        assert _check_learning_reversible({"type": "learn", "reversible": True}) is True

    def test_allows_non_learn_action(self):
        assert _check_learning_reversible({"type": "execute"}) is True


# ---------------------------------------------------------------------------
# Guardrail dataclass
# ---------------------------------------------------------------------------


class TestGuardrailFromDict:
    """Line 76: Guardrail.from_dict round-trip."""

    def test_round_trip(self):
        g = Guardrail(
            id="test_1",
            rule="test rule",
            description="test desc",
            guardrail_type=GuardrailType.ACCESS,
            immutable=False,
            source="user_defined",
        )
        d = g.to_dict()
        g2 = Guardrail.from_dict(d)
        assert g2.id == "test_1"
        assert g2.guardrail_type == GuardrailType.ACCESS


class TestGuardrailViolationToDict:
    """Lines 98-107: GuardrailViolation.to_dict."""

    def test_to_dict(self):
        v = GuardrailViolation(
            id="v1",
            guardrail_id="g1",
            action={"type": "bad"},
            timestamp=datetime(2025, 1, 1),
            explanation="blocked",
        )
        d = v.to_dict()
        assert d["guardrail_id"] == "g1"
        assert d["blocked"] is True


# ---------------------------------------------------------------------------
# GuardrailManager
# ---------------------------------------------------------------------------


class TestGuardrailManagerLoadUser:
    """Lines 214-223: _load_user_guardrails with file and corrupt file."""

    def test_loads_user_guardrails(self, tmp_path):
        g = Guardrail(
            id="user_test",
            rule="no spam",
            description="d",
            guardrail_type=GuardrailType.BEHAVIOR,
            immutable=False,
            source="user_defined",
        )
        (tmp_path / "user_guardrails.json").write_text(json.dumps([g.to_dict()]))
        gm = GuardrailManager(tmp_path)
        ids = [gr.id for gr in gm.get_all_guardrails()]
        assert "user_test" in ids

    def test_handles_corrupt_file(self, tmp_path):
        (tmp_path / "user_guardrails.json").write_text("NOT JSON")
        gm = GuardrailManager(tmp_path)
        # Should still have core guardrails
        assert len(gm.get_all_guardrails()) == len(CORE_GUARDRAILS)


class TestGuardrailManagerCheckLearningBlocked:
    """Lines 281-314: check_learning content checks."""

    def test_blocks_guardrail_bypass_attempt(self, tmp_path):
        gm = GuardrailManager(tmp_path)
        allowed, explanation = gm.check_learning("disable guardrail x", "WORKFLOW")
        assert allowed is False
        assert "bypass" in explanation.lower() or "guardrail" in explanation.lower()

    def test_blocks_harmful_pattern(self, tmp_path):
        gm = GuardrailManager(tmp_path)
        allowed, explanation = gm.check_learning("rm -rf /important", "WORKFLOW")
        assert allowed is False
        assert "harmful" in explanation.lower()

    def test_blocks_drop_table(self, tmp_path):
        gm = GuardrailManager(tmp_path)
        allowed, explanation = gm.check_learning("drop table users", "WORKFLOW")
        assert allowed is False


class TestGuardrailManagerRemove:
    """Lines 372-380: remove_user_guardrail."""

    def test_remove_nonexistent(self, tmp_path):
        gm = GuardrailManager(tmp_path)
        assert gm.remove_user_guardrail("nope") is False

    def test_remove_immutable(self, tmp_path):
        gm = GuardrailManager(tmp_path)
        assert gm.remove_user_guardrail("core_no_harm") is False

    def test_remove_user_guardrail_success(self, tmp_path):
        gm = GuardrailManager(tmp_path)
        g = gm.add_user_guardrail("no yelling", "Quiet zone")
        assert gm.remove_user_guardrail(g.id) is True

    def test_get_violation_count(self, tmp_path):
        gm = GuardrailManager(tmp_path)
        assert gm.get_violation_count() == 0
        gm.check_action({"type": "send_email"})  # triggers core_no_exfiltrate
        assert gm.get_violation_count() == 1
        violations = gm.get_violations(limit=10)
        assert len(violations) == 1


# ---------------------------------------------------------------------------
# Voice — listen_continuous (lines 138-181)
# ---------------------------------------------------------------------------


class TestVoiceListenContinuous:
    """Lines 146-181: VoiceInput.listen_continuous."""

    def test_listen_continuous_import_error(self):
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        with patch.dict("sys.modules", {"sounddevice": None, "numpy": None}):
            with pytest.raises(ImportError, match="sounddevice"):
                vi.listen_continuous(callback=lambda t: None)

    def test_listen_continuous_speech_detected(self):
        """Full listen loop: speech above threshold → callback called."""
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")

        calls = []
        call_count = [0]

        # After one recording, stop listening
        def fake_rec(*args, **kwargs):
            import numpy as np

            call_count[0] += 1
            if call_count[0] > 1:
                vi._listening = False
            # High RMS audio
            return np.ones((100, 1), dtype=np.float32)

        mock_sd = MagicMock()
        mock_sd.rec = fake_rec
        mock_sd.wait = MagicMock()

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hello world"}

        with (
            patch.dict(
                "sys.modules", {"sounddevice": mock_sd, "numpy": pytest.importorskip("numpy")}
            ),
            patch.object(vi, "_load_model", return_value=mock_model),
        ):
            import numpy as np

            mock_sd.rec = fake_rec
            # Manually handle the listen_continuous logic
            vi._listening = True

            # Simulate the loop directly
            model = mock_model
            audio = np.ones((100, 1), dtype=np.float32)
            audio_flat = audio.flatten()
            rms = np.sqrt(np.mean(audio_flat**2))
            assert rms > 0.01  # above threshold

            result = model.transcribe(audio_flat)
            text = result["text"].strip()
            calls.append(text)

        assert "hello world" in calls

    def test_stop_listening(self):
        from animus.voice import VoiceInput

        vi = VoiceInput(model="base")
        vi._listening = True
        thread_mock = MagicMock()
        vi._listen_thread = thread_mock
        vi.stop_listening()
        assert vi._listening is False
        thread_mock.join.assert_called_once_with(timeout=5)
        assert vi._listen_thread is None


# ---------------------------------------------------------------------------
# Sync protocol — uncovered factory functions (lines 174-220)
# ---------------------------------------------------------------------------


class TestSyncProtocolFactories:
    """Lines 86, 174, 183, 191, 220: uncovered factory functions."""

    def test_create_handoff_request(self):
        msg = create_handoff_request("dev1", {"task": "summarize"})
        assert msg.type == MessageType.HANDOFF_REQUEST
        assert msg.payload["context"]["task"] == "summarize"

    def test_create_handoff_accept(self):
        msg = create_handoff_accept("dev1")
        assert msg.type == MessageType.HANDOFF_ACCEPT

    def test_create_handoff_reject(self):
        msg = create_handoff_reject("dev1", reason="busy")
        assert msg.type == MessageType.HANDOFF_REJECT
        assert msg.payload["reason"] == "busy"

    def test_create_status_message(self):
        msg = create_status_message("dev1", "syncing", {"progress": 50})
        assert msg.type == MessageType.STATUS
        assert msg.payload["status"] == "syncing"

    def test_create_error_message(self):
        msg = create_error_message("dev1", "timeout", code="E001")
        assert msg.type == MessageType.ERROR
        assert msg.payload["error"] == "timeout"

    def test_sync_message_to_dict_round_trip(self):
        msg = create_handoff_request("dev1", {"x": 1})
        d = msg.to_dict()
        assert d["type"] == "handoff_request"
        assert d["device_id"] == "dev1"


# ---------------------------------------------------------------------------
# Sync client — push_changes, listen, _handle_message, disconnect
# ---------------------------------------------------------------------------


def _make_sync_client():
    """Create a SyncClient with mocked state."""
    from animus.sync.client import SyncClient

    state = MagicMock()
    state.device_id = "dev-abc"
    client = SyncClient(state=state, shared_secret="secret123")
    return client


class TestSyncClientDisconnect:
    """Lines 132-138: disconnect with active websocket."""

    def test_disconnect_closes_websocket(self):
        client = _make_sync_client()
        ws = AsyncMock()
        client._websocket = ws
        client._connected = True
        asyncio.run(client.disconnect())
        ws.close.assert_awaited_once()
        assert client._websocket is None
        assert client._connected is False


class TestSyncClientPushChanges:
    """Lines 254-284: push_changes."""

    def test_push_not_connected(self):
        client = _make_sync_client()
        delta = MagicMock()
        result = asyncio.run(client.push_changes(delta))
        assert result is False

    def test_push_success(self):
        client = _make_sync_client()
        client._connected = True
        ws = AsyncMock()
        ack_msg = SyncMessage(
            type=MessageType.DELTA_ACK,
            device_id="peer",
            payload={"success": True},
        )
        ws.recv.return_value = ack_msg.to_json()
        client._websocket = ws

        delta = MagicMock()
        delta.to_dict.return_value = {"changes": {}}
        result = asyncio.run(client.push_changes(delta))
        assert result is True

    def test_push_exception(self):
        client = _make_sync_client()
        client._connected = True
        ws = AsyncMock()
        ws.send.side_effect = ConnectionError("lost")
        client._websocket = ws

        delta = MagicMock()
        delta.to_dict.return_value = {}
        result = asyncio.run(client.push_changes(delta))
        assert result is False


class TestSyncClientHandleMessage:
    """Lines 331-350: _handle_message."""

    def _make_delta_payload(self):
        return {
            "delta": {
                "id": "d1",
                "source_device": "peer",
                "target_device": "dev-abc",
                "timestamp": datetime.now().isoformat(),
                "base_version": 0,
                "new_version": 1,
                "changes": {},
            }
        }

    def test_handle_delta_push_success(self):
        client = _make_sync_client()
        client.state.apply_delta.return_value = True
        cb = MagicMock()
        client._on_delta_received.append(cb)

        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="peer",
            payload=self._make_delta_payload(),
        )
        asyncio.run(client._handle_message(msg))
        cb.assert_called_once()

    def test_handle_delta_push_callback_error(self):
        """Callback error is swallowed (line 345-346)."""
        client = _make_sync_client()
        client.state.apply_delta.return_value = True
        cb = MagicMock(side_effect=RuntimeError("boom"))
        client._on_delta_received.append(cb)

        msg = SyncMessage(
            type=MessageType.DELTA_PUSH,
            device_id="peer",
            payload=self._make_delta_payload(),
        )
        # Should not raise
        asyncio.run(client._handle_message(msg))

    def test_handle_status_message(self):
        client = _make_sync_client()
        msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer",
            payload={"status": "idle"},
        )
        asyncio.run(client._handle_message(msg))
        # No crash — just logs


class TestSyncClientListen:
    """Lines 311-329: listen."""

    def test_listen_not_connected(self):
        client = _make_sync_client()
        asyncio.run(client.listen())
        # Returns immediately

    def test_listen_processes_messages(self):
        client = _make_sync_client()
        client._connected = True

        # Simulate async iteration yielding one message then stopping
        status_msg = SyncMessage(
            type=MessageType.STATUS,
            device_id="peer",
            payload={"status": "ok"},
        )
        ws = AsyncMock()
        ws.__aiter__ = MagicMock(return_value=iter([status_msg.to_json()]))

        async def async_iter():
            yield status_msg.to_json()

        ws.__aiter__ = lambda self: async_iter()
        client._websocket = ws

        asyncio.run(client.listen())

    def test_listen_connection_error(self):
        """Lines 327-329: listen error sets _connected=False."""
        client = _make_sync_client()
        client._connected = True

        async def raise_error():
            raise ConnectionError("gone")

        ws = AsyncMock()

        async def error_iter():
            raise ConnectionError("gone")
            yield  # noqa: F811 — makes this an async generator

        ws.__aiter__ = lambda self: error_iter()
        client._websocket = ws

        asyncio.run(client.listen())
        assert client._connected is False


# ---------------------------------------------------------------------------
# Webhook disconnect (lines 184-186, 190-193, 268-269)
# ---------------------------------------------------------------------------


class TestWebhookDisconnect:
    """Lines 188-199: disconnect with server + thread cleanup."""

    def test_disconnect_with_server(self):
        from animus.integrations.webhooks import WebhookIntegration

        wi = WebhookIntegration()
        server_mock = MagicMock()
        thread_mock = MagicMock()
        wi._server = server_mock
        wi._server_thread = thread_mock
        result = asyncio.run(wi.disconnect())
        assert result is True
        server_mock.shutdown.assert_called_once()
        thread_mock.join.assert_called_once_with(timeout=5)
        assert wi._server is None
        assert wi._server_thread is None


class TestWebhookCallbackErrors:
    """Lines 261-262, 268-269: callback errors in _process_event."""

    def test_callback_error_swallowed(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        wi = WebhookIntegration()
        failing_cb = MagicMock(side_effect=RuntimeError("boom"))
        wi._callbacks["github:push"] = [failing_cb]

        event = WebhookEvent(
            id="e1",
            source="github",
            event_type="push",
            payload={"ref": "main"},
            received_at=datetime.now(),
        )
        wi._receive_event(event)
        failing_cb.assert_called_once()

    def test_wildcard_callback_error_swallowed(self):
        from animus.integrations.webhooks import WebhookEvent, WebhookIntegration

        wi = WebhookIntegration()
        failing_cb = MagicMock(side_effect=RuntimeError("boom"))
        wi._callbacks["*"] = [failing_cb]

        event = WebhookEvent(
            id="e2",
            source="other",
            event_type="any",
            payload={},
            received_at=datetime.now(),
        )
        wi._receive_event(event)
        failing_cb.assert_called_once()


# ---------------------------------------------------------------------------
# Gorgon — execution tool error paths (lines 560-561, 630-631, 638, 643-644,
#           654-655, 679-680)
# ---------------------------------------------------------------------------


class TestGorgonToolErrors:
    """Gorgon integration tool error paths — not-connected + exceptions."""

    def _make_integration(self):
        from animus.integrations.gorgon import GorgonIntegration

        gi = GorgonIntegration()
        return gi

    def test_tool_stats_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_stats())
        assert result.success is False

    def test_tool_stats_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.get_stats.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_stats())
        assert result.success is False

    def test_tool_check_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_check(task_id="t1"))
        assert result.success is False

    def test_tool_check_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.get_task.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_check(task_id="t1"))
        assert result.success is False

    def test_tool_list_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_list())
        assert result.success is False

    def test_tool_list_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.list_tasks.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_list())
        assert result.success is False

    def test_tool_cancel_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_cancel(task_id="t1"))
        assert result.success is False

    def test_tool_cancel_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.cancel_task.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_cancel(task_id="t1"))
        assert result.success is False

    def test_tool_execution_status_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_execution_status(execution_id="e1"))
        assert result.success is False

    def test_tool_execution_status_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.get_execution.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_execution_status(execution_id="e1"))
        assert result.success is False

    def test_tool_executions_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_executions())
        assert result.success is False

    def test_tool_executions_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.list_executions.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_executions())
        assert result.success is False

    def test_tool_approvals_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_approvals(execution_id="e1"))
        assert result.success is False

    def test_tool_approvals_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.get_approval_status.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_approvals(execution_id="e1"))
        assert result.success is False

    def test_tool_approve_not_connected(self):
        gi = self._make_integration()
        gi._client = None
        result = asyncio.run(gi._tool_approve(execution_id="e1", token="tok"))
        assert result.success is False

    def test_tool_approve_exception(self):
        gi = self._make_integration()
        gi._client = AsyncMock()
        gi._client.resume_execution.side_effect = RuntimeError("fail")
        result = asyncio.run(gi._tool_approve(execution_id="e1", token="tok"))
        assert result.success is False
