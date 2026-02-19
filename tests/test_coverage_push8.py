"""Coverage push round 8 — manager credentials, swarm engine edge cases, sync client, webhooks, filesystem."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus.forge.models import AgentConfig, GateConfig, GateFailedError, StepResult, WorkflowConfig
from animus.swarm.engine import SwarmEngine
from animus.sync.protocol import MessageType, SyncMessage

# ---------------------------------------------------------------------------
# Helper: mock cognitive for SwarmEngine
# ---------------------------------------------------------------------------


def _mock_cognitive(default_response="done"):
    from animus.cognitive import CognitiveLayer, ModelConfig

    config = ModelConfig.mock(default_response=default_response)
    return CognitiveLayer(config)


# ---------------------------------------------------------------------------
# IntegrationManager — credential save/load/clear (lines 257-304, 322)
# ---------------------------------------------------------------------------


class TestManagerCredentialsFallback:
    """_save_credentials and _load_credentials fallback paths (no cryptography)."""

    def test_save_and_load_base64_fallback(self, tmp_path):
        """When fernet is not available, save/load uses base64."""
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)

        with patch.object(IntegrationManager, "_fernet_available", return_value=False):
            mgr._save_credentials("test_svc", {"token": "secret123"})

            # Verify file exists and is base64
            creds_path = tmp_path / "test_svc.json"
            assert creds_path.exists()
            raw = creds_path.read_bytes()
            decoded = json.loads(base64.b64decode(raw))
            assert decoded["token"] == "secret123"

            # Load back
            loaded = mgr._load_credentials("test_svc")
            assert loaded["token"] == "secret123"

    def test_load_credentials_nonexistent(self, tmp_path):
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)
        assert mgr._load_credentials("nonexistent") is None

    def test_load_credentials_corrupt(self, tmp_path):
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)
        (tmp_path / "bad_svc.json").write_bytes(b"not-json-not-base64")

        with patch.object(IntegrationManager, "_fernet_available", return_value=False):
            result = mgr._load_credentials("bad_svc")
            assert result is None

    def test_clear_credentials(self, tmp_path):
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)

        with patch.object(IntegrationManager, "_fernet_available", return_value=False):
            mgr._save_credentials("to_clear", {"key": "val"})
            creds_path = tmp_path / "to_clear.json"
            assert creds_path.exists()

            mgr._clear_credentials("to_clear")
            assert not creds_path.exists()

    def test_clear_credentials_nonexistent(self, tmp_path):
        """Clearing non-existent credentials is a no-op."""
        from animus.integrations.manager import IntegrationManager

        mgr = IntegrationManager(data_dir=tmp_path)
        mgr._clear_credentials("nope")  # Should not raise


# ---------------------------------------------------------------------------
# SwarmEngine — all-agents-complete stage skip (lines 116-117)
# ---------------------------------------------------------------------------


class TestSwarmEngineStageSkip:
    """Stage with all agents already complete is skipped."""

    def test_all_agents_complete_skips_stage(self, tmp_path):
        cognitive = _mock_cognitive("result text")
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)

        config = WorkflowConfig(
            name="skip_test",
            agents=[
                AgentConfig(
                    name="a1", archetype="researcher", outputs=["brief"], budget_tokens=5000
                ),
            ],
            gates=[],
        )

        # Pre-populate checkpoint with completed result
        from animus.forge.models import WorkflowState

        state = WorkflowState(workflow_name="skip_test")
        state.results = [
            StepResult(
                agent_name="a1",
                success=True,
                outputs={"brief": "done"},
                tokens_used=10,
                cost_usd=0.0,
            )
        ]
        state.status = "running"
        engine._checkpoint.save_state(state)

        # Resume — all agents already done, should skip and complete
        result = engine.run(config, resume=True)
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# SwarmEngine — thread execution error in parallel stage (lines 256-261)
# ---------------------------------------------------------------------------


class TestSwarmEngineThreadError:
    """When an agent raises in a thread, it's caught as StepResult(success=False)."""

    def test_thread_error_caught(self, tmp_path):
        cognitive = _mock_cognitive("ok")
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)

        # Two agents so the thread pool is used (not single-agent fast path)
        a1 = AgentConfig(name="a1", archetype="researcher", outputs=["o1"], budget_tokens=5000)
        a2 = AgentConfig(name="a2", archetype="analyst", outputs=["o2"], budget_tokens=5000)
        agent_configs = {"a1": a1, "a2": a2}

        # Patch ForgeAgent so a2 raises in run()
        def mock_agent_factory(ac, cog, tools):
            agent = MagicMock()
            if ac.name == "a2":
                agent.run.side_effect = RuntimeError("agent crash")
            else:
                agent.run.return_value = StepResult(
                    agent_name="a1",
                    success=True,
                    outputs={"o1": "done"},
                    tokens_used=5,
                    cost_usd=0.0,
                )
            return agent

        with patch("animus.swarm.engine.ForgeAgent", side_effect=mock_agent_factory):
            results = engine._execute_stage(["a1", "a2"], agent_configs, {})

        # a2 should have success=False with thread error
        a2_result = [r for r in results if r.agent_name == "a2"][0]
        assert a2_result.success is False
        assert "Thread execution error" in a2_result.error


# ---------------------------------------------------------------------------
# SwarmEngine — gate revise path (lines 341-342)
# ---------------------------------------------------------------------------


class TestSwarmEngineGateRevise:
    """Gate with on_fail='revise' raises GateFailedError."""

    def test_gate_revise_raises(self, tmp_path):
        cognitive = _mock_cognitive("ok")
        engine = SwarmEngine(cognitive, checkpoint_dir=tmp_path)

        gate = GateConfig(
            name="quality_gate",
            after="a1",
            type="automated",
            pass_condition="false",
            on_fail="revise",
        )

        config = WorkflowConfig(
            name="revise_test",
            agents=[
                AgentConfig(name="a1", archetype="researcher", outputs=["o1"], budget_tokens=5000),
            ],
            gates=[gate],
        )

        with pytest.raises(GateFailedError, match="revise"):
            engine.run(config)


# ---------------------------------------------------------------------------
# SyncClient — sync() full flow with callbacks + errors (lines 200-203, 250-252)
# ---------------------------------------------------------------------------


def _make_sync_client():
    from animus.sync.client import SyncClient

    state = MagicMock()
    state.device_id = "dev-abc"
    state.version = 1
    state.get_peer_version.return_value = 0
    state.collect_state.return_value = {"key": "val"}
    client = SyncClient(state=state, shared_secret="secret")
    return client


class TestSyncClientSyncTimeout:
    """Line 248-249: sync timeout."""

    def test_sync_timeout(self):
        client = _make_sync_client()
        client._connected = True
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.recv = AsyncMock(side_effect=asyncio.TimeoutError())
        client._websocket = ws
        client._peer_device_id = "peer1"

        result = asyncio.run(client.sync())
        assert result.success is False
        assert result.error == "Sync timeout"


class TestSyncClientSyncException:
    """Lines 250-252: sync general exception."""

    def test_sync_exception(self):
        client = _make_sync_client()
        client._connected = True
        ws = AsyncMock()
        ws.send = AsyncMock(side_effect=RuntimeError("ws broken"))
        client._websocket = ws
        client._peer_device_id = "peer1"

        result = asyncio.run(client.sync())
        assert result.success is False
        assert "ws broken" in result.error


class TestSyncClientNotConnected:
    """Line 151-152: sync when not connected."""

    def test_sync_not_connected(self):
        client = _make_sync_client()
        result = asyncio.run(client.sync())
        assert result.success is False
        assert result.error == "Not connected"


# ---------------------------------------------------------------------------
# SyncClient — ping (lines 286-309)
# ---------------------------------------------------------------------------


class TestSyncClientPing:
    """Lines 286-309: ping method."""

    def test_ping_not_connected(self):
        client = _make_sync_client()
        result = asyncio.run(client.ping())
        assert result is None

    def test_ping_success(self):
        client = _make_sync_client()
        client._connected = True
        pong = SyncMessage(type=MessageType.PONG, device_id="peer")
        ws = AsyncMock()
        ws.recv.return_value = pong.to_json()
        client._websocket = ws

        result = asyncio.run(client.ping())
        assert isinstance(result, int)
        assert result >= 0

    def test_ping_exception(self):
        client = _make_sync_client()
        client._connected = True
        ws = AsyncMock()
        ws.send.side_effect = ConnectionError("lost")
        client._websocket = ws

        result = asyncio.run(client.ping())
        assert result is None


# ---------------------------------------------------------------------------
# Webhooks — connect failure (lines 184-186), _verify_signature (line 110)
# ---------------------------------------------------------------------------


class TestWebhookConnectFailure:
    """Lines 184-186: connect raises exception."""

    def test_connect_port_in_use(self):
        from animus.integrations.webhooks import WebhookIntegration

        wi = WebhookIntegration()
        with patch("animus.integrations.webhooks.HTTPServer", side_effect=OSError("port in use")):
            result = asyncio.run(wi.connect({"port": 9999}))
        assert result is False


class TestWebhookVerifySignature:
    """Line 110: _verify_signature returns True when no secret."""

    def test_no_secret_passes(self):
        from animus.integrations.webhooks import WebhookHandler

        handler = MagicMock(spec=WebhookHandler)
        # Call the unbound method directly
        WebhookHandler.integration = MagicMock()
        WebhookHandler.integration._secret = None
        result = WebhookHandler._verify_signature(handler, b"body", {})
        assert result is True

    def test_no_signature_header_fails(self):
        from animus.integrations.webhooks import WebhookHandler

        WebhookHandler.integration = MagicMock()
        WebhookHandler.integration._secret = "mysecret"
        handler = MagicMock(spec=WebhookHandler)
        result = WebhookHandler._verify_signature(handler, b"body", {"Content-Type": "json"})
        assert result is False


# ---------------------------------------------------------------------------
# Filesystem — _tool_index non-directory path (line 286)
# ---------------------------------------------------------------------------


class TestFilesystemToolIndex:
    """Lines 278-291: _tool_index with non-existent and non-dir paths."""

    def test_index_non_directory(self, tmp_path):
        from animus.integrations.filesystem import FilesystemIntegration

        fi = FilesystemIntegration()
        fi._index = {}
        fi._indexed_paths = []

        # Create a regular file
        f = tmp_path / "notadir.txt"
        f.write_text("hello")

        result = asyncio.run(fi._tool_index(str(f)))
        assert result.success is False
        assert "not a directory" in result.error


class TestFilesystemSearchContentLimit:
    """Lines 375-378: search_content hitting result limit."""

    def test_search_hits_limit(self, tmp_path):
        from animus.integrations.filesystem import FileEntry, FilesystemIntegration

        fi = FilesystemIntegration()

        # Create 5 matching files
        for i in range(5):
            p = tmp_path / f"file{i}.txt"
            p.write_text(f"pattern_match line {i}")

            fi._index[str(p)] = FileEntry(
                path=str(p),
                name=p.name,
                extension=".txt",
                size=p.stat().st_size,
                modified=datetime.now().isoformat(),
                is_dir=False,
            )

        # Search with limit=2
        result = asyncio.run(fi._tool_search_content("pattern_match", limit=2))
        assert result.success is True
        assert len(result.output["results"]) == 2
