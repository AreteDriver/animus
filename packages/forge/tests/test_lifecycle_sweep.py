"""Final small-gap sweep for lifecycle/loop defensive branches on 2c8cde2.

Targets:
- websocket/broadcaster.py: 70 (queue get timeout), 76-77 (_handle_update raise)
- websocket/manager.py: 261, 263 (client-loop receive path + normal disconnect)
- workflow/version_manager.py: 534-535 (import-yaml exception swallowed)
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# websocket/broadcaster.py
# ---------------------------------------------------------------------------


from animus_forge.websocket.broadcaster import Broadcaster


@pytest.fixture
def broadcaster():
    """Broadcaster with a stubbed ConnectionManager."""
    mgr = MagicMock()
    return Broadcaster(manager=mgr)


class TestBroadcasterProcessQueue:
    """The _process_queue loop has three uncovered branches:
    - timeout on asyncio.Queue.get → `continue` (line 70)
    - exception from _handle_update → logged, loop continues (lines 76-77)
    """

    @pytest.mark.asyncio
    async def test_timeout_is_swallowed_then_exception_logged(self, broadcaster):
        # Running flag flips to False after the exception path fires,
        # so the loop exits without blocking the test forever.
        broadcaster._running = True

        async def handler(update):
            broadcaster._running = False
            raise RuntimeError("handle boom")

        broadcaster._handle_update = handler

        # No items yet → first iteration hits `asyncio.wait_for` timeout.
        # Then we enqueue one item which drives the exception branch.
        async def feed():
            await asyncio.sleep(0.05)  # give the loop one timeout cycle
            await broadcaster._queue.put({"type": "x", "execution_id": "e1"})

        await asyncio.wait_for(
            asyncio.gather(broadcaster._process_queue(), feed()),
            timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_breaks_loop(self, broadcaster):
        """Line 71-72: CancelledError from queue.get → break."""
        broadcaster._running = True

        # Patch _queue.get to raise CancelledError immediately.
        async def raising_get():
            raise asyncio.CancelledError()

        broadcaster._queue = MagicMock()
        broadcaster._queue.get = raising_get

        # Loop should exit cleanly (not propagate the cancel since it's
        # caught inside wait_for's internals). Use wait_for as a safety net.
        await asyncio.wait_for(broadcaster._process_queue(), timeout=2.0)


# ---------------------------------------------------------------------------
# websocket/manager.py — handle_connection lifecycle
# ---------------------------------------------------------------------------


from fastapi import WebSocketDisconnect

from animus_forge.websocket.manager import ConnectionManager


class TestConnectionManagerHandleConnection:
    @pytest.mark.asyncio
    async def test_disconnect_is_clean(self):
        """Lines 258-263: client-loop receives once, then gets a
        WebSocketDisconnect, and handle_connection returns cleanly."""
        mgr = ConnectionManager()

        # Stub connect/disconnect/handle_client_message on the manager.
        fake_conn = MagicMock()
        fake_conn.id = "conn-1"
        fake_conn.send = AsyncMock()

        async def fake_connect(ws):
            return fake_conn

        async def fake_disconnect(cid):
            return None

        mgr.connect = fake_connect
        mgr.disconnect = fake_disconnect
        mgr.handle_client_message = AsyncMock()

        # Fake WebSocket: first receive succeeds, second raises disconnect.
        fake_ws = MagicMock()
        receive_calls = {"n": 0}

        async def receive_text():
            receive_calls["n"] += 1
            if receive_calls["n"] == 1:
                return '{"type":"ping"}'
            raise WebSocketDisconnect()

        fake_ws.receive_text = receive_text

        # Must not raise; both receive + handle_client_message called once,
        # then disconnect is invoked once.
        await mgr.handle_connection(fake_ws)

        assert receive_calls["n"] == 2
        mgr.handle_client_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unexpected_error_is_logged_and_disconnect_runs(self):
        """Lines 264-267: arbitrary Exception is logged, finally clause
        still calls disconnect — no leaked connections."""
        mgr = ConnectionManager()
        fake_conn = MagicMock()
        fake_conn.id = "conn-2"

        async def fake_connect(ws):
            return fake_conn

        disconnect_calls = {"n": 0}

        async def fake_disconnect(cid):
            disconnect_calls["n"] += 1

        mgr.connect = fake_connect
        mgr.disconnect = fake_disconnect
        mgr.handle_client_message = AsyncMock()

        fake_ws = MagicMock()

        async def raising_receive():
            raise RuntimeError("socket blew up")

        fake_ws.receive_text = raising_receive

        await mgr.handle_connection(fake_ws)
        # Finally clause ran even though the inner loop raised.
        assert disconnect_calls["n"] == 1


# ---------------------------------------------------------------------------
# workflow/version_manager.py — migrate-existing-workflows exception path
# ---------------------------------------------------------------------------


from animus_forge.state.backends import SQLiteBackend
from animus_forge.workflow.version_manager import WorkflowVersionManager


@pytest.fixture
def version_mgr():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "vm.db")
        backend = SQLiteBackend(db_path=db_path)
        # Run migrations to create the version tables.
        from animus_forge.state.migrations import run_migrations

        run_migrations(backend)
        yield WorkflowVersionManager(backend=backend)
        backend.close()


class TestMigrateExistingWorkflowsErrorPath:
    """Lines 534-535: `except Exception as e: logger.error(...)` when
    import_from_file raises on a corrupt YAML file."""

    def test_broken_file_is_logged_but_not_raised(self, version_mgr, tmp_path):
        """When import_from_file raises, migrate_existing_workflows must log
        and continue rather than propagate (lines 534-535)."""
        # Two valid YAML stubs — we force import_from_file to raise on both
        # so we hit the except branch twice.
        (tmp_path / "alpha.yaml").write_text(
            "name: wf-alpha\nversion: '1.0.0'\nsteps:\n  - id: s1\n    type: shell\n"
        )
        (tmp_path / "beta.yaml").write_text(
            "name: wf-beta\nversion: '1.0.0'\nsteps:\n  - id: s1\n    type: shell\n"
        )

        # Stub import_from_file to always raise.
        def always_raise(path, *args, **kwargs):
            raise RuntimeError(f"storage exploded for {path}")

        version_mgr.import_from_file = always_raise

        # Must not raise despite BOTH files failing — exception path
        # caught both times in the loop.
        imported = version_mgr.migrate_existing_workflows(str(tmp_path))
        assert imported == []  # nothing landed, but no exception escaped
