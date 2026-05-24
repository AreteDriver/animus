"""Tests for the /memory hit-rate dashboard panel."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from animus_bootstrap.dashboard.app import app
from animus_bootstrap.intelligence.memory_scores import MemoryScoreStore
from animus_bootstrap.intelligence.recall_log import RecallLog


def _attach_runtime(stores_dir, content_lookup) -> SimpleNamespace:
    score_store = MemoryScoreStore(stores_dir / "scores.db")
    recall_log = RecallLog(stores_dir / "recall.db")

    memory_manager = SimpleNamespace(get_by_id=AsyncMock(side_effect=content_lookup))
    runtime = SimpleNamespace(
        _memory_score_store=score_store,
        _recall_log=recall_log,
        memory_manager=memory_manager,
    )
    return runtime, score_store, recall_log


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


class TestMemoryPanel:
    def test_empty_state_renders(self, client: TestClient, tmp_path) -> None:
        async def lookup(_):
            return None

        runtime, ss, rl = _attach_runtime(tmp_path, lookup)
        try:
            client.app.state.runtime = runtime
            resp = client.get("/memory")
            assert resp.status_code == 200
            body = resp.text
            assert "Recalls Logged" in body
            assert "Memories Scored" in body
            assert "No scored memories yet" in body
        finally:
            client.app.state.runtime = None
            ss.close()
            rl.close()

    def test_panel_shows_scored_memories(self, client: TestClient, tmp_path) -> None:
        catalog = {
            "good-id": {"id": "good-id", "type": "episodic", "content": "BG ask fix landed"},
            "bad-id": {"id": "bad-id", "type": "semantic", "content": "stale projects entry"},
        }

        async def lookup(memory_id):
            return catalog.get(memory_id)

        runtime, ss, rl = _attach_runtime(tmp_path, lookup)
        try:
            for _ in range(3):
                ss.record_outcome("good-id", 1.0)
            for _ in range(3):
                ss.record_outcome("bad-id", -1.0)
            rl.log_recall(query="q", memory_entries=[("good-id", "episodic")])

            client.app.state.runtime = runtime
            resp = client.get("/memory")
            assert resp.status_code == 200
            body = resp.text
            assert "BG ask fix landed" in body
            assert "stale projects entry" in body
            # Scores rendered
            assert "+1.00" in body
            assert "-1.00" in body
        finally:
            client.app.state.runtime = None
            ss.close()
            rl.close()

    def test_scores_json_endpoint(self, client: TestClient, tmp_path) -> None:
        async def lookup(memory_id):
            return {"id": memory_id, "type": "episodic", "content": f"content for {memory_id}"}

        runtime, ss, rl = _attach_runtime(tmp_path, lookup)
        try:
            ss.record_outcome("a", 1.0)
            ss.record_outcome("b", -1.0)

            client.app.state.runtime = runtime
            resp = client.get("/memory/scores")
            assert resp.status_code == 200
            data = resp.json()
            assert data["stats"]["scored_memories"] == 2
            assert any(m["memory_id"] == "a" for m in data["top"])
            assert any(m["memory_id"] == "b" for m in data["bottom"])
        finally:
            client.app.state.runtime = None
            ss.close()
            rl.close()

    def test_scores_endpoint_503_without_runtime(self, client: TestClient) -> None:
        client.app.state.runtime = None
        resp = client.get("/memory/scores")
        assert resp.status_code == 503
