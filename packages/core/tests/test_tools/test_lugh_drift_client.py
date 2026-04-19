"""Tests for DriftMonitorClient — HTTP POST + NDJSON fallback."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus.lugh.drift_client import DriftMonitorClient


@pytest.fixture
def sink(tmp_path) -> Path:
    return tmp_path / "drift.ndjson"


class TestDriftMonitorClient:
    def test_no_url_writes_to_sink(self, sink):
        client = DriftMonitorClient(url=None, sink_file=sink)
        ok = client.publish({"session_id": "s1", "signals": {"x": 1}})
        assert ok is False  # no HTTP path means no "posted" success
        assert sink.exists()
        lines = sink.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["session_id"] == "s1"

    def test_successful_post_returns_true(self, sink):
        client = DriftMonitorClient(url="https://drift.example/ingest", sink_file=sink)
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda self_: self_
        resp.__exit__ = lambda self_, *a: False
        with patch("urllib.request.urlopen", return_value=resp) as mock_open:
            ok = client.publish({"session_id": "s1"})
            assert ok is True
            assert mock_open.called
        # Sink NOT written on successful POST
        assert not sink.exists()

    def test_http_error_falls_back_to_sink(self, sink):
        client = DriftMonitorClient(url="https://drift.example/ingest", sink_file=sink)
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                "https://drift.example/ingest", 500, "boom", {}, io.BytesIO(b"")
            ),
        ):
            ok = client.publish({"session_id": "s1"})
            assert ok is False
        assert sink.exists()
        assert "s1" in sink.read_text()

    def test_url_error_falls_back_to_sink(self, sink):
        client = DriftMonitorClient(url="https://drift.example/ingest", sink_file=sink)
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns")):
            ok = client.publish({"session_id": "s1"})
            assert ok is False
        assert sink.exists()

    def test_token_sets_auth_header(self, sink):
        client = DriftMonitorClient(
            url="https://drift.example/ingest",
            token="sekret",
            sink_file=sink,
        )
        captured = {}

        def fake_open(req, timeout):
            captured["headers"] = dict(req.header_items())
            resp = MagicMock()
            resp.status = 204
            resp.__enter__ = lambda self_: self_
            resp.__exit__ = lambda self_, *a: False
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_open):
            client.publish({"session_id": "s2"})
        # urllib's Request normalizes header names to title case
        assert captured["headers"].get("Authorization") == "Bearer sekret"

    def test_publish_many_counts_posted_and_appended(self, sink):
        from animus.lugh.models import SessionSummary

        def summary(sid: str) -> SessionSummary:
            return SessionSummary(
                session_id=sid,
                session_file=f"proj/{sid}.jsonl",
                is_sidechain=False,
                agent_id=None,
                project_dir="-home-arete-projects",
                primary_cwd="/home/arete/projects",
                date="2026-04-10",
                duration_seconds=60.0,
                active_seconds=60.0,
                turn_count=2,
                tool_call_count=0,
                mcp_call_count=0,
                model_distribution={"claude-opus-4-6": 2},
                total_input_tokens=10,
                total_output_tokens=20,
                total_cache_read_tokens=0,
                total_cache_write_tokens=0,
                total_cost_usd=0.01,
                activity_breakdown={"conversation": 100.0},
                cwd_breakdown={"/home/arete/projects": 2},
                unknown_line_types=[],
                unknown_models=[],
                turns=[],
            )

        client = DriftMonitorClient(url="https://drift.example/ingest", sink_file=sink)
        # First call succeeds, second fails
        responses = [
            _mock_response(204),
            urllib.error.URLError("transient"),
        ]

        def fake_open(*a, **kw):
            resp = responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp

        with patch("urllib.request.urlopen", side_effect=fake_open):
            stats = client.publish_many([summary("a"), summary("b")])
        assert stats == {"posted": 1, "appended": 1}
        assert sink.exists()
        assert '"b"' in sink.read_text()

    def test_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LUGH_DRIFT_MONITOR_URL", "https://env.example/ingest")
        monkeypatch.setenv("LUGH_DRIFT_MONITOR_TOKEN", "envtoken")
        monkeypatch.setenv("LUGH_DRIFT_SINK_FILE", str(tmp_path / "env.ndjson"))
        client = DriftMonitorClient()
        assert client.url == "https://env.example/ingest"
        assert client.token == "envtoken"
        assert client.sink_file == tmp_path / "env.ndjson"

    def test_explicit_args_override_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LUGH_DRIFT_MONITOR_URL", "https://env.example/ingest")
        client = DriftMonitorClient(url="https://explicit.example/ingest")
        assert client.url == "https://explicit.example/ingest"


def _mock_response(status: int):
    r = MagicMock()
    r.status = status
    r.__enter__ = lambda self_: self_
    r.__exit__ = lambda self_, *a: False
    return r
