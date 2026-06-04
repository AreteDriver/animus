"""Tests for the MCP-egress DLP scrubber (Stage 3.A).

The scrubber is defense-in-depth on the egress boundary. It catches:
- Legacy memories written before Stage 1 redaction was deployed
- PII patterns added to the redaction set after memory was stored
- Any leak path that bypasses ingest (e.g., direct ChromaDB writes)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from animus.memory.types import Memory, MemoryType

pytestmark = pytest.mark.skipif(
    pytest.importorskip("mcp.server.fastmcp", reason="mcp SDK not installed") is None,
    reason="mcp SDK not installed",
)


@pytest.fixture
def server(monkeypatch, tmp_path):
    """Build an MCP server with a mocked memory backend so we can seed responses."""
    mock_memory = MagicMock()
    mock_tasks = MagicMock()
    monkeypatch.setattr("animus.mcp_server.MemoryLayer", lambda *_a, **_kw: mock_memory)
    monkeypatch.setattr("animus.mcp_server.TaskTracker", lambda *_a, **_kw: mock_tasks)

    cfg = MagicMock()
    cfg.data_dir = tmp_path
    cfg.memory.backend = "local"
    cfg.ensure_dirs = MagicMock()
    monkeypatch.setattr("animus.mcp_server.AnimusConfig", MagicMock(load=lambda: cfg))

    from animus.mcp_server import create_mcp_server

    return create_mcp_server(), mock_memory


def _make_memory(content: str, tags: list[str] | None = None) -> Memory:
    from datetime import datetime

    return Memory(
        id="abcdef01-0000",
        content=content,
        memory_type=MemoryType.SEMANTIC,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={},
        tags=tags or [],
    )


def _run(coro):
    """Run an async tool call."""
    import asyncio

    return asyncio.run(coro)


class TestScrubEgressHelper:
    """The pure helper — same regex set as ingest redaction."""

    def test_clean_text_passes_through(self):
        from animus.mcp_server import _scrub_egress

        text = "Normal memory about Linux SSDs."
        scrubbed, count = _scrub_egress(text)
        assert scrubbed == text
        assert count == 0

    def test_anthropic_key_scrubbed(self):
        from animus.mcp_server import _scrub_egress

        text = "key=sk-ant-abcdefghijklmnopqrstuvwxyz12345 end"
        scrubbed, count = _scrub_egress(text)
        assert "sk-ant-" not in scrubbed
        assert count >= 1

    def test_empty_input(self):
        from animus.mcp_server import _scrub_egress

        assert _scrub_egress("") == ("", 0)

    def test_personal_email_scrubbed_on_egress(self):
        """Even if ingest somehow missed it (e.g., legacy data), egress catches."""
        from animus.mcp_server import _scrub_egress

        text = "Contact: aretedriver@gmail.com for follow-up"
        scrubbed, count = _scrub_egress(text)
        assert "aretedriver@gmail.com" not in scrubbed
        assert count >= 1


class TestRecallScrubsOnEgress:
    """End-to-end: a memory bypassing ingest redaction still gets scrubbed at egress."""

    def test_recall_scrubs_legacy_secret_in_content(self, server):
        srv, mock_memory = server
        # Simulate a legacy memory that was stored before ingest redaction —
        # raw secret survives in the content. Egress scrubber must catch it.
        leaked = _make_memory(
            content="legacy: token sk-ant-abcdefghijklmnopqrstuvwxyz12345 was here"
        )
        mock_memory.recall.return_value = [leaked]

        result = _run(srv.call_tool("animus_recall", {"query": "token"}))
        text = result[0][0].text
        assert "sk-ant-" not in text
        assert "REDACTED:anthropic_key" in text

    def test_recall_returns_clean_when_no_secrets(self, server):
        srv, mock_memory = server
        clean = _make_memory(content="TRIM vs secure-erase on SSDs")
        mock_memory.recall.return_value = [clean]

        result = _run(srv.call_tool("animus_recall", {"query": "ssd"}))
        text = result[0][0].text
        assert "TRIM" in text
        assert "REDACTED" not in text

    def test_search_tags_scrubs_legacy_secret(self, server):
        srv, mock_memory = server
        leaked = _make_memory(
            content="api token ghp_abcdefghij1234567890ABCDEFGH leaked here",
            tags=["secrets"],
        )
        mock_memory.recall_by_tags.return_value = [leaked]

        result = _run(srv.call_tool("animus_search_tags", {"tags": "secrets"}))
        text = result[0][0].text
        assert "ghp_" not in text
        # PR #61 added ``credential_label_separated`` (catches ``api
        # token ghp_…`` from position 0) which wins the span merge over
        # ``github_token`` for label-prefixed test strings. Removal of
        # the secret is the invariant; the classifier label that wins
        # the merge is bookkeeping.
        assert "REDACTED:" in text
        assert any(
            f"REDACTED:{label}" in text
            for label in (
                "github_token",
                "credential_label_separated",
                "credential_label_camelcase",
            )
        )

    def test_brief_scrubs_legacy_secret(self, server):
        srv, mock_memory = server
        leaked = _make_memory(content="briefing context includes AKIAIOSFODNN7EXAMPLE somehow")
        mock_memory.recall.return_value = [leaked]

        result = _run(srv.call_tool("animus_brief", {}))
        text = result[0][0].text
        assert "AKIA" not in text
        assert "REDACTED:aws_access_key" in text

    def test_recall_scrubs_personal_email_at_egress(self, server):
        """The headline adversarial test: personal email cannot leak even if
        it bypassed ingest redaction (e.g., from a legacy memory)."""
        srv, mock_memory = server
        leaked = _make_memory(content="legacy contact: aretedriver@gmail.com")
        mock_memory.recall.return_value = [leaked]

        result = _run(srv.call_tool("animus_recall", {"query": "contact"}))
        text = result[0][0].text
        assert "aretedriver@gmail.com" not in text
        assert "REDACTED:personal_email" in text
