"""Tests for the MCP-egress audit log (Stage 3.B)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from animus.audit import EgressAuditLog, prune_old_logs
from animus.memory.types import Sensitivity


class TestEgressAuditLog:
    """Append-only JSONL daily-rotated audit log."""

    def test_record_creates_file_for_today(self, tmp_path: Path):
        log = EgressAuditLog(tmp_path)
        log.record("animus_recall", {Sensitivity.PUBLIC}, 3, 0, 200)

        today = datetime.now(timezone.utc).date().isoformat()
        file_path = tmp_path / "audit" / f"mcp-egress-{today}.jsonl"
        assert file_path.exists()

    def test_record_schema(self, tmp_path: Path):
        log = EgressAuditLog(tmp_path)
        log.record("animus_recall", {Sensitivity.PUBLIC}, 2, 1, 150)

        today = datetime.now(timezone.utc).date().isoformat()
        file_path = tmp_path / "audit" / f"mcp-egress-{today}.jsonl"
        line = file_path.read_text().strip()
        entry = json.loads(line)

        assert entry["tool"] == "animus_recall"
        assert entry["tier_scope"] == ["public"]
        assert entry["result_count"] == 2
        assert entry["redaction_count"] == 1
        assert entry["response_bytes"] == 150
        assert "ts" in entry

    def test_record_appends_multiple_entries(self, tmp_path: Path):
        log = EgressAuditLog(tmp_path)
        log.record("animus_recall", {Sensitivity.PUBLIC}, 1, 0, 100)
        log.record("animus_search_tags", {Sensitivity.PUBLIC}, 0, 0, 50)
        log.record("animus_brief", {Sensitivity.PUBLIC}, 5, 2, 800)

        today = datetime.now(timezone.utc).date().isoformat()
        file_path = tmp_path / "audit" / f"mcp-egress-{today}.jsonl"
        lines = file_path.read_text().splitlines()
        assert len(lines) == 3

        tools = [json.loads(line)["tool"] for line in lines]
        assert tools == ["animus_recall", "animus_search_tags", "animus_brief"]

    def test_record_does_not_log_raw_content(self, tmp_path: Path):
        """The audit log must not become a secondary exfil vector."""
        log = EgressAuditLog(tmp_path)
        log.record("animus_recall", {Sensitivity.PUBLIC}, 1, 1, 200)

        today = datetime.now(timezone.utc).date().isoformat()
        file_path = tmp_path / "audit" / f"mcp-egress-{today}.jsonl"
        contents = file_path.read_text()
        # No "content", "text", "body", or similar key
        assert "content" not in json.loads(contents.strip())

    def test_record_tier_scope_sorted(self, tmp_path: Path):
        log = EgressAuditLog(tmp_path)
        log.record(
            "animus_recall",
            {Sensitivity.SECRET, Sensitivity.PUBLIC, Sensitivity.PERSONAL},
            1,
            0,
            100,
        )

        today = datetime.now(timezone.utc).date().isoformat()
        file_path = tmp_path / "audit" / f"mcp-egress-{today}.jsonl"
        entry = json.loads(file_path.read_text().strip())
        assert entry["tier_scope"] == ["personal", "public", "secret"]

    def test_record_tier_scope_none(self, tmp_path: Path):
        """None tier_scope (caller passed no filter) is preserved as null."""
        log = EgressAuditLog(tmp_path)
        log.record("animus_recall", None, 1, 0, 100)

        today = datetime.now(timezone.utc).date().isoformat()
        entry = json.loads((tmp_path / "audit" / f"mcp-egress-{today}.jsonl").read_text().strip())
        assert entry["tier_scope"] is None

    def test_record_never_raises_on_filesystem_error(self, tmp_path, monkeypatch):
        """Audit failures must not break tool execution."""
        log = EgressAuditLog(tmp_path)

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        # Sabotage the file open
        monkeypatch.setattr(Path, "open", lambda *_a, **_kw: boom())
        # Must not raise
        log.record("animus_recall", {Sensitivity.PUBLIC}, 1, 0, 100)


class TestPruneOldLogs:
    """30-day retention via filename-date matching."""

    def _write_log(self, audit_dir: Path, date_str: str, content: str = "{}\n") -> Path:
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / f"mcp-egress-{date_str}.jsonl"
        path.write_text(content)
        return path

    def test_prune_deletes_files_older_than_30_days(self, tmp_path: Path):
        audit_dir = tmp_path / "audit"
        today = datetime.now(timezone.utc).date()
        old = self._write_log(audit_dir, (today - timedelta(days=40)).isoformat())
        recent = self._write_log(audit_dir, today.isoformat())
        deleted = prune_old_logs(tmp_path)

        assert old in deleted
        assert recent not in deleted
        assert not old.exists()
        assert recent.exists()

    def test_prune_respects_custom_retention(self, tmp_path: Path):
        audit_dir = tmp_path / "audit"
        today = datetime.now(timezone.utc).date()
        five_days_old = self._write_log(audit_dir, (today - timedelta(days=5)).isoformat())
        deleted = prune_old_logs(tmp_path, retention_days=3)
        assert five_days_old in deleted

    def test_prune_skips_non_matching_filenames(self, tmp_path: Path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        unrelated = audit_dir / "backfill-review.jsonl"
        unrelated.write_text("{}\n")
        # Older-than-30d egress file in the same dir
        today = datetime.now(timezone.utc).date()
        old = self._write_log(audit_dir, (today - timedelta(days=40)).isoformat())

        prune_old_logs(tmp_path)
        assert unrelated.exists()  # untouched
        assert not old.exists()

    def test_prune_handles_missing_dir(self, tmp_path: Path):
        # No audit dir at all — should not raise
        deleted = prune_old_logs(tmp_path)
        assert deleted == []

    def test_prune_empty_dir(self, tmp_path: Path):
        (tmp_path / "audit").mkdir()
        deleted = prune_old_logs(tmp_path)
        assert deleted == []
