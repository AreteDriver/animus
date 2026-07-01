"""Tests for the governance audit trail."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from animus_forge.governance.audit import AuditEntry, AuditTrail


class MockDecision:
    """Simple mock decision for testing audit recording."""

    def __init__(self, **kwargs):
        self.effect = kwargs.get("effect", "allow")
        self.action = kwargs.get("action", "file.write")
        self.policy = kwargs.get("policy", "file_access")
        self.rule = kwargs.get("rule", "allow_tmp")
        self.reason = kwargs.get("reason", "test")
        self.context = kwargs.get("context", {})
        self.request_id = kwargs.get("request_id", "")


class TestAuditTrail:
    def test_record_and_query(self):
        trail = AuditTrail()
        trail.record(MockDecision(effect="allow", action="file.write"))
        trail.record(MockDecision(effect="deny", action="file.delete"))
        trail.record(MockDecision(effect="allow", action="file.write"))

        assert len(trail.entries()) == 3
        assert len(trail.entries(action="file.write")) == 2
        assert len(trail.entries(action="file.delete")) == 1

    def test_filter_by_policy(self):
        trail = AuditTrail()
        trail.record(MockDecision(policy="p1"))
        trail.record(MockDecision(policy="p2"))

        assert len(trail.entries(policy="p1")) == 1
        assert len(trail.entries(policy="p2")) == 1

    def test_filter_by_since(self):
        trail = AuditTrail()
        trail.record(MockDecision())
        # All entries should be after epoch
        assert len(trail.entries(since=datetime(2000, 1, 1))) == 1
        # None should be after far future
        assert len(trail.entries(since=datetime(2100, 1, 1))) == 0

    def test_summary(self):
        trail = AuditTrail()
        trail.record(MockDecision(effect="allow"))
        trail.record(MockDecision(effect="deny"))
        trail.record(MockDecision(effect="allow"))

        summary = trail.summary()
        assert summary["total_entries"] == 3
        assert summary["by_decision_type"]["allow"] == 2
        assert summary["by_decision_type"]["deny"] == 1

    def test_export_jsonl(self, tmp_path: Path):
        trail = AuditTrail()
        trail.record(MockDecision(effect="allow", action="a1"))
        trail.record(MockDecision(effect="deny", action="a2"))

        out_file = tmp_path / "audit.jsonl"
        trail.export(str(out_file), format="jsonl")

        lines = out_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert '"decision_type": "allow"' in lines[0]
        assert '"decision_type": "deny"' in lines[1]

    def test_export_json(self, tmp_path: Path):
        trail = AuditTrail()
        trail.record(MockDecision(effect="allow"))

        out_file = tmp_path / "audit.json"
        trail.export(str(out_file), format="json")

        data = out_file.read_text()
        assert '"decision_type": "allow"' in data

    def test_file_persistence(self, tmp_path: Path):
        audit_file = tmp_path / "audit.jsonl"
        trail = AuditTrail(file_path=audit_file)
        trail.record(MockDecision(effect="allow", action="test"))

        assert audit_file.exists()
        content = audit_file.read_text()
        assert "test" in content

    def test_load_from_file(self, tmp_path: Path):
        audit_file = tmp_path / "audit.jsonl"
        # Write entries directly
        audit_file.write_text(
            '{"timestamp": "2026-01-01T00:00:00", "decision_type": "allow", "action": "a1", "policy": "p1", "rule": null, "reason": "r1", "context": {}, "request_id": ""}\n'
        )

        trail = AuditTrail()
        trail.load(str(audit_file))
        assert len(trail.entries()) == 1
        assert trail.entries()[0].action == "a1"

    def test_entry_roundtrip(self):
        entry = AuditEntry(
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
            decision_type="allow",
            action="file.write",
            policy="file_access",
            rule="allow_tmp",
            reason="matched rule",
            context={"path": "/tmp/test.py"},
            request_id="req-42",
        )

        data = entry.to_dict()
        restored = AuditEntry.from_dict(data)
        assert restored.action == entry.action
        assert restored.request_id == entry.request_id
