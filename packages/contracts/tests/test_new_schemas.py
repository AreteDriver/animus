"""Tests for Kernel-native contract schemas.

Ensures all 5 new schemas compile under Draft 2020-12 and validate
representative positive/negative fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from animus_contracts import ValidationError, validate

SCHEMAS_DIR = Path(__file__).resolve().parent.parent


def _schema_path(name: str) -> Path:
    return SCHEMAS_DIR / f"{name}.schema.json"


def _load_schema(name: str) -> dict:
    return json.loads(_schema_path(name).read_text())


class TestSchemaCompilation:
    """Every new schema must be valid Draft 2020-12."""

    @pytest.mark.parametrize(
        "name",
        [
            "ledger_event",
            "object_version",
            "outbox_entry",
            "capability_grant",
            "policy_decision",
        ],
    )
    def test_compiles(self, name: str):
        schema = _load_schema(name)
        Draft202012Validator.check_schema(schema)


class TestLedgerEvent:
    """ledger_event.schema.json — immutable append-only event."""

    @pytest.fixture
    def valid(self):
        return {
            "event_id": "evt-abc123",
            "event_type": "created",
            "object_id": "mem-001",
            "object_version": 1,
            "principal": "agent-test",
            "workspace_id": "ws-test",
            "payload": {"artifact_type": "memory", "schema_id": "memory_candidate"},
            "integrity_hash": "a" * 64,
            "tx_time": "2026-07-06T00:00:00Z",
            "parent_event_id": None,
        }

    def test_valid(self, valid):
        validate(valid, "ledger_event")

    def test_missing_required(self, valid):
        invalid = {k: v for k, v in valid.items() if k != "integrity_hash"}
        with pytest.raises(ValidationError):
            validate(invalid, "ledger_event")

    def test_invalid_event_type(self, valid):
        valid["event_type"] = "hacked"
        with pytest.raises(ValidationError):
            validate(valid, "ledger_event")

    def test_bad_integrity_hash_length(self, valid):
        valid["integrity_hash"] = "short"
        with pytest.raises(ValidationError):
            validate(valid, "ledger_event")


class TestObjectVersion:
    """object_version.schema.json — bitemporal canonical object."""

    @pytest.fixture
    def valid(self):
        return {
            "object_id": "mem-001",
            "object_version": 1,
            "schema_id": "memory_candidate",
            "schema_version": "1.0.0",
            "owner_id": "owner-test",
            "workspace_id": "ws-test",
            "subject_domain": "self",
            "artifact_type": "memory",
            "cognitive_role": "memory",
            "workflow_status": "active",
            "epistemic_status": "supported",
            "lifecycle_status": "active",
            "storage_tier": "warm",
            "presentation": "canonical",
            "security_class": "internal",
            "valid_from": "2026-07-06T00:00:00Z",
            "recorded_at": "2026-07-06T00:00:00Z",
            "created_by": "pytest",
            "content_sha256": "b" * 64,
            "payload": {"content": "hello"},
            "tags": ["test"],
        }

    def test_valid(self, valid):
        validate(valid, "object_version")

    def test_valid_agent_contract(self, valid):
        valid["artifact_type"] = "agent_contract"
        validate(valid, "object_version")

    def test_missing_required(self, valid):
        invalid = {k: v for k, v in valid.items() if k != "content_sha256"}
        with pytest.raises(ValidationError):
            validate(invalid, "object_version")

    def test_invalid_schema_version(self, valid):
        valid["schema_version"] = "1.0"
        with pytest.raises(ValidationError):
            validate(valid, "object_version")

    def test_invalid_workspace_id(self, valid):
        valid["workspace_id"] = "bad"
        with pytest.raises(ValidationError):
            validate(valid, "object_version")


class TestOutboxEntry:
    """outbox_entry.schema.json — transactional outbox."""

    @pytest.fixture
    def valid(self):
        return {
            "entry_id": "obx-001",
            "topic": "object.created",
            "payload": {"object_id": "mem-001"},
            "headers": {"x-request-id": "req-123"},
            "created_at": "2026-07-06T00:00:00Z",
            "claimed_at": None,
            "claimed_by": None,
            "retry_count": 0,
            "processed_at": None,
            "error_message": None,
        }

    def test_valid(self, valid):
        validate(valid, "outbox_entry")

    def test_missing_required(self, valid):
        invalid = {k: v for k, v in valid.items() if k != "topic"}
        with pytest.raises(ValidationError):
            validate(invalid, "outbox_entry")

    def test_negative_retry_count(self, valid):
        valid["retry_count"] = -1
        with pytest.raises(ValidationError):
            validate(valid, "outbox_entry")


class TestCapabilityGrant:
    """capability_grant.schema.json — scoped authorization."""

    @pytest.fixture
    def valid(self):
        return {
            "grant_id": "grant-researcher",
            "principal": "agent-researcher",
            "scope": ["read", "write"],
            "resource": "ws-demo/*",
            "action": ["create", "read", "update"],
            "granted_by": "owner-arete",
            "granted_at": "2026-07-06T00:00:00Z",
            "expires_at": None,
            "budget": {"max_calls": 1000, "window_seconds": 3600},
            "conditions": {"allowed_workspaces": ["ws-demo"]},
            "revoked_at": None,
            "revoked_by": None,
            "revocation_reason": None,
        }

    def test_valid(self, valid):
        validate(valid, "capability_grant")

    def test_minimal(self):
        minimal = {
            "grant_id": "grant-test",
            "principal": "agent-test",
            "scope": ["read"],
            "resource": "*",
            "action": ["read"],
            "granted_by": "owner-test",
            "granted_at": "2026-07-06T00:00:00Z",
        }
        validate(minimal, "capability_grant")

    def test_invalid_action(self, valid):
        valid["action"] = ["hack"]
        with pytest.raises(ValidationError):
            validate(valid, "capability_grant")

    def test_empty_scope(self, valid):
        valid["scope"] = []
        with pytest.raises(ValidationError):
            validate(valid, "capability_grant")


class TestPolicyDecision:
    """policy_decision.schema.json — deterministic enforcement record."""

    @pytest.fixture
    def valid(self):
        return {
            "decision_id": "dec-001",
            "rule_id": "rule-default-deny",
            "input_context": {
                "action": "delete",
                "resource": "mem-001",
                "principal": "agent-hacker",
                "workspace_id": "ws-test",
                "schema_id": None,
                "payload_summary": None,
            },
            "decision": "deny",
            "confidence": 1.0,
            "reason": "No capability grants found",
            "denial_reason_code": "missing_scope",
            "obligations": None,
            "principal": "agent-hacker",
            "workspace_id": "ws-test",
            "tx_time": "2026-07-06T00:00:00Z",
            "parent_decision_id": None,
        }

    def test_valid(self, valid):
        validate(valid, "policy_decision")

    def test_escalate_with_obligations(self, valid):
        valid["decision"] = "escalate"
        valid["denial_reason_code"] = "escalation_required"
        valid["obligations"] = [
            {
                "obligation_type": "approve",
                "description": "Approve delete on mem-001",
                "target": "owner",
                "deadline": None,
            }
        ]
        validate(valid, "policy_decision")

    def test_invalid_decision(self, valid):
        valid["decision"] = "maybe"
        with pytest.raises(ValidationError):
            validate(valid, "policy_decision")

    def test_invalid_denial_reason(self, valid):
        valid["denial_reason_code"] = "because_i_said_so"
        with pytest.raises(ValidationError):
            validate(valid, "policy_decision")

    def test_missing_input_context_action(self, valid):
        del valid["input_context"]["action"]
        with pytest.raises(ValidationError):
            validate(valid, "policy_decision")
