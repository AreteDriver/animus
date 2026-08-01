"""Comprehensive validation tests for all Animus contract schemas.

Ensures every schema is valid Draft 2020-12 and validates representative
positive and negative fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from animus_contracts import SCHEMAS_DIR, ValidationError, validate
from jsonschema import Draft202012Validator

ALL_SCHEMA_NAMES = [
    "action",
    "action_receipt",
    "approval_receipt",
    "assessment",
    "capability_grant",
    "claim",
    "common",
    "context_envelope",
    "decision",
    "entity",
    "event",
    "forecast",
    "hypothesis",
    "ledger_event",
    "lesson",
    "memory_candidate",
    "object_version",
    "observation",
    "outbox_entry",
    "outcome",
    "pattern",
    "policy_decision",
    "signal",
    "source",
    "trace",
]


def _schema_path(name: str) -> Path:
    return SCHEMAS_DIR / f"{name}.schema.json"


def _load_schema(name: str) -> dict:
    return json.loads(_schema_path(name).read_text())


# ── Compilation tests ──────────────────────────────────────────────────────


class TestSchemaCompilation:
    """Every schema must be valid Draft 2020-12."""

    @pytest.mark.parametrize("name", ALL_SCHEMA_NAMES)
    def test_compiles(self, name: str):
        schema = _load_schema(name)
        Draft202012Validator.check_schema(schema)


# ── Common envelope fixture ────────────────────────────────────────────────


def _common_envelope(artifact_type: str, schema_id: str, payload: dict) -> dict:
    return {
        "object_id": "obj-test-001",
        "object_version": 1,
        "schema_id": schema_id,
        "schema_version": "1.0.0",
        "owner_id": "owner-test",
        "workspace_id": "ws-test",
        "subject_domain": "self",
        "artifact_type": artifact_type,
        "cognitive_role": "memory",
        "workflow_status": "candidate",
        "epistemic_status": "supported",
        "lifecycle_status": "active",
        "storage_tier": "warm",
        "presentation": "canonical",
        "security_class": "internal",
        "valid_time": {
            "valid_from": "2026-07-01T00:00:00Z",
            "valid_to": None,
        },
        "transaction_time": {
            "recorded_at": "2026-07-01T00:00:00Z",
            "superseded_at": None,
        },
        "provenance": {
            "created_by": "pytest",
            "source_refs": [],
            "derived_from": [],
            "trace_id": None,
        },
        "integrity": {
            "content_sha256": "a" * 64,
            "previous_version_sha256": None,
        },
        "payload": payload,
    }


# ── Action ─────────────────────────────────────────────────────────────────


class TestAction:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="action",
            schema_id="https://animus.local/schemas/action.schema.json",
            payload={
                "action_kind": "review",
                "risk_class": "R2",
                "target": "file.py",
                "parameters": {},
                "approval_required": False,
                "approval_id": None,
                "idempotency_key": "key-12345",
                "status": "proposed",
            },
        )

    def test_valid(self, valid):
        validate(valid, "action")

    def test_missing_required_payload_field(self, valid):
        del valid["payload"]["action_kind"]
        with pytest.raises(ValidationError):
            validate(valid, "action")

    def test_invalid_risk_class(self, valid):
        valid["payload"]["risk_class"] = "R99"
        with pytest.raises(ValidationError):
            validate(valid, "action")

    def test_approval_required_without_id(self, valid):
        valid["payload"]["approval_required"] = True
        valid["payload"]["approval_id"] = None
        with pytest.raises(ValidationError):
            validate(valid, "action")


# ── Action Receipt ───────────────────────────────────────────────────────


class TestActionReceipt:
    @pytest.fixture
    def valid(self):
        return {
            "receipt_id": "ar-001",
            "trace_id": "trace-test",
            "principal_id": "agent-test",
            "tool_id": "tool-review",
            "risk_class": "R2",
            "request_hash": "a" * 64,
            "policy_decision": "allow",
            "approval_id": None,
            "idempotency_key": "key-12345",
            "status": "succeeded",
            "requested_at": "2026-07-01T00:00:00Z",
            "completed_at": "2026-07-01T00:01:00Z",
            "side_effect_refs": [],
        }

    def test_valid(self, valid):
        validate(valid, "action_receipt")

    def test_missing_required(self, valid):
        del valid["status"]
        with pytest.raises(ValidationError):
            validate(valid, "action_receipt")

    def test_invalid_status(self, valid):
        valid["status"] = "done"
        with pytest.raises(ValidationError):
            validate(valid, "action_receipt")


# ── Approval Receipt ───────────────────────────────────────────────────────


class TestApprovalReceipt:
    @pytest.fixture
    def valid(self):
        return {
            "approval_id": "apr-001",
            "principal_id": "agent-test",
            "owner_id": "owner-test",
            "workspace_id": "ws-test",
            "purpose": "Review code",
            "resource_refs": ["file.py"],
            "action_scope": {},
            "issued_at": "2026-07-01T00:00:00Z",
            "expires_at": "2026-07-02T00:00:00Z",
            "policy_version": "v1.0.0",
            "status": "active",
            "signature": "sig-" + "a" * 20,
        }

    def test_valid(self, valid):
        validate(valid, "approval_receipt")

    def test_missing_required(self, valid):
        del valid["status"]
        with pytest.raises(ValidationError):
            validate(valid, "approval_receipt")

    def test_invalid_status(self, valid):
        valid["status"] = "maybe"
        with pytest.raises(ValidationError):
            validate(valid, "approval_receipt")


# ── Assessment ─────────────────────────────────────────────────────────────


class TestAssessment:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="assessment",
            schema_id="https://animus.local/schemas/assessment.schema.json",
            payload={
                "question": "Is this code secure?",
                "judgment": "Yes, with caveats",
                "evidence_refs": ["obs-001"],
                "alternative_judgments": ["No, needs review"],
                "confidence_dimensions": {
                    "evidence": 0.8,
                    "reasoning": 0.7,
                    "prediction": 0.6,
                    "action": 0.9,
                },
                "dissent": [
                    {
                        "position": "Could be unsafe",
                        "severity": "medium",
                        "agent_ref": None,
                    }
                ],
            },
        )

    def test_valid(self, valid):
        validate(valid, "assessment")

    def test_missing_required_payload(self, valid):
        del valid["payload"]["question"]
        with pytest.raises(ValidationError):
            validate(valid, "assessment")

    def test_invalid_confidence_out_of_range(self, valid):
        valid["payload"]["confidence_dimensions"]["evidence"] = 1.5
        with pytest.raises(ValidationError):
            validate(valid, "assessment")


# ── Capability Grant ───────────────────────────────────────────────────────


class TestCapabilityGrant:
    @pytest.fixture
    def valid(self):
        return {
            "grant_id": "grant-researcher",
            "principal": "agent-researcher",
            "scope": ["read", "write"],
            "resource": "ws-demo/*",
            "action": ["create", "read", "update"],
            "granted_by": "owner-arete",
            "granted_at": "2026-07-01T00:00:00Z",
            "expires_at": None,
            "budget": {"max_calls": 1000, "window_seconds": 3600},
            "conditions": {"allowed_workspaces": ["ws-demo"]},
            "revoked_at": None,
            "revoked_by": None,
            "revocation_reason": None,
        }

    def test_valid(self, valid):
        validate(valid, "capability_grant")

    def test_invalid_action(self, valid):
        valid["action"] = ["hack"]
        with pytest.raises(ValidationError):
            validate(valid, "capability_grant")


# ── Claim ────────────────────────────────────────────────────────────────────


class TestClaim:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="claim",
            schema_id="https://animus.local/schemas/claim.schema.json",
            payload={
                "proposition": "The system is production-ready",
                "claim_kind": "factual",
                "supporting_evidence": ["obs-001"],
                "contradicting_evidence": [],
                "confidence": 0.85,
            },
        )

    def test_valid(self, valid):
        validate(valid, "claim")

    def test_missing_required(self, valid):
        del valid["payload"]["proposition"]
        with pytest.raises(ValidationError):
            validate(valid, "claim")

    def test_invalid_claim_kind(self, valid):
        valid["payload"]["claim_kind"] = "guess"
        with pytest.raises(ValidationError):
            validate(valid, "claim")


# ── Context Envelope ───────────────────────────────────────────────────────


class TestContextEnvelope:
    @pytest.fixture
    def valid(self):
        return {
            "envelope_id": "ce-001",
            "trace_id": "trace-test",
            "owner_id": "owner-test",
            "workspace_id": "ws-test",
            "purpose": "Answer user question",
            "query": "What is the status?",
            "as_of": "2026-07-01T00:00:00Z",
            "security_ceiling": "internal",
            "token_budget": 4000,
            "items": [
                {
                    "object_id": "obj-001",
                    "object_version": 1,
                    "score": 0.9,
                    "role": "evidence",
                    "source_refs": ["src-001"],
                }
            ],
            "contradictions": [],
            "omissions": [],
            "freshness": {
                "oldest_recorded_at": "2026-07-01T00:00:00Z",
                "newest_recorded_at": "2026-07-01T00:00:00Z",
            },
            "created_at": "2026-07-01T00:00:00Z",
        }

    def test_valid(self, valid):
        validate(valid, "context_envelope")

    def test_missing_required(self, valid):
        del valid["query"]
        with pytest.raises(ValidationError):
            validate(valid, "context_envelope")

    def test_invalid_security_ceiling(self, valid):
        valid["security_ceiling"] = "secret"
        with pytest.raises(ValidationError):
            validate(valid, "context_envelope")


# ── Decision ─────────────────────────────────────────────────────────────────


class TestDecision:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="decision",
            schema_id="https://animus.local/schemas/decision.schema.json",
            payload={
                "question": "Should we deploy?",
                "chosen_option": "Yes",
                "alternatives": ["No", "Wait"],
                "rationale": "Tests pass",
                "authority": "owner",
                "decision_at": "2026-07-01T00:00:00Z",
                "review_trigger": None,
            },
        )

    def test_valid(self, valid):
        validate(valid, "decision")

    def test_missing_required(self, valid):
        del valid["payload"]["authority"]
        with pytest.raises(ValidationError):
            validate(valid, "decision")

    def test_invalid_authority(self, valid):
        valid["payload"]["authority"] = "boss"
        with pytest.raises(ValidationError):
            validate(valid, "decision")


# ── Entity ─────────────────────────────────────────────────────────────────


class TestEntity:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="entity",
            schema_id="https://animus.local/schemas/entity.schema.json",
            payload={
                "entity_kind": "system",
                "canonical_name": "Animus",
                "aliases": ["AI Exocortex"],
            },
        )

    def test_valid(self, valid):
        validate(valid, "entity")

    def test_missing_required(self, valid):
        del valid["payload"]["canonical_name"]
        with pytest.raises(ValidationError):
            validate(valid, "entity")

    def test_invalid_entity_kind(self, valid):
        valid["payload"]["entity_kind"] = "robot"
        with pytest.raises(ValidationError):
            validate(valid, "entity")


# ── Event ────────────────────────────────────────────────────────────────────


class TestEvent:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="event",
            schema_id="https://animus.local/schemas/event.schema.json",
            payload={
                "event_kind": "deployment",
                "occurred_at": "2026-07-01T00:00:00Z",
                "actor_refs": ["agent-test"],
                "object_refs": ["obj-001"],
                "event_data": {"version": "1.0.0"},
            },
        )

    def test_valid(self, valid):
        validate(valid, "event")

    def test_missing_required(self, valid):
        del valid["payload"]["event_kind"]
        with pytest.raises(ValidationError):
            validate(valid, "event")


# ── Forecast ─────────────────────────────────────────────────────────────────


class TestForecast:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="forecast",
            schema_id="https://animus.local/schemas/forecast.schema.json",
            payload={
                "statement": "Load will spike at 14:00",
                "probability": 0.75,
                "horizon": "2026-07-02T00:00:00Z",
                "resolution_rule": "Check metrics at 14:00",
                "assumptions": ["Normal traffic"],
                "alternative_outcomes": [{"outcome": "No spike", "probability": 0.25}],
            },
        )

    def test_valid(self, valid):
        validate(valid, "forecast")

    def test_missing_required(self, valid):
        del valid["payload"]["statement"]
        with pytest.raises(ValidationError):
            validate(valid, "forecast")

    def test_probability_out_of_range(self, valid):
        valid["payload"]["probability"] = 1.5
        with pytest.raises(ValidationError):
            validate(valid, "forecast")


# ── Hypothesis ───────────────────────────────────────────────────────────────


class TestHypothesis:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="hypothesis",
            schema_id="https://animus.local/schemas/hypothesis.schema.json",
            payload={
                "statement": "Adding caching improves latency",
                "pattern_refs": ["pat-001"],
                "predictions": ["Latency drops 20%"],
                "falsification_conditions": ["Latency increases"],
                "alternatives": ["No change"],
            },
        )

    def test_valid(self, valid):
        validate(valid, "hypothesis")

    def test_missing_required(self, valid):
        del valid["payload"]["statement"]
        with pytest.raises(ValidationError):
            validate(valid, "hypothesis")


# ── Ledger Event ─────────────────────────────────────────────────────────────


class TestLedgerEvent:
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
            "tx_time": "2026-07-01T00:00:00Z",
            "parent_event_id": None,
        }

    def test_valid(self, valid):
        validate(valid, "ledger_event")

    def test_missing_required(self, valid):
        del valid["integrity_hash"]
        with pytest.raises(ValidationError):
            validate(valid, "ledger_event")


# ── Lesson ───────────────────────────────────────────────────────────────────


class TestLesson:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="lesson",
            schema_id="https://animus.local/schemas/lesson.schema.json",
            payload={
                "statement": "Always validate inputs",
                "evidence_refs": ["evt-001"],
                "scope": "general",
                "confidence": 0.9,
                "application_conditions": ["Add schema validation"],
            },
        )

    def test_valid(self, valid):
        validate(valid, "lesson")

    def test_missing_required(self, valid):
        del valid["payload"]["statement"]
        with pytest.raises(ValidationError):
            validate(valid, "lesson")


# ── Memory Candidate ─────────────────────────────────────────────────────────


class TestMemoryCandidate:
    @pytest.fixture
    def valid(self):
        return {
            "candidate_id": "mc-test-001",
            "owner_id": "owner-test",
            "workspace_id": "ws-test",
            "proposed_object": {"content": "Important fact"},
            "candidate_reason": "Extracted from conversation",
            "duplicate_refs": [],
            "contradiction_refs": [],
            "validation": {
                "source_valid": True,
                "scope_valid": True,
                "time_valid": True,
                "sensitivity_valid": True,
                "confidence_valid": True,
            },
            "approval_required": False,
            "status": "candidate",
            "created_at": "2026-07-01T00:00:00Z",
        }

    def test_valid(self, valid):
        validate(valid, "memory_candidate")

    def test_missing_required(self, valid):
        del valid["candidate_reason"]
        with pytest.raises(ValidationError):
            validate(valid, "memory_candidate")

    def test_invalid_candidate_id(self, valid):
        valid["candidate_id"] = "bad"
        with pytest.raises(ValidationError):
            validate(valid, "memory_candidate")


# ── Object Version ───────────────────────────────────────────────────────────


class TestObjectVersion:
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
            "workflow_status": "candidate",
            "epistemic_status": "supported",
            "lifecycle_status": "active",
            "storage_tier": "warm",
            "presentation": "canonical",
            "security_class": "internal",
            "valid_from": "2026-07-01T00:00:00Z",
            "recorded_at": "2026-07-01T00:00:00Z",
            "created_by": "pytest",
            "content_sha256": "b" * 64,
            "payload": {"content": "hello"},
        }

    def test_valid(self, valid):
        validate(valid, "object_version")

    def test_invalid_schema_version(self, valid):
        valid["schema_version"] = "1.0"
        with pytest.raises(ValidationError):
            validate(valid, "object_version")


# ── Observation ────────────────────────────────────────────────────────────────


class TestObservation:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="observation",
            schema_id="https://animus.local/schemas/observation.schema.json",
            payload={
                "description": "CPU usage spiked to 95%",
                "observed_at": "2026-07-01T00:00:00Z",
                "observation_method": "measurement",
                "source_anchor": {
                    "source_id": "monitor-001",
                    "locator": "metrics/cpu",
                },
                "measurement": None,
            },
        )

    def test_valid(self, valid):
        validate(valid, "observation")

    def test_missing_required(self, valid):
        del valid["payload"]["observation_method"]
        with pytest.raises(ValidationError):
            validate(valid, "observation")

    def test_invalid_observation_method(self, valid):
        valid["payload"]["observation_method"] = "guessing"
        with pytest.raises(ValidationError):
            validate(valid, "observation")


# ── Outbox Entry ─────────────────────────────────────────────────────────────


class TestOutboxEntry:
    @pytest.fixture
    def valid(self):
        return {
            "entry_id": "obx-001",
            "topic": "object.created",
            "payload": {"object_id": "mem-001"},
            "headers": {"x-request-id": "req-123"},
            "created_at": "2026-07-01T00:00:00Z",
            "claimed_at": None,
            "claimed_by": None,
            "retry_count": 0,
            "processed_at": None,
            "error_message": None,
        }

    def test_valid(self, valid):
        validate(valid, "outbox_entry")

    def test_negative_retry_count(self, valid):
        valid["retry_count"] = -1
        with pytest.raises(ValidationError):
            validate(valid, "outbox_entry")


# ── Outcome ──────────────────────────────────────────────────────────────────


class TestOutcome:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="outcome",
            schema_id="https://animus.local/schemas/outcome.schema.json",
            payload={
                "action_refs": ["act-001"],
                "description": "Deployment succeeded",
                "observed_at": "2026-07-01T00:00:00Z",
                "metrics": {"latency_ms": 45},
                "expected_comparison": "as_expected",
            },
        )

    def test_valid(self, valid):
        validate(valid, "outcome")

    def test_missing_required(self, valid):
        del valid["payload"]["description"]
        with pytest.raises(ValidationError):
            validate(valid, "outcome")

    def test_invalid_comparison(self, valid):
        valid["payload"]["expected_comparison"] = "unknown"
        with pytest.raises(ValidationError):
            validate(valid, "outcome")


# ── Pattern ──────────────────────────────────────────────────────────────────


class TestPattern:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="pattern",
            schema_id="https://animus.local/schemas/pattern.schema.json",
            payload={
                "description": "Circuit Breaker pattern",
                "signal_refs": ["sig-001", "sig-002"],
                "support_count": 5,
                "independence_score": 0.85,
                "method": "correlation_analysis",
            },
        )

    def test_valid(self, valid):
        validate(valid, "pattern")

    def test_missing_required(self, valid):
        del valid["payload"]["description"]
        with pytest.raises(ValidationError):
            validate(valid, "pattern")

    def test_too_few_signals(self, valid):
        valid["payload"]["signal_refs"] = ["sig-001"]
        with pytest.raises(ValidationError):
            validate(valid, "pattern")


# ── Policy Decision ────────────────────────────────────────────────────────


class TestPolicyDecision:
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
            "tx_time": "2026-07-01T00:00:00Z",
            "parent_decision_id": None,
        }

    def test_valid(self, valid):
        validate(valid, "policy_decision")

    def test_invalid_decision(self, valid):
        valid["decision"] = "maybe"
        with pytest.raises(ValidationError):
            validate(valid, "policy_decision")


# ── Signal ───────────────────────────────────────────────────────────────────


class TestSignal:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="signal",
            schema_id="https://animus.local/schemas/signal.schema.json",
            payload={
                "description": "Anomaly detected in log stream",
                "observation_refs": ["obs-001"],
                "relevance": 0.9,
                "novelty": 0.7,
                "expiry_at": None,
            },
        )

    def test_valid(self, valid):
        validate(valid, "signal")

    def test_missing_required(self, valid):
        del valid["payload"]["relevance"]
        with pytest.raises(ValidationError):
            validate(valid, "signal")

    def test_relevance_out_of_range(self, valid):
        valid["payload"]["relevance"] = 1.5
        with pytest.raises(ValidationError):
            validate(valid, "signal")


# ── Source ───────────────────────────────────────────────────────────────────


class TestSource:
    @pytest.fixture
    def valid(self):
        return _common_envelope(
            artifact_type="source",
            schema_id="https://animus.local/schemas/source.schema.json",
            payload={
                "title": "Architecture Decision Record",
                "source_kind": "document",
                "capture_uri": "https://example.com/adr-001",
                "captured_at": "2026-07-01T00:00:00Z",
                "content_sha256": "a" * 64,
                "authority_status": "canonical",
            },
        )

    def test_valid(self, valid):
        validate(valid, "source")

    def test_missing_required(self, valid):
        del valid["payload"]["source_kind"]
        with pytest.raises(ValidationError):
            validate(valid, "source")

    def test_invalid_authority_status(self, valid):
        valid["payload"]["authority_status"] = "maybe"
        with pytest.raises(ValidationError):
            validate(valid, "source")


# ── Trace ────────────────────────────────────────────────────────────────────


class TestTrace:
    @pytest.fixture
    def valid(self):
        return {
            "trace_id": "tr-test-001",
            "owner_id": "owner-test",
            "workspace_id": "ws-test",
            "purpose": "Execute workflow",
            "started_at": "2026-07-01T00:00:00Z",
            "completed_at": "2026-07-01T00:01:00Z",
            "status": "succeeded",
            "spans": [
                {
                    "span_id": "span-001",
                    "kind": "model",
                    "started_at": "2026-07-01T00:00:00Z",
                    "status": "succeeded",
                }
            ],
            "input_hashes": ["a" * 64],
            "output_hashes": ["b" * 64],
            "policy_versions": ["v1.0.0"],
            "schema_versions": ["v1.0.0"],
            "reproducibility": {
                "captured_inputs": True,
                "captured_prompts": True,
                "captured_model_ids": True,
                "captured_tool_versions": True,
                "replay_status": "passed",
            },
        }

    def test_valid(self, valid):
        validate(valid, "trace")

    def test_missing_required(self, valid):
        del valid["purpose"]
        with pytest.raises(ValidationError):
            validate(valid, "trace")

    def test_invalid_status(self, valid):
        valid["status"] = "unknown"
        with pytest.raises(ValidationError):
            validate(valid, "trace")
