"""Lightweight end-to-end integration test.

Verifies that the core subsystems can work together in-process without
requiring a running server stack:
- Kernel memory stores and recalls data
- Contracts schemas validate payloads
- Types are shared across package boundaries
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


def _ensure_paths():
    for rel in (
        "packages/core",
        "packages/kernel/src",
        "packages/types/src",
        "packages/contracts/src",
    ):
        abs_path = str(REPO_ROOT / rel)
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)


@pytest.fixture(scope="module", autouse=True)
def _setup_paths():
    _ensure_paths()


class TestMemoryWorkflow:
    def test_store_recall_roundtrip(self, tmp_path: Path):
        """MemoryLayer can store and retrieve a semantic fact."""
        from animus_kernel.memory.layer import MemoryLayer
        from animus_kernel.memory.types import MemoryType

        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        mem = layer.remember_fact("Alice", "likes", "tea", category="preference")

        results = layer.recall("Alice likes tea")
        assert any("Alice likes tea" in r.content for r in results)
        assert mem.memory_type == MemoryType.SEMANTIC

    def test_versioning_chain(self, tmp_path: Path):
        """A memory can be versioned and its history walked."""
        from animus_kernel.memory.layer import MemoryLayer

        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        v1 = layer.remember("original content")
        v2 = layer.update_with_version(v1.id, content="updated content")
        v3 = layer.update_with_version(v2.id, content="final content")

        history = layer.get_version_history(v3.id)
        assert len(history) == 3
        assert history[0].content == "final content"
        assert history[1].content == "updated content"
        assert history[2].content == "original content"

    def test_snapshot_and_restore(self, tmp_path: Path):
        """A snapshot can be exported and imported."""
        from animus_kernel.memory.layer import MemoryLayer

        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        layer.remember("persistent data")
        meta = layer.snapshot("integration-test")

        # Clear current store
        for mem in list(layer.store.list_all()):
            layer.store.delete(mem.id)
        assert len(layer.store.list_all()) == 0

        restored = layer.restore_snapshot(meta["path"])
        assert restored == 1
        assert len(layer.store.list_all()) == 1
        assert layer.store.list_all()[0].content == "persistent data"


class TestContractsValidation:
    def test_claim_schema_validates(self):
        """A claim payload passes the contracts schema."""
        from animus_contracts.validator import validate

        payload = {
            "object_id": "claim-001",
            "object_version": 1,
            "schema_id": "https://animus.local/schemas/claim.schema.json",
            "schema_version": "1.0.0",
            "owner_id": "owner-test",
            "workspace_id": "ws-test",
            "subject_domain": "self",
            "artifact_type": "claim",
            "cognitive_role": "knowledge",
            "workflow_status": "candidate",
            "epistemic_status": "supported",
            "lifecycle_status": "active",
            "storage_tier": "warm",
            "presentation": "canonical",
            "security_class": "public",
            "valid_time": {"valid_from": "2024-01-01T00:00:00", "valid_to": None},
            "transaction_time": {"recorded_at": "2024-01-01T00:00:00", "superseded_at": None},
            "provenance": {
                "created_by": "test",
                "source_refs": [],
                "derived_from": [],
                "trace_id": None,
            },
            "integrity": {"content_sha256": "a" * 64, "previous_version_sha256": None},
            "payload": {
                "proposition": "The system is reliable",
                "claim_kind": "factual",
                "supporting_evidence": ["evidence-1"],
                "contradicting_evidence": [],
                "confidence": 0.95,
            },
        }
        validate(payload, "claim")  # raises ValidationError on failure

    def test_memory_candidate_schema_validates(self):
        """A memory candidate payload passes the contracts schema."""
        from animus_contracts.validator import validate

        payload = {
            "candidate_id": "mc-test-001",
            "owner_id": "owner-test",
            "workspace_id": "ws-test",
            "proposed_object": {"content": "hello world"},
            "candidate_reason": "New memory ingested",
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
            "created_at": "2024-01-01T00:00:00",
        }
        validate(payload, "memory_candidate")


class TestCrossPackageIntegration:
    def test_sensitivity_gates_memory_and_egress(self):
        """Sensitivity from animus_types gates memory recall and egress policy."""
        from animus_kernel.memory.layer import MemoryLayer
        from animus_types import Sensitivity

        layer = MemoryLayer(data_dir=Path("/tmp/animus-integ-test"), backend="local")
        public = layer.remember("public info", sensitivity=Sensitivity.PUBLIC)
        secret = layer.remember("secret info", sensitivity=Sensitivity.SECRET)

        # Egress-safe recall should only return PUBLIC
        egress_results = layer.recall_for_egress("info")
        assert all(r.sensitivity == Sensitivity.PUBLIC for r in egress_results)
        assert any(r.id == public.id for r in egress_results)
        assert not any(r.id == secret.id for r in egress_results)
