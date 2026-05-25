"""Tests for the Sensitivity enum + Memory.sensitivity field (Stage 2.A)."""

from __future__ import annotations

from datetime import datetime

import pytest

from animus.memory.types import Memory, MemoryType, Sensitivity


class TestSensitivityEnum:
    """The four-tier disclosure classification."""

    def test_enum_values(self):
        assert Sensitivity.PUBLIC.value == "public"
        assert Sensitivity.PERSONAL.value == "personal"
        assert Sensitivity.CONFIDENTIAL.value == "confidential"
        assert Sensitivity.SECRET.value == "secret"

    def test_enum_has_four_members(self):
        assert len(list(Sensitivity)) == 4

    def test_enum_construct_from_value(self):
        assert Sensitivity("public") is Sensitivity.PUBLIC
        assert Sensitivity("personal") is Sensitivity.PERSONAL
        assert Sensitivity("confidential") is Sensitivity.CONFIDENTIAL
        assert Sensitivity("secret") is Sensitivity.SECRET

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Sensitivity("medium")


class TestMemoryDefault:
    """Default sensitivity is PUBLIC for backward compat."""

    def test_default_is_public(self):
        now = datetime.now()
        mem = Memory(
            id="m1",
            content="x",
            memory_type=MemoryType.SEMANTIC,
            created_at=now,
            updated_at=now,
            metadata={},
        )
        assert mem.sensitivity is Sensitivity.PUBLIC

    def test_explicit_sensitivity_preserved(self):
        now = datetime.now()
        mem = Memory(
            id="m1",
            content="x",
            memory_type=MemoryType.SEMANTIC,
            created_at=now,
            updated_at=now,
            metadata={},
            sensitivity=Sensitivity.CONFIDENTIAL,
        )
        assert mem.sensitivity is Sensitivity.CONFIDENTIAL

    def test_create_factory_default(self):
        mem = Memory.create(content="hello")
        assert mem.sensitivity is Sensitivity.PUBLIC

    def test_create_factory_with_sensitivity(self):
        mem = Memory.create(content="client notes", sensitivity=Sensitivity.CONFIDENTIAL)
        assert mem.sensitivity is Sensitivity.CONFIDENTIAL


class TestSerialization:
    """to_dict / from_dict round-trip preserves sensitivity."""

    def test_to_dict_emits_value_not_enum(self):
        mem = Memory.create(content="x", sensitivity=Sensitivity.SECRET)
        d = mem.to_dict()
        assert d["sensitivity"] == "secret"
        assert not isinstance(d["sensitivity"], Sensitivity)

    def test_to_dict_default_public(self):
        mem = Memory.create(content="x")
        assert mem.to_dict()["sensitivity"] == "public"

    def test_round_trip_preserves_sensitivity(self):
        original = Memory.create(content="x", sensitivity=Sensitivity.CONFIDENTIAL)
        round_tripped = Memory.from_dict(original.to_dict())
        assert round_tripped.sensitivity is Sensitivity.CONFIDENTIAL

    def test_round_trip_all_tiers(self):
        for tier in Sensitivity:
            original = Memory.create(content="x", sensitivity=tier)
            assert Memory.from_dict(original.to_dict()).sensitivity is tier

    def test_from_dict_missing_sensitivity_defaults_to_public(self):
        """Backward compat: legacy memory dicts without the new field must load."""
        now = datetime.now().isoformat()
        legacy_data = {
            "id": "legacy-1",
            "content": "old memory",
            "memory_type": "semantic",
            "created_at": now,
            "updated_at": now,
            "metadata": {},
        }
        mem = Memory.from_dict(legacy_data)
        assert mem.sensitivity is Sensitivity.PUBLIC

    def test_from_dict_with_explicit_sensitivity(self):
        now = datetime.now().isoformat()
        data = {
            "id": "x",
            "content": "y",
            "memory_type": "episodic",
            "created_at": now,
            "updated_at": now,
            "metadata": {},
            "sensitivity": "personal",
        }
        assert Memory.from_dict(data).sensitivity is Sensitivity.PERSONAL


class TestMemoryLayerIntegration:
    """MemoryLayer.remember() accepts and propagates sensitivity."""

    def test_remember_default_public(self, tmp_path):
        from animus.memory.layer import MemoryLayer

        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        mem = layer.remember(content="hello")
        assert mem.sensitivity is Sensitivity.PUBLIC

    def test_remember_explicit_confidential(self, tmp_path):
        from animus.memory.layer import MemoryLayer

        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        mem = layer.remember(
            content="client meeting notes",
            sensitivity=Sensitivity.CONFIDENTIAL,
        )
        assert mem.sensitivity is Sensitivity.CONFIDENTIAL

    def test_remember_with_all_tiers(self, tmp_path):
        from animus.memory.layer import MemoryLayer

        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        for tier in Sensitivity:
            mem = layer.remember(content=f"tier={tier.value}", sensitivity=tier)
            assert mem.sensitivity is tier


class TestAllowedTiersGate:
    """Adversarial tests — recall(allowed_tiers=...) must enforce the tier gate."""

    def _seed_all_tiers(self, tmp_path):
        from animus.memory.layer import MemoryLayer

        layer = MemoryLayer(data_dir=tmp_path, backend="local")
        layer.remember(content="public fact about TRIM", sensitivity=Sensitivity.PUBLIC)
        layer.remember(content="my own draft notes", sensitivity=Sensitivity.PERSONAL)
        layer.remember(content="TIAID client kickoff", sensitivity=Sensitivity.CONFIDENTIAL)
        layer.remember(content="sk-ant-fake-credential-stub", sensitivity=Sensitivity.SECRET)
        return layer

    def test_public_scope_only_sees_public(self, tmp_path):
        layer = self._seed_all_tiers(tmp_path)
        results = layer.recall(query="", allowed_tiers={Sensitivity.PUBLIC})
        assert len(results) == 1
        assert results[0].sensitivity is Sensitivity.PUBLIC
        assert "TRIM" in results[0].content

    def test_confidential_scope_invisible_to_public(self, tmp_path):
        """The headline adversarial test — TIAID material cannot leak to MCP."""
        layer = self._seed_all_tiers(tmp_path)
        results = layer.recall(query="TIAID", allowed_tiers={Sensitivity.PUBLIC})
        assert results == []

    def test_secret_scope_invisible_to_public(self, tmp_path):
        layer = self._seed_all_tiers(tmp_path)
        results = layer.recall(query="credential", allowed_tiers={Sensitivity.PUBLIC})
        assert results == []

    def test_local_cli_scope_sees_three_tiers(self, tmp_path):
        layer = self._seed_all_tiers(tmp_path)
        results = layer.recall(
            query="",
            allowed_tiers={
                Sensitivity.PUBLIC,
                Sensitivity.PERSONAL,
                Sensitivity.CONFIDENTIAL,
            },
            limit=100,
        )
        tiers = {r.sensitivity for r in results}
        assert tiers == {
            Sensitivity.PUBLIC,
            Sensitivity.PERSONAL,
            Sensitivity.CONFIDENTIAL,
        }
        # Secret is excluded — explicit opt-in required
        assert Sensitivity.SECRET not in tiers

    def test_no_filter_returns_all_backward_compat(self, tmp_path):
        layer = self._seed_all_tiers(tmp_path)
        results = layer.recall(query="", limit=100)
        # No allowed_tiers → no filter → all 4 tiers visible
        assert len(results) == 4

    def test_empty_set_returns_nothing(self, tmp_path):
        layer = self._seed_all_tiers(tmp_path)
        results = layer.recall(query="", allowed_tiers=set(), limit=100)
        assert results == []

    def test_secret_explicit_opt_in(self, tmp_path):
        layer = self._seed_all_tiers(tmp_path)
        results = layer.recall(query="", allowed_tiers={Sensitivity.SECRET})
        assert len(results) == 1
        assert results[0].sensitivity is Sensitivity.SECRET
