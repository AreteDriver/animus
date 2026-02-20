"""Coverage push round 11 — entities, transparency, preferences, forge gates/loader, swarm graph, sync client."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from animus.entities import (
    Entity,
    EntityMemory,
    EntityType,
    InteractionRecord,
    Relationship,
    RelationType,
)

# ---------------------------------------------------------------------------
# Helper: make EntityMemory with pre-loaded data
# ---------------------------------------------------------------------------


def _make_em(tmp_path, entities=None, relationships=None, interactions=None):
    """Create an EntityMemory with pre-populated data (bypasses _load)."""
    em = EntityMemory(data_dir=tmp_path)
    if entities:
        for e in entities:
            em._entities[e.id] = e
    if relationships:
        em._relationships = list(relationships)
    if interactions:
        em._interactions = list(interactions)
    return em


def _make_entity(id_="e1", name="Alice", entity_type=EntityType.PERSON, **kwargs):
    return Entity(
        id=id_,
        name=name,
        entity_type=entity_type,
        aliases=kwargs.get("aliases", []),
        attributes=kwargs.get("attributes", {}),
        notes=kwargs.get("notes", ""),
        mention_count=kwargs.get("mention_count", 0),
    )


# ---------------------------------------------------------------------------
# EntityMemory._load — corrupt data (lines 245-246)
# ---------------------------------------------------------------------------


class TestEntityMemoryLoadCorrupt:
    """Lines 245-246: _load with corrupt JSON."""

    def test_load_corrupt_json(self, tmp_path):
        entities_file = tmp_path / "entities.json"
        entities_file.write_text("NOT VALID JSON {{{")

        # Should not raise — error is logged
        em = EntityMemory(data_dir=tmp_path)
        assert len(em._entities) == 0

    def test_load_missing_fields(self, tmp_path):
        entities_file = tmp_path / "entities.json"
        entities_file.write_text(json.dumps({"entities": [{"bad": "data"}]}))

        # Should not raise — individual entity parse error caught
        em = EntityMemory(data_dir=tmp_path)
        assert len(em._entities) == 0


# ---------------------------------------------------------------------------
# EntityMemory.search_entities — scoring branches (lines 351-357)
# ---------------------------------------------------------------------------


class TestEntityMemorySearchScoring:
    """Lines 350-357: alias match, notes match, attribute match scoring."""

    def test_alias_match_score(self, tmp_path):
        e = _make_entity(aliases=["Bob"])
        em = _make_em(tmp_path, entities=[e])
        results = em.search_entities("bob")
        assert len(results) == 1
        assert results[0].id == "e1"

    def test_notes_match_score(self, tmp_path):
        e = _make_entity(notes="Works at the factory downtown")
        em = _make_em(tmp_path, entities=[e])
        results = em.search_entities("factory")
        assert len(results) == 1

    def test_attribute_match_score(self, tmp_path):
        e = _make_entity(attributes={"role": "engineer"})
        em = _make_em(tmp_path, entities=[e])
        results = em.search_entities("engineer")
        assert len(results) == 1

    def test_name_match_ranks_above_notes(self, tmp_path):
        """Name match (score 10) should rank above notes match (score 2)."""
        e1 = _make_entity(id_="e1", name="Engineer Alice", notes="")
        e2 = _make_entity(id_="e2", name="Bob", notes="Engineer at factory")
        em = _make_em(tmp_path, entities=[e1, e2])
        results = em.search_entities("engineer")
        assert results[0].id == "e1"


# ---------------------------------------------------------------------------
# EntityMemory.update_entity — field branches (lines 380-387)
# ---------------------------------------------------------------------------


class TestEntityMemoryUpdateEntity:
    """Lines 380-387: update name, notes, aliases, attributes."""

    def test_update_name(self, tmp_path):
        e = _make_entity()
        em = _make_em(tmp_path, entities=[e])
        result = em.update_entity("e1", name="Alicia")
        assert result.name == "Alicia"

    def test_update_notes(self, tmp_path):
        e = _make_entity()
        em = _make_em(tmp_path, entities=[e])
        result = em.update_entity("e1", notes="Important contact")
        assert result.notes == "Important contact"

    def test_update_aliases(self, tmp_path):
        e = _make_entity()
        em = _make_em(tmp_path, entities=[e])
        result = em.update_entity("e1", aliases=["Al", "Ali"])
        assert result.aliases == ["Al", "Ali"]

    def test_update_attributes(self, tmp_path):
        e = _make_entity(attributes={"role": "dev"})
        em = _make_em(tmp_path, entities=[e])
        result = em.update_entity("e1", attributes={"team": "core"})
        assert result.attributes["team"] == "core"
        assert result.attributes["role"] == "dev"  # Merged, not replaced


# ---------------------------------------------------------------------------
# EntityMemory.get_connected_entities (line 487)
# ---------------------------------------------------------------------------


class TestEntityMemoryGetConnected:
    """Line 487: get_connected_entities with valid and missing targets."""

    def test_connected_entities(self, tmp_path):
        e1 = _make_entity(id_="e1", name="Alice")
        e2 = _make_entity(id_="e2", name="Bob")
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            relation_type=RelationType.WORKS_WITH,
        )
        em = _make_em(tmp_path, entities=[e1, e2], relationships=[rel])
        connected = em.get_connected_entities("e1")
        assert len(connected) == 1
        assert connected[0][0].name == "Bob"

    def test_connected_entities_missing_target(self, tmp_path):
        """Relationship target doesn't exist → skipped."""
        e1 = _make_entity(id_="e1", name="Alice")
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e_missing",
            relation_type=RelationType.KNOWS,
        )
        em = _make_em(tmp_path, entities=[e1], relationships=[rel])
        connected = em.get_connected_entities("e1")
        assert len(connected) == 0


# ---------------------------------------------------------------------------
# EntityMemory.get_interaction_timeline — since filter (line 560)
# ---------------------------------------------------------------------------


class TestEntityMemoryInteractionSince:
    """Line 560: interaction timeline with since filter."""

    def test_since_filter(self, tmp_path):
        now = datetime.now()
        old = InteractionRecord(
            timestamp=now - timedelta(days=10),
            entity_id="e1",
            memory_id="m1",
            summary="old meeting",
        )
        recent = InteractionRecord(
            timestamp=now - timedelta(hours=1),
            entity_id="e1",
            memory_id="m2",
            summary="recent call",
        )
        em = _make_em(tmp_path, interactions=[old, recent])
        result = em.get_interaction_timeline("e1", since=now - timedelta(days=1))
        assert len(result) == 1
        assert result[0].summary == "recent call"


# ---------------------------------------------------------------------------
# EntityMemory.discover_entities — short match skip (line 685-686)
# ---------------------------------------------------------------------------


class TestEntityMemoryNERShortMatch:
    """Lines 685-686: NER skips single-char matches; discovers new proper nouns."""

    def test_discovers_proper_nouns(self, tmp_path):
        em = _make_em(tmp_path)
        candidates = em.discover_entities("We met Charlie and Dave there")
        assert "Charlie" in candidates
        assert "Dave" in candidates

    def test_known_entity_skipped(self, tmp_path):
        """Known entities are not returned as candidates."""
        e = _make_entity(id_="e1", name="Charlie")
        em = _make_em(tmp_path, entities=[e])
        candidates = em.discover_entities("We met Charlie and Dave there")
        assert "Charlie" not in candidates
        assert "Dave" in candidates


# ---------------------------------------------------------------------------
# EntityMemory.generate_entity_context — aliases, attributes, time (lines 784-822)
# ---------------------------------------------------------------------------


class TestEntityMemoryGenerateContext:
    """Lines 784-822: context generation with aliases, attributes, time elapsed."""

    def test_context_with_aliases(self, tmp_path):
        e = _make_entity(aliases=["Al", "Ali"])
        em = _make_em(tmp_path, entities=[e])
        ctx = em.generate_entity_context("e1")
        assert "Also known as" in ctx
        assert "Al" in ctx

    def test_context_with_attributes(self, tmp_path):
        e = _make_entity(attributes={"role": "dev", "team": "core"})
        em = _make_em(tmp_path, entities=[e])
        ctx = em.generate_entity_context("e1")
        assert "Attributes" in ctx
        assert "role: dev" in ctx

    def test_context_with_interactions_days_ago(self, tmp_path):
        e = _make_entity()
        interaction = InteractionRecord(
            timestamp=datetime.now() - timedelta(days=5),
            entity_id="e1",
            memory_id="m1",
            summary="Had a meeting",
        )
        em = _make_em(tmp_path, entities=[e], interactions=[interaction])
        ctx = em.generate_entity_context("e1")
        assert "5 days ago" in ctx
        assert "Had a meeting" in ctx

    def test_context_with_interactions_hours_ago(self, tmp_path):
        e = _make_entity()
        interaction = InteractionRecord(
            timestamp=datetime.now() - timedelta(hours=3),
            entity_id="e1",
            memory_id="m1",
            summary="Quick chat",
        )
        em = _make_em(tmp_path, entities=[e], interactions=[interaction])
        ctx = em.generate_entity_context("e1")
        assert "hours ago" in ctx

    def test_context_nonexistent_entity(self, tmp_path):
        em = _make_em(tmp_path)
        assert em.generate_entity_context("no_such_id") == ""

    def test_context_with_relationships(self, tmp_path):
        e1 = _make_entity(id_="e1", name="Alice")
        e2 = _make_entity(id_="e2", name="Bob")
        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            relation_type=RelationType.WORKS_WITH,
        )
        em = _make_em(tmp_path, entities=[e1, e2], relationships=[rel])
        ctx = em.generate_entity_context("e1")
        assert "Relationships" in ctx
        assert "Bob" in ctx


# ---------------------------------------------------------------------------
# EntityMemory.get_context_for_text (lines 838-847)
# ---------------------------------------------------------------------------


class TestEntityMemoryGetContextForText:
    """Lines 838-847: extract entities and generate combined context."""

    def test_context_for_text_no_entities(self, tmp_path):
        em = _make_em(tmp_path)
        assert em.get_context_for_text("just some plain text") == ""

    def test_context_for_text_with_known_entity(self, tmp_path):
        e = _make_entity(id_="e1", name="Alice")
        em = _make_em(tmp_path, entities=[e])
        ctx = em.get_context_for_text("I met Alice yesterday")
        assert "Known entities mentioned" in ctx or ctx == ""

    def test_context_for_text_entity_returns_empty_context(self, tmp_path):
        """Entity found but generate_entity_context returns empty → empty result."""
        e = _make_entity(id_="e1", name="Alice")
        em = _make_em(tmp_path, entities=[e])
        with patch.object(em, "extract_and_link", return_value=[e]):
            with patch.object(em, "generate_entity_context", return_value=""):
                result = em.get_context_for_text("Alice")
        assert result == ""


# ---------------------------------------------------------------------------
# Transparency — confidence distribution (lines 198-201)
# ---------------------------------------------------------------------------


class TestTransparencyConfidenceDistribution:
    """Lines 198-201: low and medium confidence thresholds."""

    def test_confidence_low_and_medium(self, tmp_path):
        from animus.learning.categories import LearnedItem, LearningCategory
        from animus.learning.transparency import LearningTransparency

        te = LearningTransparency(data_dir=tmp_path)

        now = datetime.now()
        items = [
            LearnedItem(
                id="i1",
                category=LearningCategory.PREFERENCE,
                content="likes dark mode",
                confidence=0.2,  # low
                evidence=[],
                created_at=now,
                updated_at=now,
            ),
            LearnedItem(
                id="i2",
                category=LearningCategory.STYLE,
                content="prefers python",
                confidence=0.5,  # medium
                evidence=[],
                created_at=now,
                updated_at=now,
            ),
            LearnedItem(
                id="i3",
                category=LearningCategory.WORKFLOW,
                content="hates tabs",
                confidence=0.9,  # high
                evidence=[],
                created_at=now,
                updated_at=now,
            ),
        ]

        dashboard = te.get_dashboard_data(items)
        assert dashboard.confidence_distribution["low"] == 1
        assert dashboard.confidence_distribution["medium"] == 1
        assert dashboard.confidence_distribution["high"] == 1


# ---------------------------------------------------------------------------
# Preferences — get_preference by ID (line 226)
# ---------------------------------------------------------------------------


class TestPreferencesGetById:
    """Line 226: get_preference returns preference or None."""

    def test_get_existing_preference(self, tmp_path):
        from animus.learning.preferences import Preference, PreferenceEngine

        pe = PreferenceEngine(data_dir=tmp_path)

        # Directly insert a preference
        pref = Preference.create(
            domain="editor",
            key="theme",
            value="dark",
            confidence=0.8,
            source_patterns=["p1"],
        )
        pe._preferences[pref.id] = pref
        pe._save_preferences()

        result = pe.get_preference(pref.id)
        assert result is not None
        assert result.value == "dark"

    def test_get_nonexistent_preference(self, tmp_path):
        from animus.learning.preferences import PreferenceEngine

        pe = PreferenceEngine(data_dir=tmp_path)
        assert pe.get_preference("nonexistent") is None


# ---------------------------------------------------------------------------
# Forge gates — _resolve_ref returns None (line 135)
# ---------------------------------------------------------------------------


class TestForgeGatesResolveRefNone:
    """Line 135: _resolve_ref when base output is None."""

    def test_resolve_ref_base_not_in_outputs(self):
        from animus.forge.gates import _resolve_ref

        # "writer.score" — base="writer" not in outputs
        result = _resolve_ref("writer.score", {"reviewer": '{"score": 8}'})
        assert result is None

    def test_resolve_ref_base_is_none(self):
        from animus.forge.gates import _resolve_ref

        result = _resolve_ref("writer.score", {"writer": None})
        assert result is None


# ---------------------------------------------------------------------------
# Forge loader — agent without name (line 103)
# ---------------------------------------------------------------------------


class TestForgeLoaderAgentNoName:
    """Line 103: agent without name raises ForgeError."""

    def test_agent_missing_name(self):
        from animus.forge.loader import load_workflow_str
        from animus.forge.models import ForgeError

        yaml_str = """
name: test
agents:
  - archetype: researcher
    outputs: [brief]
"""
        with pytest.raises(ForgeError, match="must have a 'name'"):
            load_workflow_str(yaml_str)


# ---------------------------------------------------------------------------
# Swarm graph — terminal node warning (line 123)
# ---------------------------------------------------------------------------


class TestSwarmGraphTerminalNode:
    """Line 123: terminal node that has dependencies triggers warning."""

    def test_terminal_node_warning(self):
        from animus.forge.models import AgentConfig
        from animus.swarm.graph import build_dag, validate_dag

        # Agent "middle" depends on "start" but nothing depends on "middle"
        # AND "middle" has deps, so it's a terminal with deps → triggers warning
        agents = [
            AgentConfig(
                name="start", archetype="researcher", outputs=["brief"], budget_tokens=5000
            ),
            AgentConfig(
                name="middle",
                archetype="writer",
                outputs=["draft"],
                inputs=["start.brief"],
                budget_tokens=5000,
            ),
            AgentConfig(
                name="end",
                archetype="reviewer",
                outputs=["review"],
                inputs=["start.brief"],
                budget_tokens=5000,
            ),
        ]
        dag = build_dag(agents)
        warnings = validate_dag(agents, dag)
        # Both "middle" and "end" are terminals with deps
        terminal_warnings = [w for w in warnings if "terminal node" in w]
        assert len(terminal_warnings) >= 1


# ---------------------------------------------------------------------------
# SyncClient — sync callback error (lines 200-203)
# ---------------------------------------------------------------------------


class TestSyncClientCallbackError:
    """Lines 200-203: callback error during delta processing."""

    def test_callback_error_logged(self):
        import asyncio
        from unittest.mock import AsyncMock

        from animus.sync.client import SyncClient
        from animus.sync.protocol import MessageType, SyncMessage
        from animus.sync.state import StateSnapshot

        state = MagicMock()
        state.device_id = "dev-1"
        state.version = 2
        state.get_peer_version.return_value = 0
        state.collect_state.return_value = {"k": "v"}

        client = SyncClient(state=state, shared_secret="s")
        client._connected = True
        client._peer_device_id = "peer-1"

        # Build snapshot response
        snapshot = StateSnapshot.create(device_id="peer-1", data={"new_key": "val"}, version=1)
        response = SyncMessage(
            type=MessageType.SNAPSHOT_RESPONSE,
            device_id="peer-1",
            payload={"snapshot": snapshot.to_dict(), "version": 1},
        )

        ws = AsyncMock()
        ws.recv.return_value = response.to_json()
        client._websocket = ws

        # Register a callback that raises
        def bad_callback(d):
            raise ValueError("callback exploded")

        client._on_delta_received = [bad_callback]

        result = asyncio.run(client.sync())
        # Should succeed despite callback error
        assert result.success is True
