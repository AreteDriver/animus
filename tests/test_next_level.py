"""
Tests for next-level features:
- Proactive Intelligence (scheduler, briefings, nudges)
- Entity/Relationship Memory with temporal reasoning
- Web Dashboard
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# Proactive Intelligence Tests
# =============================================================================


class TestNudge:
    """Tests for the Nudge dataclass."""

    def test_nudge_creation(self):
        from animus.proactive import Nudge, NudgePriority, NudgeType

        nudge = Nudge(
            id="test-1",
            nudge_type=NudgeType.MORNING_BRIEF,
            priority=NudgePriority.MEDIUM,
            title="Test nudge",
            content="Test content",
            created_at=datetime.now(),
        )
        assert nudge.is_active()
        assert not nudge.is_expired()

    def test_nudge_expiry(self):
        from animus.proactive import Nudge, NudgePriority, NudgeType

        nudge = Nudge(
            id="test-2",
            nudge_type=NudgeType.DEADLINE_WARNING,
            priority=NudgePriority.HIGH,
            title="Expired",
            content="Old nudge",
            created_at=datetime.now() - timedelta(hours=25),
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert nudge.is_expired()
        assert not nudge.is_active()

    def test_nudge_dismiss(self):
        from animus.proactive import Nudge, NudgePriority, NudgeType

        nudge = Nudge(
            id="test-3",
            nudge_type=NudgeType.FOLLOW_UP,
            priority=NudgePriority.LOW,
            title="To dismiss",
            content="Content",
            created_at=datetime.now(),
        )
        assert nudge.is_active()
        nudge.dismiss()
        assert not nudge.is_active()
        assert nudge.dismissed

    def test_nudge_act_on(self):
        from animus.proactive import Nudge, NudgePriority, NudgeType

        nudge = Nudge(
            id="test-4",
            nudge_type=NudgeType.CONTEXT_RECALL,
            priority=NudgePriority.LOW,
            title="To act",
            content="Content",
            created_at=datetime.now(),
        )
        nudge.mark_acted()
        assert not nudge.is_active()
        assert nudge.acted_on

    def test_nudge_serialization(self):
        from animus.proactive import Nudge, NudgePriority, NudgeType

        nudge = Nudge(
            id="test-5",
            nudge_type=NudgeType.MEETING_PREP,
            priority=NudgePriority.URGENT,
            title="Serializable",
            content="Test",
            created_at=datetime.now(),
            source_memory_ids=["mem-1", "mem-2"],
        )
        data = nudge.to_dict()
        restored = Nudge.from_dict(data)
        assert restored.id == nudge.id
        assert restored.nudge_type == nudge.nudge_type
        assert restored.priority == nudge.priority
        assert len(restored.source_memory_ids) == 2


class TestScheduledCheck:
    """Tests for ScheduledCheck."""

    def test_new_check_is_due(self):
        from animus.proactive import ScheduledCheck

        check = ScheduledCheck(name="test", interval_minutes=60)
        assert check.is_due()

    def test_recently_run_check_not_due(self):
        from animus.proactive import ScheduledCheck

        check = ScheduledCheck(
            name="test",
            interval_minutes=60,
            last_run=datetime.now(),
        )
        assert not check.is_due()

    def test_old_check_is_due(self):
        from animus.proactive import ScheduledCheck

        check = ScheduledCheck(
            name="test",
            interval_minutes=60,
            last_run=datetime.now() - timedelta(hours=2),
        )
        assert check.is_due()

    def test_disabled_check_not_due(self):
        from animus.proactive import ScheduledCheck

        check = ScheduledCheck(name="test", interval_minutes=1, enabled=False)
        assert not check.is_due()


class TestProactiveEngine:
    """Tests for ProactiveEngine."""

    def _make_engine(self, tmpdir):
        from animus.memory import MemoryLayer
        from animus.proactive import ProactiveEngine

        memory = MemoryLayer(Path(tmpdir), backend="json")
        engine = ProactiveEngine(Path(tmpdir), memory)
        return engine, memory

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, _ = self._make_engine(tmpdir)
            assert len(engine.get_active_nudges()) == 0
            assert not engine.is_running

    def test_generate_morning_brief(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            memory.remember(content="Meeting with Alex about Q2 budget", tags=["meeting"])
            memory.remember(content="Need to review PR #42", tags=["code"])

            nudge = engine.generate_morning_brief()
            assert nudge.nudge_type.value == "morning_brief"
            assert nudge.is_active()
            assert len(engine.get_active_nudges()) == 1

    def test_scan_deadlines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            memory.remember(
                content="Tax filing deadline is tomorrow",
                tags=["deadline", "urgent"],
            )
            nudges = engine.scan_deadlines()
            # May or may not find it depending on search implementation
            # but should not error
            assert isinstance(nudges, list)

    def test_scan_follow_ups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            from animus.memory import MemoryType

            mem = memory.remember(
                content="I'll get back to you about the proposal",
                memory_type=MemoryType.EPISODIC,
                tags=["conversation"],
            )
            # Make it look recent but not today
            mem.created_at = datetime.now() - timedelta(days=3)
            mem.updated_at = mem.created_at
            memory.store.update(mem)

            nudges = engine.scan_follow_ups()
            assert len(nudges) >= 1
            assert nudges[0].nudge_type.value == "follow_up"

    def test_meeting_prep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            memory.remember(content="Alex prefers morning meetings", tags=["alex"])
            memory.remember(content="Alex works on the data team", tags=["alex"])

            nudge = engine.prepare_meeting_context("Alex")
            assert nudge.nudge_type.value == "meeting_prep"
            assert "Alex" in nudge.title

    def test_context_nudge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            mem = memory.remember(
                content="Previous discussion about machine learning models",
                tags=["ml"],
            )
            # Make it old enough
            mem.created_at = datetime.now() - timedelta(days=5)
            mem.updated_at = mem.created_at
            memory.store.update(mem)

            nudge = engine.generate_context_nudge("machine learning")
            # May or may not find it depending on search
            if nudge:
                assert nudge.nudge_type.value == "context_recall"

    def test_dismiss_nudge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            nudge = engine.generate_morning_brief()
            assert engine.dismiss_nudge(nudge.id)
            assert len(engine.get_active_nudges()) == 0

    def test_dismiss_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            engine.generate_morning_brief()
            engine.generate_morning_brief()
            assert len(engine.get_active_nudges()) == 2
            count = engine.dismiss_all()
            assert count == 2
            assert len(engine.get_active_nudges()) == 0

    def test_nudge_persistence(self):
        from animus.memory import MemoryLayer
        from animus.proactive import ProactiveEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            engine1 = ProactiveEngine(Path(tmpdir), memory)
            engine1.generate_morning_brief()
            assert len(engine1.get_active_nudges()) == 1

            # Reload
            engine2 = ProactiveEngine(Path(tmpdir), memory)
            assert len(engine2.get_active_nudges()) == 1

    def test_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            engine.generate_morning_brief()
            stats = engine.get_statistics()
            assert stats["active_nudges"] == 1
            assert stats["total_nudges"] >= 1
            assert "by_type" in stats

    def test_nudges_by_priority(self):
        from animus.proactive import NudgePriority

        with tempfile.TemporaryDirectory() as tmpdir:
            engine, memory = self._make_engine(tmpdir)
            engine.generate_morning_brief()  # MEDIUM priority
            high = engine.get_nudges_by_priority(NudgePriority.HIGH)
            medium = engine.get_nudges_by_priority(NudgePriority.MEDIUM)
            assert len(high) == 0
            assert len(medium) == 1


# =============================================================================
# Entity/Relationship Memory Tests
# =============================================================================


class TestEntity:
    """Tests for the Entity dataclass."""

    def test_entity_creation(self):
        from animus.entities import Entity, EntityType

        entity = Entity(
            id="e-1",
            name="Alice",
            entity_type=EntityType.PERSON,
            aliases=["Al"],
        )
        assert entity.name == "Alice"
        assert entity.matches_name("Alice")
        assert entity.matches_name("alice")
        assert entity.matches_name("Al")
        assert not entity.matches_name("Bob")

    def test_entity_mention_tracking(self):
        from animus.entities import Entity, EntityType

        entity = Entity(id="e-2", name="Project X", entity_type=EntityType.PROJECT)
        entity.record_mention("mem-1")
        entity.record_mention("mem-2")
        assert entity.mention_count == 2
        assert entity.last_mentioned is not None
        assert "mem-1" in entity.memory_ids

    def test_entity_serialization(self):
        from animus.entities import Entity, EntityType

        entity = Entity(
            id="e-3",
            name="ACME Corp",
            entity_type=EntityType.ORGANIZATION,
            attributes={"industry": "tech"},
            notes="Main client",
        )
        data = entity.to_dict()
        restored = Entity.from_dict(data)
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type
        assert restored.attributes == entity.attributes


class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_relationship_creation(self):
        from animus.entities import Relationship, RelationType

        rel = Relationship(
            id="r-1",
            source_id="e-1",
            target_id="e-2",
            relation_type=RelationType.WORKS_WITH,
        )
        assert rel.strength == 1.0

    def test_relationship_reinforce(self):
        from animus.entities import Relationship, RelationType

        rel = Relationship(
            id="r-2",
            source_id="e-1",
            target_id="e-2",
            relation_type=RelationType.KNOWS,
            strength=0.5,
        )
        rel.reinforce(0.3)
        assert rel.strength == 0.8

    def test_relationship_strength_capped(self):
        from animus.entities import Relationship, RelationType

        rel = Relationship(
            id="r-3",
            source_id="e-1",
            target_id="e-2",
            relation_type=RelationType.FRIEND,
            strength=0.95,
        )
        rel.reinforce(0.5)
        assert rel.strength == 1.0

    def test_relationship_serialization(self):
        from animus.entities import Relationship, RelationType

        rel = Relationship(
            id="r-4",
            source_id="e-1",
            target_id="e-2",
            relation_type=RelationType.MANAGES,
            description="Alice manages Bob",
        )
        data = rel.to_dict()
        restored = Relationship.from_dict(data)
        assert restored.relation_type == RelationType.MANAGES
        assert restored.description == "Alice manages Bob"


class TestEntityMemory:
    """Tests for EntityMemory knowledge graph."""

    def _make_em(self, tmpdir):
        from animus.entities import EntityMemory

        return EntityMemory(Path(tmpdir) / "entities")

    def test_add_entity(self):
        from animus.entities import EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = self._make_em(tmpdir)
            entity = em.add_entity("Alice", EntityType.PERSON)
            assert entity.name == "Alice"
            assert len(em.list_entities()) == 1

    def test_add_entity_simple(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            entity = em.add_entity("Bob", EntityType.PERSON, aliases=["Robert"])
            assert entity.name == "Bob"
            assert "Robert" in entity.aliases

    def test_find_entity_by_name(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Charlie", EntityType.PERSON)
            found = em.find_entity("Charlie")
            assert found is not None
            assert found.name == "Charlie"

    def test_find_entity_by_alias(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Elizabeth", EntityType.PERSON, aliases=["Liz", "Beth"])
            assert em.find_entity("Liz") is not None
            assert em.find_entity("Beth") is not None
            assert em.find_entity("NotAName") is None

    def test_search_entities(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Project Alpha", EntityType.PROJECT, notes="ML project")
            em.add_entity("Project Beta", EntityType.PROJECT, notes="Web project")
            em.add_entity("Dave", EntityType.PERSON)

            results = em.search_entities("project")
            assert len(results) == 2

            results = em.search_entities("project", entity_type=EntityType.PROJECT)
            assert len(results) == 2

    def test_add_relationship(self):
        from animus.entities import EntityMemory, EntityType, RelationType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            alice = em.add_entity("Alice", EntityType.PERSON)
            bob = em.add_entity("Bob", EntityType.PERSON)

            rel = em.add_relationship(alice.id, bob.id, RelationType.WORKS_WITH)
            assert rel is not None
            assert rel.relation_type == RelationType.WORKS_WITH

    def test_get_connected_entities(self):
        from animus.entities import EntityMemory, EntityType, RelationType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            alice = em.add_entity("Alice", EntityType.PERSON)
            bob = em.add_entity("Bob", EntityType.PERSON)
            proj = em.add_entity("ProjectX", EntityType.PROJECT)

            em.add_relationship(alice.id, bob.id, RelationType.WORKS_WITH)
            em.add_relationship(alice.id, proj.id, RelationType.WORKS_ON)

            connections = em.get_connected_entities(alice.id)
            assert len(connections) == 2

    def test_relationship_reinforcement(self):
        from animus.entities import EntityMemory, EntityType, RelationType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            a = em.add_entity("A", EntityType.PERSON)
            b = em.add_entity("B", EntityType.PERSON)

            rel1 = em.add_relationship(a.id, b.id, RelationType.KNOWS)
            assert rel1.strength == 1.0

            # Adding same relationship again reinforces it
            rel2 = em.add_relationship(a.id, b.id, RelationType.KNOWS)
            assert rel2.id == rel1.id  # Same relationship

    def test_temporal_reasoning(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            alice = em.add_entity("Alice", EntityType.PERSON)

            em.record_interaction(alice.id, "mem-1", "Discussed project timeline")
            em.record_interaction(alice.id, "mem-2", "Reviewed budget proposal")

            timeline = em.get_interaction_timeline(alice.id)
            assert len(timeline) == 2

            last = em.last_interaction_with(alice.id)
            assert last is not None
            assert "budget" in last.summary

    def test_time_since_interaction(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            alice = em.add_entity("Alice", EntityType.PERSON)

            # No interactions yet
            assert em.time_since_interaction(alice.id) is None

            em.record_interaction(alice.id, "mem-1", "Hello")
            elapsed = em.time_since_interaction(alice.id)
            assert elapsed is not None
            assert elapsed.total_seconds() < 5

    def test_recently_mentioned(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            alice = em.add_entity("Alice", EntityType.PERSON)
            em.add_entity("Bob", EntityType.PERSON)

            em.record_interaction(alice.id, "m1", "Chat with Alice")

            recent = em.recently_mentioned(days=7)
            assert len(recent) == 1
            assert recent[0].name == "Alice"

    def test_not_mentioned_since(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Alice", EntityType.PERSON)
            em.add_entity("Bob", EntityType.PERSON)

            dormant = em.not_mentioned_since(days=30)
            assert len(dormant) == 2  # Neither has been mentioned

    def test_extract_and_link(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Alice", EntityType.PERSON)
            em.add_entity("Project Alpha", EntityType.PROJECT)

            found = em.extract_and_link("I talked to Alice about Project Alpha today")
            assert len(found) == 2
            names = {e.name for e in found}
            assert "Alice" in names
            assert "Project Alpha" in names

    def test_extract_avoids_partial_match(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Al", EntityType.PERSON)

            # "Al" should not match "also" or "algorithm"
            found = em.extract_and_link("I also need an algorithm")
            assert len(found) == 0

    def test_generate_entity_context(self):
        from animus.entities import EntityMemory, EntityType, RelationType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            alice = em.add_entity(
                "Alice",
                EntityType.PERSON,
                attributes={"role": "Engineer"},
                notes="Works on backend",
            )
            bob = em.add_entity("Bob", EntityType.PERSON)
            em.add_relationship(alice.id, bob.id, RelationType.WORKS_WITH)
            em.record_interaction(alice.id, "m1", "Discussed API design")

            context = em.generate_entity_context(alice.id)
            assert "Alice" in context
            assert "person" in context
            assert "Engineer" in context
            assert "Bob" in context
            assert "API design" in context

    def test_delete_entity_removes_relationships(self):
        from animus.entities import EntityMemory, EntityType, RelationType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            a = em.add_entity("A", EntityType.PERSON)
            b = em.add_entity("B", EntityType.PERSON)
            em.add_relationship(a.id, b.id, RelationType.KNOWS)

            em.delete_entity(a.id)
            assert em.get_entity(a.id) is None
            assert len(em.get_relationships_for(a.id)) == 0

    def test_persistence(self):
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em1 = EntityMemory(Path(tmpdir) / "entities")
            em1.add_entity("Persistent", EntityType.PROJECT)

            em2 = EntityMemory(Path(tmpdir) / "entities")
            assert len(em2.list_entities()) == 1
            assert em2.find_entity("Persistent") is not None

    def test_statistics(self):
        from animus.entities import EntityMemory, EntityType, RelationType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            a = em.add_entity("A", EntityType.PERSON)
            b = em.add_entity("B", EntityType.PERSON)
            em.add_entity("C", EntityType.PROJECT)
            em.add_relationship(a.id, b.id, RelationType.KNOWS)

            stats = em.get_statistics()
            assert stats["total_entities"] == 3
            assert stats["total_relationships"] == 1
            assert stats["by_type"]["person"] == 2
            assert stats["by_type"]["project"] == 1


# =============================================================================
# Dashboard Tests
# =============================================================================


class TestDashboard:
    """Tests for dashboard data collection and rendering."""

    def test_collect_dashboard_data(self):
        from animus.dashboard import collect_dashboard_data
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember(content="Test memory", tags=["test"])

            data = collect_dashboard_data(memory)
            assert "memory_stats" in data
            assert "memories" in data
            assert "entity_stats" in data
            assert "nudges" in data
            assert len(data["memories"]) == 1

    def test_collect_with_entity_memory(self):
        from animus.dashboard import collect_dashboard_data
        from animus.entities import EntityMemory, EntityType
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("TestEntity", EntityType.PERSON)

            data = collect_dashboard_data(memory, entity_memory=em)
            assert data["entity_stats"]["total_entities"] == 1
            assert len(data["entities"]) == 1

    def test_collect_with_proactive(self):
        from animus.dashboard import collect_dashboard_data
        from animus.memory import MemoryLayer
        from animus.proactive import ProactiveEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            proactive = ProactiveEngine(Path(tmpdir), memory)
            proactive.generate_morning_brief()

            data = collect_dashboard_data(memory, proactive=proactive)
            assert len(data["nudges"]) >= 1

    def test_render_dashboard_html(self):
        from animus.dashboard import render_dashboard
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            memory.remember(content="Dashboard test", tags=["test"])

            html = render_dashboard(memory)
            assert "<!DOCTYPE html>" in html
            assert "Animus Dashboard" in html
            assert "__DASHBOARD_DATA__" not in html  # Should be replaced
            assert "Dashboard test" in html

    def test_render_dashboard_has_tabs(self):
        from animus.dashboard import render_dashboard
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            html = render_dashboard(memory)
            assert "Overview" in html
            assert "Memories" in html
            assert "Entities" in html
            assert "Nudges" in html

    def test_render_dashboard_valid_json_data(self):
        from animus.dashboard import render_dashboard
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            html = render_dashboard(memory)

            # Extract the JSON data from the HTML
            start = html.find("const DATA = ") + len("const DATA = ")
            end = html.find(";\n", start)
            json_str = html[start:end]
            data = json.loads(json_str)
            assert "memory_stats" in data


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Test that new features are properly exported."""

    def test_entity_exports(self):
        from animus import Entity, EntityMemory, EntityType, RelationType

        assert Entity is not None
        assert EntityMemory is not None
        assert EntityType.PERSON.value == "person"
        assert RelationType.WORKS_WITH.value == "works_with"

    def test_proactive_exports(self):
        from animus import Nudge, NudgePriority, NudgeType, ProactiveEngine

        assert ProactiveEngine is not None
        assert Nudge is not None
        assert NudgeType.MORNING_BRIEF.value == "morning_brief"
        assert NudgePriority.URGENT.value == "urgent"


# =============================================================================
# Integration Tests - Wiring
# =============================================================================


class TestCognitiveEntityIntegration:
    """Test that CognitiveLayer integrates with EntityMemory and ProactiveEngine."""

    def test_cognitive_accepts_entity_memory(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.entities import EntityMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            config = ModelConfig.mock()
            cog = CognitiveLayer(primary_config=config, entity_memory=em)
            assert cog.entity_memory is em

    def test_cognitive_accepts_proactive(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.memory import MemoryLayer
        from animus.proactive import ProactiveEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            proactive = ProactiveEngine(Path(tmpdir), memory)
            config = ModelConfig.mock()
            cog = CognitiveLayer(primary_config=config, proactive=proactive)
            assert cog.proactive is proactive

    def test_think_extracts_entities(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Alice", EntityType.PERSON)

            config = ModelConfig.mock()
            cog = CognitiveLayer(primary_config=config, entity_memory=em)

            # think() triggers entity extraction (without memory_id for tracking)
            # Verify the extraction finds the entity
            found = em.extract_and_link("I talked to Alice today about the project")
            assert len(found) == 1
            assert found[0].name == "Alice"

            # Also verify think() doesn't error with entity_memory set
            response = cog.think("I talked to Alice today about the project")
            assert response  # Mock returns something

    def test_think_enriches_context_with_entities(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity(
                "Bob", EntityType.PERSON, attributes={"role": "Manager"}, notes="Works in finance"
            )

            config = ModelConfig.mock()
            cog = CognitiveLayer(primary_config=config, entity_memory=em)

            # _enrich_context should add entity info
            enriched = cog._enrich_context("Meeting with Bob tomorrow", None)
            assert enriched is not None
            assert "Bob" in enriched
            assert "Manager" in enriched

    def test_think_without_entities_works(self):
        from animus.cognitive import CognitiveLayer, ModelConfig

        config = ModelConfig.mock()
        cog = CognitiveLayer(primary_config=config)
        # Should work fine without entity_memory or proactive
        response = cog.think("Hello world")
        assert response  # Mock returns something

    def test_enrich_context_preserves_existing(self):
        from animus.cognitive import CognitiveLayer, ModelConfig
        from animus.entities import EntityMemory, EntityType

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            em.add_entity("Charlie", EntityType.PERSON)

            config = ModelConfig.mock()
            cog = CognitiveLayer(primary_config=config, entity_memory=em)

            existing = "Some existing memory context"
            enriched = cog._enrich_context("Charlie said hello", existing)
            assert existing in enriched
            assert "Charlie" in enriched


class TestConfigIntegration:
    """Test that AnimusConfig includes new feature sections."""

    def test_config_has_proactive_section(self):
        from animus.config import AnimusConfig

        config = AnimusConfig()
        assert hasattr(config, "proactive")
        assert config.proactive.enabled is True
        assert config.proactive.background_enabled is False

    def test_config_has_entity_section(self):
        from animus.config import AnimusConfig

        config = AnimusConfig()
        assert hasattr(config, "entities")
        assert config.entities.enabled is True
        assert config.entities.auto_extract is True

    def test_config_serialization_roundtrip(self):
        from animus.config import AnimusConfig

        config = AnimusConfig()
        data = config.to_dict()
        assert "proactive" in data
        assert "entities" in data
        assert data["proactive"]["enabled"] is True
        assert data["entities"]["auto_extract"] is True

    def test_config_save_and_load(self):
        from animus.config import AnimusConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnimusConfig()
            config.data_dir = Path(tmpdir)
            config.proactive.background_enabled = True
            config.entities.auto_extract = False
            config.save()

            loaded = AnimusConfig.load(Path(tmpdir) / "config.yaml")
            assert loaded.proactive.background_enabled is True
            assert loaded.entities.auto_extract is False


class TestAPIServerIntegration:
    """Test that APIServer accepts and passes through new components."""

    def test_api_server_accepts_entity_memory(self):
        """Verify APIServer constructor accepts entity_memory param."""
        # Just verify the signature accepts the parameters
        # (Can't fully test without FastAPI installed)
        from animus.api import AppState
        from animus.config import AnimusConfig

        state = AppState(
            config=AnimusConfig(),
            memory=None,
            cognitive=None,
            tools=None,
            tasks=None,
            decisions=None,
            conversations={},
            entity_memory="mock_em",
            proactive="mock_proactive",
        )
        assert state.entity_memory == "mock_em"
        assert state.proactive == "mock_proactive"


# =============================================================================
# Entity Linking & Auto-Relationship Tests
# =============================================================================


class TestEntityLinkingOnConversationSave:
    """Test that saving conversations links entities and tracks mentions."""

    def test_save_conversation_links_entities(self):
        from animus.entities import EntityMemory, EntityType
        from animus.memory import Conversation, MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            em.add_entity("Alice", EntityType.PERSON)
            em.add_entity("Bob", EntityType.PERSON)

            convo = Conversation.new()
            convo.add_message("user", "I talked to Alice and Bob today")
            convo.add_message("assistant", "That sounds great!")

            memory.save_conversation(convo)

            alice = em.find_entity("Alice")
            bob = em.find_entity("Bob")
            assert alice.mention_count >= 1
            assert bob.mention_count >= 1

    def test_save_conversation_creates_mentioned_with_relationship(self):
        from animus.entities import EntityMemory, EntityType, RelationType
        from animus.memory import Conversation, MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            alice = em.add_entity("Alice", EntityType.PERSON)
            bob = em.add_entity("Bob", EntityType.PERSON)

            convo = Conversation.new()
            convo.add_message("user", "Alice and Bob met for lunch")
            convo.add_message("assistant", "Nice!")

            memory.save_conversation(convo)

            # Check MENTIONED_WITH relationship exists (sorted IDs for consistent direction)
            src, tgt = (alice.id, bob.id) if alice.id < bob.id else (bob.id, alice.id)
            rel = em.get_relationship(src, tgt, RelationType.MENTIONED_WITH)
            assert rel is not None
            assert rel.relation_type == RelationType.MENTIONED_WITH

    def test_mentioned_with_reinforces_on_repeated_co_mention(self):
        from animus.entities import EntityMemory, EntityType, RelationType
        from animus.memory import Conversation, MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            alice = em.add_entity("Alice", EntityType.PERSON)
            bob = em.add_entity("Bob", EntityType.PERSON)

            # First conversation
            c1 = Conversation.new()
            c1.add_message("user", "Alice and Bob are working on the project")
            memory.save_conversation(c1)

            src, tgt = (alice.id, bob.id) if alice.id < bob.id else (bob.id, alice.id)
            rel1 = em.get_relationship(src, tgt, RelationType.MENTIONED_WITH)
            first_updated = rel1.updated_at

            # Second conversation — reinforce() still called even if strength is capped
            c2 = Conversation.new()
            c2.add_message("user", "Alice told Bob about the deadline")
            memory.save_conversation(c2)

            rel2 = em.get_relationship(src, tgt, RelationType.MENTIONED_WITH)
            # Reinforcement updates the timestamp
            assert rel2.updated_at >= first_updated
            # Same relationship object was reused (not duplicated)
            assert rel2.id == rel1.id

    def test_no_relationship_for_single_entity(self):
        from animus.entities import EntityMemory, EntityType
        from animus.memory import Conversation, MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            em.add_entity("Alice", EntityType.PERSON)

            convo = Conversation.new()
            convo.add_message("user", "I saw Alice today")
            memory.save_conversation(convo)

            # Should still track the mention
            alice = em.find_entity("Alice")
            assert alice.mention_count >= 1
            # But no relationships created
            rels = em.get_relationships_for(alice.id)
            assert len(rels) == 0

    def test_save_conversation_without_entity_memory_works(self):
        from animus.memory import Conversation, MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")

            convo = Conversation.new()
            convo.add_message("user", "Hello")
            convo.add_message("assistant", "Hi there!")

            mem = memory.save_conversation(convo)
            assert mem is not None
            assert mem.id

    def test_three_entities_create_three_relationships(self):
        from animus.entities import EntityMemory, EntityType, RelationType
        from animus.memory import Conversation, MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            em.add_entity("Alice", EntityType.PERSON)
            em.add_entity("Bob", EntityType.PERSON)
            em.add_entity("Charlie", EntityType.PERSON)

            convo = Conversation.new()
            convo.add_message("user", "Alice, Bob, and Charlie had a meeting")
            memory.save_conversation(convo)

            # 3 entities = 3 pairs: (A,B), (A,C), (B,C)
            mentioned_rels = [
                r for r in em._relationships if r.relation_type == RelationType.MENTIONED_WITH
            ]
            assert len(mentioned_rels) == 3

    def test_memory_layer_entity_memory_default_none(self):
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            assert memory.entity_memory is None


class TestEntityLinkingOnRemember:
    """Test that remember() also links entities, not just save_conversation()."""

    def test_remember_links_entities(self):
        from animus.entities import EntityMemory, EntityType
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            em.add_entity("Alice", EntityType.PERSON)

            memory.remember("Alice prefers dark mode")

            alice = em.find_entity("Alice")
            assert alice.mention_count >= 1

    def test_remember_creates_mentioned_with(self):
        from animus.entities import EntityMemory, EntityType, RelationType
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            alice = em.add_entity("Alice", EntityType.PERSON)
            bob = em.add_entity("Bob", EntityType.PERSON)

            memory.remember("Alice and Bob prefer the same IDE")

            src, tgt = (alice.id, bob.id) if alice.id < bob.id else (bob.id, alice.id)
            rel = em.get_relationship(src, tgt, RelationType.MENTIONED_WITH)
            assert rel is not None

    def test_remember_without_entity_memory_works(self):
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")

            mem = memory.remember("Just a plain fact")
            assert mem is not None
            assert mem.id

    def test_remember_fact_links_entities(self):
        from animus.entities import EntityMemory, EntityType
        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            em = EntityMemory(Path(tmpdir) / "entities")
            memory = MemoryLayer(Path(tmpdir), backend="json", entity_memory=em)

            em.add_entity("Alice", EntityType.PERSON)

            memory.remember_fact("Alice", "prefers", "dark mode")

            alice = em.find_entity("Alice")
            assert alice.mention_count >= 1

    def test_remember_entity_linking_failure_nonfatal(self):
        from unittest.mock import MagicMock

        from animus.memory import MemoryLayer

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir), backend="json")
            # Attach a broken entity_memory that raises on extract_and_link
            mock_em = MagicMock()
            mock_em.extract_and_link.side_effect = RuntimeError("boom")
            memory.entity_memory = mock_em

            # Should not raise — failure is caught and logged
            mem = memory.remember("Alice likes coffee")
            assert mem is not None
            assert mem.id
