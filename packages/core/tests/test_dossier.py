"""Tests for the Animus Dossier package (knowledge representation layer)."""

from __future__ import annotations

import pytest

from animus.dossier import NEREngine
from animus.dossier.models import (
    Dossier,
    Entity,
    EntityRelationship,
    EntityType,
    EvidenceItem,
)
from animus.dossier.ner import ExtractionResult


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestEntityType:
    def test_entity_type_values(self):
        assert EntityType.PERSON.name == "PERSON"
        assert EntityType.COMPANY.name == "COMPANY"
        assert EntityType.REPOSITORY.name == "REPOSITORY"

    def test_unknown_fallback(self):
        assert EntityType.UNKNOWN.name == "UNKNOWN"


class TestEvidenceItem:
    def test_creation(self):
        item = EvidenceItem(
            source="web",
            content="Test evidence",
            confidence=0.8,
            tags=["test", "sample"],
        )
        assert item.source == "web"
        assert item.confidence == 0.8

    def test_confidence_validation(self):
        with pytest.raises(ValueError, match="confidence must be between"):
            EvidenceItem(source="x", content="y", confidence=1.5)

        with pytest.raises(ValueError, match="confidence must be between"):
            EvidenceItem(source="x", content="y", confidence=-0.1)


class TestEntity:
    def test_auto_canonical(self):
        entity = Entity(name="Alice Smith", type=EntityType.PERSON)
        assert entity.canonical == "alice smith"

    def test_custom_canonical(self):
        entity = Entity(name="Alice", type=EntityType.PERSON, canonical="alice_smith")
        assert entity.canonical == "alice_smith"


class TestDossier:
    def test_default_creation(self):
        dossier = Dossier(name="Test Entity", entity_type=EntityType.PROJECT)
        assert dossier.name == "Test Entity"
        assert dossier.entity_type == EntityType.PROJECT
        assert dossier.confidence_score == 0.0  # No evidence yet
        assert dossier.freshness_score == 1.0
        assert dossier.evidence == []
        assert dossier.relationships == []

    def test_evidence_accumulation(self):
        dossier = Dossier(name="Test", entity_type=EntityType.PERSON)
        dossier.evidence.append(EvidenceItem(source="web", content="Found online"))
        assert len(dossier.evidence) == 1


# ---------------------------------------------------------------------------
# NER Engine tests
# ---------------------------------------------------------------------------


class TestNEREngine:
    def test_initialization(self):
        engine = NEREngine()
        assert engine._gazetteers["people"] == set()
        assert engine._gazetteers["places"] == set()
        assert engine._gazetteers["orgs"] == set()

    def test_add_gazetteer(self):
        engine = NEREngine()
        engine.add_gazetteer("people", {"Alice", "Bob"})
        assert "alice" in engine._gazetteers["people"]

        engine.add_gazetteer("orgs", {"Acme Corp", "Globex"})
        assert "acme corp" in engine._gazetteers["orgs"]

    def test_extract_empty_text(self):
        engine = NEREngine()
        result = engine.extract("")
        assert result.people == []
        assert result.dates == []
        assert result.keywords == []

    def test_extract_dates(self):
        engine = NEREngine()
        text = "Meeting scheduled for 2024-03-15 and March 20, 2024."
        result = engine.extract(text)
        # Should find ISO date
        assert any("2024" in d["name"] for d in result.dates)

    def test_gazetteer_match(self):
        engine = NEREngine()
        engine.add_gazetteer("people", {"Alice", "Bob"})
        text = "Alice and Bob are working on the project."
        result = engine.extract(text)
        names = {e["name"] for e in result.people}
        assert "Alice" in names
        assert "Bob" in names

    def test_heuristic_names(self):
        engine = NEREngine()
        text = "Attendees included John Smith and Mary Johnson at the conference."
        result = engine.extract(text)
        names = {e["name"] for e in result.people}
        assert "John Smith" in names
        assert "Mary Johnson" in names

    def test_keywords(self):
        engine = NEREngine()
        text = "machine learning and artificial intelligence are important."
        result = engine.extract(text)
        # Should extract some keywords
        assert len(result.keywords) > 0


# ---------------------------------------------------------------------------
# Graph Analyzer tests (requires networkx)
# ---------------------------------------------------------------------------


class TestGraphAnalyzer:
    def test_initialization(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        assert analyzer._graph.number_of_nodes() == 0

    def test_add_node_and_edge(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("alice", name="Alice", entity_type="person")
        analyzer.add_node("acme", name="Acme Corp", entity_type="company")
        analyzer.add_edge("alice", "acme", weight=3)

        assert analyzer._graph.number_of_nodes() == 2
        assert analyzer._graph.number_of_edges() == 1

    def test_duplicate_edge_weight_sum(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("a", name="A", entity_type="test")
        analyzer.add_node("b", name="B", entity_type="test")
        analyzer.add_edge("a", "b", weight=1)
        analyzer.add_edge("a", "b", weight=2)

        assert analyzer._graph["a"]["b"]["weight"] == 3

    def test_get_stats_empty(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        stats = analyzer.get_stats()
        assert stats.node_count == 0
        assert stats.edge_count == 0

    def test_get_stats_populated(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("a", name="A", entity_type="test")
        analyzer.add_node("b", name="B", entity_type="test")
        analyzer.add_edge("a", "b", weight=2)

        stats = analyzer.get_stats()
        assert stats.node_count == 2
        assert stats.edge_count == 1
        assert stats.avg_degree == 1.0

    def test_get_centrality(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("hub", name="Hub", entity_type="test")
        analyzer.add_node("spoke1", name="Spoke 1", entity_type="test")
        analyzer.add_node("spoke2", name="Spoke 2", entity_type="test")
        analyzer.add_edge("hub", "spoke1", weight=1)
        analyzer.add_edge("hub", "spoke2", weight=1)

        results = analyzer.get_centrality(metric="degree", limit=10)
        assert len(results) == 3
        # Hub should have highest degree
        assert results[0].name == "Hub"
        assert results[0].weighted_degree == 2

    def test_invalid_metric(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        with pytest.raises(ValueError, match="Invalid metric"):
            analyzer.get_centrality(metric="invalid")

    def test_get_neighbors(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("alice", name="Alice", entity_type="person")
        analyzer.add_node("bob", name="Bob", entity_type="person")
        analyzer.add_node("carol", name="Carol", entity_type="person")
        analyzer.add_edge("alice", "bob", weight=5)
        analyzer.add_edge("alice", "carol", weight=1)

        neighbors = analyzer.get_neighbors("alice", hops=1, min_weight=2)
        assert len(neighbors) == 1
        assert neighbors[0]["name"] == "Bob"

    def test_find_shortest_path(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("a", name="A", entity_type="test")
        analyzer.add_node("b", name="B", entity_type="test")
        analyzer.add_node("c", name="C", entity_type="test")
        analyzer.add_edge("a", "b", weight=1)
        analyzer.add_edge("b", "c", weight=1)

        path = analyzer.find_shortest_path("a", "c")
        assert path is not None
        assert len(path.nodes) == 3
        assert path.hops == 2

    def test_shortest_path_no_path(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("a", name="A", entity_type="test")
        analyzer.add_node("b", name="B", entity_type="test")

        path = analyzer.find_shortest_path("a", "b")
        assert path is None

    def test_get_subgraph(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        for i in range(1, 4):
            analyzer.add_node(f"n{i}", name=f"Node {i}", entity_type="test")
        analyzer.add_edge("n1", "n2", weight=1)
        analyzer.add_edge("n2", "n3", weight=1)

        sub = analyzer.get_subgraph(["n1", "n2"])
        assert len(sub["nodes"]) == 2
        assert len(sub["edges"]) == 1

    def test_serialization_roundtrip(self):
        pytest.importorskip("networkx")
        from animus.dossier.graph import GraphAnalyzer

        analyzer = GraphAnalyzer()
        analyzer.add_node("alice", name="Alice", entity_type="person")
        analyzer.add_node("bob", name="Bob", entity_type="person")
        analyzer.add_edge("alice", "bob", weight=3)

        data = analyzer.to_dict()
        restored = GraphAnalyzer.from_dict(data)

        assert restored._graph.number_of_nodes() == 2
        assert restored._graph.number_of_edges() == 1
        assert restored._graph["alice"]["bob"]["weight"] == 3
