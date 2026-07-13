"""Animus Dossier System — Persistent knowledge objects with evidence accumulation.

Ported and generalized from the Dossier investigative document intelligence
project. Every important entity in Animus has a Dossier that continuously
accumulates notes, evidence, timelines, citations, relationships, confidence
scores, and tasks over time.

Components:
    - NER Engine: Gazetteer-based entity extraction (generalized from Dossier)
    - Graph Analysis: Entity co-occurrence network analysis (from Dossier)
    - Models: Dossier data structures and schemas

Usage:
    from animus.dossier import Dossier, NEREngine, GraphAnalyzer
    from animus.dossier.models import EntityType

    engine = NEREngine()
    entities = engine.extract(text)

    dossier = Dossier(entity_id="animus", entity_type=EntityType.PROJECT)
    dossier.add_evidence(source="harvest", content="...", confidence=0.9)
"""

from __future__ import annotations

from animus.dossier.graph import (
    Community,
    GraphAnalyzer,
    GraphStats,
    NodeMetrics,
    PathResult,
)
from animus.dossier.models import (
    Dossier,
    Entity,
    EntityRelationship,
    EntityType,
    EvidenceItem,
)
from animus.dossier.ner import (
    CATEGORY_SIGNALS,
    NEREngine,
    classify_document,
    generate_title,
)

__all__ = [
    # Models
    "Dossier",
    "Entity",
    "EntityRelationship",
    "EntityType",
    "EvidenceItem",
    # NER
    "NEREngine",
    "CATEGORY_SIGNALS",
    "classify_document",
    "generate_title",
    # Graph
    "GraphAnalyzer",
    "GraphStats",
    "NodeMetrics",
    "Community",
    "PathResult",
]
