"""Animus Dossier — Data models for persistent knowledge objects.

Inspired by the Dossier investigative document intelligence project,
generalized for universal knowledge representation across Animus.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class EntityType(Enum):
    """Types of entities that can have dossiers."""

    PERSON = auto()
    COMPANY = auto()
    PROJECT = auto()
    REPOSITORY = auto()
    TECHNOLOGY = auto()
    RESEARCH_TOPIC = auto()
    CONCEPT = auto()
    LOCATION = auto()
    ORGANIZATION = auto()
    EVENT = auto()
    DOCUMENT = auto()
    UNKNOWN = auto()


@dataclass
class EvidenceItem:
    """A single piece of evidence attached to a dossier."""

    source: str  # Where did this come from?
    content: str  # The evidence itself
    confidence: float = 0.5  # 0.0–1.0
    timestamp: datetime = field(default_factory=datetime.now)
    provenance: dict[str, Any] = field(default_factory=dict)
    # e.g., {"url": "...", "page": 3, "line": 42}
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class Entity:
    """A named entity extracted from documents or external sources."""

    name: str
    type: EntityType
    canonical: str = ""  # Normalized form for deduplication
    aliases: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    occurrence_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.canonical:
            self.canonical = self.name.lower().strip()


@dataclass
class EntityRelationship:
    """A relationship between two entities."""

    source_id: str
    target_id: str
    relation_type: str  # e.g., "works_with", "located_in", "depends_on"
    weight: float = 1.0  # Strength of relationship
    evidence: list[EvidenceItem] = field(default_factory=list)
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)


@dataclass
class Dossier:
    """A persistent knowledge object for an entity.

    Every important object in Animus has a Dossier that continuously
    accumulates knowledge over time.
    """

    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: EntityType = EntityType.UNKNOWN
    name: str = ""
    summary: str = ""  # Auto-generated or human-written summary

    # Timelines
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_reviewed: datetime | None = None

    # Knowledge accumulation
    notes: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[EntityRelationship] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)

    # Confidence and quality
    confidence_score: float = 0.0  # Overall confidence in dossier completeness
    freshness_score: float = 1.0  # 1.0 = fresh, 0.0 = stale
    tags: list[str] = field(default_factory=list)

    # Embeddings (optional)
    embedding: list[float] | None = None
    embedding_model: str = ""

    def add_evidence(
        self,
        source: str,
        content: str,
        confidence: float = 0.5,
        provenance: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        """Add a new evidence item to the dossier."""
        item = EvidenceItem(
            source=source,
            content=content,
            confidence=confidence,
            provenance=provenance or {},
        )
        self.evidence.append(item)
        self.updated_at = datetime.now()
        self._update_confidence()
        return item

    def add_relationship(
        self,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        evidence: EvidenceItem | None = None,
    ) -> EntityRelationship:
        """Add a relationship to another entity."""
        rel = EntityRelationship(
            source_id=self.entity_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            evidence=[evidence] if evidence else [],
        )
        self.relationships.append(rel)
        self.updated_at = datetime.now()
        return rel

    def add_note(self, note: str) -> None:
        """Add a free-form note to the dossier."""
        self.notes.append(note)
        self.updated_at = datetime.now()

    def get_evidence_by_source(self, source: str) -> list[EvidenceItem]:
        """Retrieve all evidence from a specific source."""
        return [e for e in self.evidence if e.source == source]

    def get_evidence_by_confidence(self, min_confidence: float) -> list[EvidenceItem]:
        """Retrieve evidence meeting a confidence threshold."""
        return [e for e in self.evidence if e.confidence >= min_confidence]

    def _update_confidence(self) -> None:
        """Recalculate overall confidence based on evidence."""
        if not self.evidence:
            self.confidence_score = 0.0
            return

        # Simple average of evidence confidence, weighted by count
        total_confidence = sum(e.confidence for e in self.evidence)
        self.confidence_score = min(1.0, total_confidence / len(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        """Serialize dossier to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.name,
            "name": self.name,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_reviewed": self.last_reviewed.isoformat() if self.last_reviewed else None,
            "confidence_score": self.confidence_score,
            "freshness_score": self.freshness_score,
            "evidence_count": len(self.evidence),
            "relationship_count": len(self.relationships),
            "tags": self.tags,
        }

    def compute_hash(self) -> str:
        """Compute a stable hash of dossier contents for integrity checking."""
        content = (
            f"{self.entity_id}:{self.name}:{self.summary}:{len(self.evidence)}:{len(self.notes)}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
