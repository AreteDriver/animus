"""
Entity and Relationship Memory System

Knowledge graph for tracking people, projects, places, and their relationships
with full temporal reasoning support.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("entities")


class EntityType(Enum):
    """Types of entities tracked."""

    PERSON = "person"
    PROJECT = "project"
    ORGANIZATION = "organization"
    PLACE = "place"
    TOPIC = "topic"
    EVENT = "event"
    TOOL = "tool"
    CUSTOM = "custom"


class RelationType(Enum):
    """Types of relationships between entities."""

    # Person relationships
    WORKS_WITH = "works_with"
    REPORTS_TO = "reports_to"
    KNOWS = "knows"
    FRIEND = "friend"
    FAMILY = "family"

    # Project/org relationships
    MEMBER_OF = "member_of"
    WORKS_ON = "works_on"
    OWNS = "owns"
    MANAGES = "manages"

    # General relationships
    RELATED_TO = "related_to"
    LOCATED_AT = "located_at"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    MENTIONED_WITH = "mentioned_with"


@dataclass
class Entity:
    """A tracked entity with attributes and interaction history."""

    id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_mentioned: datetime | None = None
    mention_count: int = 0
    memory_ids: list[str] = field(default_factory=list)
    notes: str = ""

    def matches_name(self, query: str) -> bool:
        """Check if query matches this entity's name or aliases."""
        q = query.lower().strip()
        if q == self.name.lower():
            return True
        return any(q == alias.lower() for alias in self.aliases)

    def record_mention(self, memory_id: str | None = None) -> None:
        """Record that this entity was mentioned."""
        self.last_mentioned = datetime.now()
        self.mention_count += 1
        self.updated_at = datetime.now()
        if memory_id and memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on this entity."""
        self.attributes[key] = value
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_mentioned": self.last_mentioned.isoformat() if self.last_mentioned else None,
            "mention_count": self.mention_count,
            "memory_ids": self.memory_ids[-100:],  # Keep last 100
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            aliases=data.get("aliases", []),
            attributes=data.get("attributes", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_mentioned=(
                datetime.fromisoformat(data["last_mentioned"])
                if data.get("last_mentioned")
                else None
            ),
            mention_count=data.get("mention_count", 0),
            memory_ids=data.get("memory_ids", []),
            notes=data.get("notes", ""),
        )


@dataclass
class Relationship:
    """A relationship between two entities."""

    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    description: str = ""
    strength: float = 1.0  # 0.0-1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def reinforce(self, amount: float = 0.1) -> None:
        """Reinforce this relationship (increases strength)."""
        self.strength = min(1.0, self.strength + amount)
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "description": self.description,
            "strength": self.strength,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            description=data.get("description", ""),
            strength=data.get("strength", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InteractionRecord:
    """A timestamped record of an interaction involving an entity."""

    timestamp: datetime
    entity_id: str
    memory_id: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "entity_id": self.entity_id,
            "memory_id": self.memory_id,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionRecord:
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            entity_id=data["entity_id"],
            memory_id=data["memory_id"],
            summary=data["summary"],
        )


class EntityMemory:
    """
    Knowledge graph for entities and their relationships.

    Provides:
    - Entity CRUD with aliases and attributes
    - Relationship tracking between entities
    - Temporal reasoning (when did I last interact with X?)
    - Interaction timeline for any entity
    - Entity extraction from text
    - Context generation for conversations about known entities
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []
        self._interactions: list[InteractionRecord] = []
        self._load()
        logger.info(
            f"EntityMemory initialized: {len(self._entities)} entities, "
            f"{len(self._relationships)} relationships"
        )

    def _load(self) -> None:
        """Load entities and relationships from disk."""
        entities_file = self.data_dir / "entities.json"
        if entities_file.exists():
            try:
                data = json.loads(entities_file.read_text())
                for item in data.get("entities", []):
                    ent = Entity.from_dict(item)
                    self._entities[ent.id] = ent
                for item in data.get("relationships", []):
                    self._relationships.append(Relationship.from_dict(item))
                for item in data.get("interactions", [])[-500:]:
                    self._interactions.append(InteractionRecord.from_dict(item))
            except Exception as e:
                logger.error(f"Failed to load entity data: {e}")

    def _save(self) -> None:
        """Persist entities and relationships to disk."""
        entities_file = self.data_dir / "entities.json"
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relationships": [r.to_dict() for r in self._relationships],
            "interactions": [i.to_dict() for i in self._interactions[-500:]],
        }
        entities_file.write_text(json.dumps(data, indent=2, default=str))

    # =========================================================================
    # Entity CRUD
    # =========================================================================

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        aliases: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
        notes: str = "",
    ) -> Entity:
        """
        Add a new entity.

        Args:
            name: Primary name
            entity_type: Type of entity
            aliases: Alternative names
            attributes: Key-value attributes
            notes: Freeform notes

        Returns:
            The created Entity
        """
        # Check for duplicate
        existing = self.find_entity(name)
        if existing:
            logger.info(f"Entity '{name}' already exists, updating")
            if aliases:
                for alias in aliases:
                    if alias.lower() not in [a.lower() for a in existing.aliases]:
                        existing.aliases.append(alias)
            if attributes:
                existing.attributes.update(attributes)
            if notes:
                existing.notes = notes
            existing.updated_at = datetime.now()
            self._save()
            return existing

        entity = Entity(
            id=str(uuid.uuid4()),
            name=name,
            entity_type=entity_type,
            aliases=aliases or [],
            attributes=attributes or {},
            notes=notes,
        )
        self._entities[entity.id] = entity
        self._save()
        logger.info(f"Added entity: {name} ({entity_type.value})")
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def find_entity(self, name: str) -> Entity | None:
        """Find an entity by name or alias (case-insensitive)."""
        for entity in self._entities.values():
            if entity.matches_name(name):
                return entity
        return None

    def search_entities(
        self,
        query: str,
        entity_type: EntityType | None = None,
        limit: int = 10,
    ) -> list[Entity]:
        """
        Search entities by name, alias, or attribute content.

        Args:
            query: Search string
            entity_type: Optional type filter
            limit: Max results

        Returns:
            Matching entities sorted by mention count
        """
        q = query.lower()
        results = []

        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue

            score = 0
            if q in entity.name.lower():
                score += 10
            for alias in entity.aliases:
                if q in alias.lower():
                    score += 5
            if q in entity.notes.lower():
                score += 2
            for val in entity.attributes.values():
                if isinstance(val, str) and q in val.lower():
                    score += 1

            if score > 0:
                results.append((score, entity))

        results.sort(key=lambda x: (-x[0], -x[1].mention_count))
        return [entity for _, entity in results[:limit]]

    def update_entity(self, entity_id: str, **kwargs: Any) -> Entity | None:
        """
        Update entity fields.

        Args:
            entity_id: ID of entity to update
            **kwargs: Fields to update (name, notes, aliases, attributes)

        Returns:
            Updated entity or None if not found
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return None

        if "name" in kwargs:
            entity.name = kwargs["name"]
        if "notes" in kwargs:
            entity.notes = kwargs["notes"]
        if "aliases" in kwargs:
            entity.aliases = kwargs["aliases"]
        if "attributes" in kwargs:
            entity.attributes.update(kwargs["attributes"])

        entity.updated_at = datetime.now()
        self._save()
        return entity

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity, its relationships, and interaction records."""
        if entity_id not in self._entities:
            return False

        del self._entities[entity_id]
        self._relationships = [
            r for r in self._relationships if r.source_id != entity_id and r.target_id != entity_id
        ]
        self._interactions = [i for i in self._interactions if i.entity_id != entity_id]
        self._save()
        logger.info(f"Deleted entity: {entity_id}")
        return True

    def list_entities(
        self,
        entity_type: EntityType | None = None,
        limit: int = 50,
    ) -> list[Entity]:
        """List entities, optionally filtered by type."""
        entities = list(self._entities.values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        entities.sort(key=lambda e: e.mention_count, reverse=True)
        return entities[:limit]

    # =========================================================================
    # Relationships
    # =========================================================================

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        description: str = "",
    ) -> Relationship | None:
        """
        Add a relationship between two entities.

        Returns:
            The relationship, or None if entities not found
        """
        if source_id not in self._entities or target_id not in self._entities:
            logger.warning("Cannot create relationship: entity not found")
            return None

        # Check for existing relationship
        existing = self.get_relationship(source_id, target_id, relation_type)
        if existing:
            existing.reinforce()
            self._save()
            return existing

        rel = Relationship(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            description=description,
        )
        self._relationships.append(rel)
        self._save()

        src = self._entities[source_id].name
        tgt = self._entities[target_id].name
        logger.info(f"Relationship: {src} --[{relation_type.value}]--> {tgt}")
        return rel

    def get_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType | None = None,
    ) -> Relationship | None:
        """Find an existing relationship."""
        for rel in self._relationships:
            if rel.source_id == source_id and rel.target_id == target_id:
                if relation_type is None or rel.relation_type == relation_type:
                    return rel
        return None

    def get_relationships_for(self, entity_id: str) -> list[Relationship]:
        """Get all relationships involving an entity."""
        return [
            r for r in self._relationships if r.source_id == entity_id or r.target_id == entity_id
        ]

    def get_connected_entities(self, entity_id: str) -> list[tuple[Entity, Relationship]]:
        """Get all entities connected to a given entity with their relationships."""
        results = []
        for rel in self.get_relationships_for(entity_id):
            other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            other = self._entities.get(other_id)
            if other:
                results.append((other, rel))
        return results

    # =========================================================================
    # Temporal Reasoning
    # =========================================================================

    def record_interaction(
        self,
        entity_id: str,
        memory_id: str,
        summary: str,
    ) -> None:
        """Record an interaction with an entity."""
        entity = self._entities.get(entity_id)
        if not entity:
            return

        entity.record_mention(memory_id)
        self._interactions.append(
            InteractionRecord(
                timestamp=datetime.now(),
                entity_id=entity_id,
                memory_id=memory_id,
                summary=summary[:200],
            )
        )
        self._save()

    def remove_interactions_for_memory(self, memory_id: str) -> int:
        """Remove all interaction records and entity references for a memory.

        Used when a memory is deleted (e.g. during consolidation) to prevent
        orphaned references.

        Returns:
            Number of interaction records removed
        """
        before = len(self._interactions)
        self._interactions = [i for i in self._interactions if i.memory_id != memory_id]
        removed = before - len(self._interactions)

        # Also remove memory_id from entity memory_ids lists
        refs_cleaned = False
        for entity in self._entities.values():
            if memory_id in entity.memory_ids:
                entity.memory_ids.remove(memory_id)
                refs_cleaned = True

        if removed > 0 or refs_cleaned:
            self._save()
        return removed

    def get_interaction_timeline(
        self,
        entity_id: str,
        since: datetime | None = None,
        limit: int = 20,
    ) -> list[InteractionRecord]:
        """
        Get the interaction timeline for an entity.

        Args:
            entity_id: Entity to query
            since: Only include interactions after this time
            limit: Max interactions to return

        Returns:
            List of interactions, newest first
        """
        interactions = [i for i in self._interactions if i.entity_id == entity_id]
        if since:
            interactions = [i for i in interactions if i.timestamp >= since]
        interactions.sort(key=lambda i: i.timestamp, reverse=True)
        return interactions[:limit]

    def last_interaction_with(self, entity_id: str) -> InteractionRecord | None:
        """Get the most recent interaction with an entity."""
        timeline = self.get_interaction_timeline(entity_id, limit=1)
        return timeline[0] if timeline else None

    def time_since_interaction(self, entity_id: str) -> timedelta | None:
        """Get the time elapsed since last interaction with an entity."""
        last = self.last_interaction_with(entity_id)
        if last:
            return datetime.now() - last.timestamp
        return None

    def recently_mentioned(self, days: int = 7) -> list[Entity]:
        """Get entities mentioned within the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        result = [
            e for e in self._entities.values() if e.last_mentioned and e.last_mentioned >= cutoff
        ]
        result.sort(key=lambda e: e.last_mentioned or datetime.min, reverse=True)
        return result

    def not_mentioned_since(self, days: int = 30) -> list[Entity]:
        """Get entities NOT mentioned in the last N days (may need follow-up)."""
        cutoff = datetime.now() - timedelta(days=days)
        result = [
            e
            for e in self._entities.values()
            if e.last_mentioned is None or e.last_mentioned < cutoff
        ]
        result.sort(key=lambda e: e.mention_count, reverse=True)
        return result

    # =========================================================================
    # Entity Extraction
    # =========================================================================

    # Common words that look like proper nouns but aren't entities
    _NER_STOPWORDS: set[str] = {
        "I",
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Here",
        "There",
        "What",
        "When",
        "Where",
        "Who",
        "Why",
        "How",
        "Which",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Today",
        "Tomorrow",
        "Yesterday",
        "Yes",
        "No",
        "Ok",
        "Sure",
        "Hello",
        "Hi",
        "Hey",
        "Thanks",
        "Thank",
        "Please",
        "Sorry",
        "Also",
        "Just",
        "Well",
        "Now",
        "Then",
        "But",
        "And",
        "Or",
        "User",
        "Animus",
        "Conversation",
    }

    def discover_entities(self, text: str) -> list[str]:
        """
        Discover potential new entity names from text using heuristic NER.

        Looks for capitalized proper noun patterns that don't match existing
        entities or common stopwords. Returns candidate names for review.

        Args:
            text: Text to scan for potential entities

        Returns:
            List of candidate entity name strings
        """
        # Match capitalized words/phrases that look like proper nouns
        # Pattern: one or more capitalized words in sequence
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        matches = re.findall(pattern, text)

        candidates = []
        seen = set()
        for match in matches:
            # Skip single-character or very short matches
            if len(match) < 2:
                continue
            # Skip stopwords
            if match in self._NER_STOPWORDS:
                continue
            # Skip if it's a known entity already
            if self.find_entity(match):
                continue
            # Deduplicate
            if match.lower() in seen:
                continue
            seen.add(match.lower())
            candidates.append(match)

        return candidates

    def extract_and_link(
        self,
        text: str,
        memory_id: str | None = None,
        auto_discover: bool = False,
    ) -> list[Entity]:
        """
        Extract known entity mentions from text and record interactions.

        When a memory_id is provided, records interactions (bumping mention counts)
        and auto-creates MENTIONED_WITH relationships between co-occurring entities.

        When auto_discover is True, uses heuristic NER to find and auto-create
        new entities from capitalized proper noun patterns in the text.

        Args:
            text: Text to scan for entity mentions
            memory_id: Optional memory ID to link
            auto_discover: If True, auto-create entities from proper noun patterns

        Returns:
            List of entities found in the text (including any newly discovered)
        """
        # Auto-discover new entities first (if enabled)
        if auto_discover:
            candidates = self.discover_entities(text)
            for name in candidates:
                self.add_entity(name, EntityType.CUSTOM, notes="Auto-discovered from text")
                logger.info(f"Auto-discovered entity: {name}")

        found = []
        text_lower = text.lower()

        for entity in self._entities.values():
            names_to_check = [entity.name] + entity.aliases
            for name in names_to_check:
                # Word boundary matching to avoid partial matches
                pattern = r"\b" + re.escape(name.lower()) + r"\b"
                if re.search(pattern, text_lower):
                    found.append(entity)
                    if memory_id:
                        self.record_interaction(
                            entity.id,
                            memory_id,
                            text[:200],
                        )
                    break  # Don't double-count same entity

        # Auto-create MENTIONED_WITH relationships for co-occurring entities
        if memory_id and len(found) >= 2:
            for i, e1 in enumerate(found):
                for e2 in found[i + 1 :]:
                    # Use sorted IDs for consistent directionality
                    src, tgt = (e1.id, e2.id) if e1.id < e2.id else (e2.id, e1.id)
                    self.add_relationship(
                        src,
                        tgt,
                        RelationType.MENTIONED_WITH,
                        f"Co-mentioned in memory {memory_id[:8]}",
                    )

        return found

    # =========================================================================
    # Context Generation
    # =========================================================================

    def generate_entity_context(self, entity_id: str) -> str:
        """
        Generate a context string about an entity for prompt augmentation.

        Args:
            entity_id: Entity to generate context for

        Returns:
            Human-readable context string
        """
        entity = self._entities.get(entity_id)
        if not entity:
            return ""

        parts = [f"{entity.name} ({entity.entity_type.value})"]

        if entity.aliases:
            parts.append(f"Also known as: {', '.join(entity.aliases)}")

        if entity.attributes:
            attrs = [f"{k}: {v}" for k, v in entity.attributes.items()]
            parts.append(f"Attributes: {'; '.join(attrs)}")

        if entity.notes:
            parts.append(f"Notes: {entity.notes}")

        # Relationships
        connections = self.get_connected_entities(entity_id)
        if connections:
            rel_strs = []
            for other, rel in connections:
                direction = (
                    f"{entity.name} {rel.relation_type.value} {other.name}"
                    if rel.source_id == entity_id
                    else f"{other.name} {rel.relation_type.value} {entity.name}"
                )
                rel_strs.append(direction)
            parts.append(f"Relationships: {'; '.join(rel_strs)}")

        # Recent interactions
        timeline = self.get_interaction_timeline(entity_id, limit=3)
        if timeline:
            parts.append("Recent interactions:")
            for interaction in timeline:
                date_str = interaction.timestamp.strftime("%b %d")
                parts.append(f"  [{date_str}] {interaction.summary}")

        # Time since last interaction
        elapsed = self.time_since_interaction(entity_id)
        if elapsed:
            if elapsed.days > 0:
                parts.append(f"Last interaction: {elapsed.days} days ago")
            else:
                hours = elapsed.seconds // 3600
                parts.append(f"Last interaction: {hours} hours ago")

        return "\n".join(parts)

    def get_context_for_text(self, text: str) -> str:
        """
        Extract entities from text and generate combined context.

        Args:
            text: User input or conversation text

        Returns:
            Context string about all mentioned entities
        """
        entities = self.extract_and_link(text)
        if not entities:
            return ""

        contexts = []
        for entity in entities[:5]:  # Limit to 5 to avoid context overflow
            ctx = self.generate_entity_context(entity.id)
            if ctx:
                contexts.append(ctx)

        if not contexts:
            return ""

        return "Known entities mentioned:\n\n" + "\n\n---\n\n".join(contexts)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get entity memory statistics."""
        by_type: dict[str, int] = {}
        for entity in self._entities.values():
            by_type[entity.entity_type.value] = by_type.get(entity.entity_type.value, 0) + 1

        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "total_interactions": len(self._interactions),
            "by_type": by_type,
            "recently_active": len(self.recently_mentioned(days=7)),
            "dormant": len(self.not_mentioned_since(days=30)),
        }
