"""Animus Dossier — Named Entity Recognition Engine.

Generalized and ported from the Dossier investigative document intelligence
project. Uses a layered approach: gazetteer lookup, pattern matching, heuristic
NER, and keyword extraction. No GPU required — runs anywhere.

Usage:
    from animus.dossier.ner import NEREngine

    engine = NEREngine()
    engine.add_gazetteer("people", {"alice smith", "bob jones"})
    result = engine.extract(text)
    # result.people, result.places, result.orgs, result.dates, etc.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionResult:
    """Result of NER extraction."""

    people: list[dict[str, Any]] = field(default_factory=list)
    places: list[dict[str, Any]] = field(default_factory=list)
    orgs: list[dict[str, Any]] = field(default_factory=list)
    dates: list[dict[str, Any]] = field(default_factory=list)
    keywords: list[dict[str, Any]] = field(default_factory=list)


class NEREngine:
    """Named Entity Recognition Engine with user-extensible gazetteers.

    Layer 1: Gazetteer lookup (known entities)
    Layer 2: Pattern-based extraction (dates, addresses, identifiers)
    Layer 3: Heuristic NER (capitalized multi-word sequences)
    Layer 4: Keyword extraction (TF-based frequency analysis)
    """

    # Title patterns that precede names
    TITLE_PATTERNS = (
        r"(?:Mr\.|Mrs\.|Ms\.|Dr\.|Judge|Det\.|Detective|Agent|"
        r"Senator|Governor|President|Prince|Professor|Atty\.|Attorney)"
    )

    # Date patterns
    DATE_PATTERNS = [
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b(?:19|20)\d{2}\b",  # standalone years
    ]

    # Common stop words for keyword extraction
    STOP_WORDS: set[str] = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "were", "are",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "that", "this", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "his", "her", "their", "our", "your", "my", "him",
        "them", "us", "not", "no", "nor", "as", "if", "then", "than", "so",
        "up", "out", "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "same", "each", "every", "all", "both",
        "few", "more", "most", "other", "some", "such", "only", "own",
        "just", "also", "very", "often", "here", "there", "when", "where",
        "why", "how", "what", "which", "who", "whom", "whose", "any",
        "many", "much", "over", "under", "again", "further", "once",
        "said", "one", "two", "three", "first", "second", "new", "old",
        "see", "page", "document", "file", "yes", "no", "per", "via",
        "i", "me", "re", "cc", "q", "a", "mr", "ms", "mrs", "dr",
    }

    def __init__(self) -> None:
        """Initialize with empty gazetteers."""
        self._gazetteers: dict[str, set[str]] = {
            "people": set(),
            "places": set(),
            "orgs": set(),
        }
        self._compiled: dict[str, re.Pattern | None] = {
            "people": None,
            "places": None,
            "orgs": None,
        }

    def add_gazetteer(self, entity_type: str, names: set[str]) -> None:
        """Add names to a gazetteer and recompile regex.

        Args:
            entity_type: "people", "places", or "orgs"
            names: Set of entity names (lowercase)
        """
        if entity_type not in self._gazetteers:
            raise ValueError(f"Unknown entity type: {entity_type}")

        self._gazetteers[entity_type].update(n.lower().strip() for n in names)
        self._compiled[entity_type] = self._compile_regex(
            self._gazetteers[entity_type]
        )

    def _compile_regex(self, names: set[str]) -> re.Pattern | None:
        """Compile names into a single alternation regex."""
        if not names:
            return None
        sorted_names = sorted(names, key=len, reverse=True)
        escaped = [re.escape(n) for n in sorted_names]
        return re.compile("|".join(escaped))

    def extract(self, text: str) -> ExtractionResult:
        """Extract all entities from text.

        Args:
            text: Input text to analyze

        Returns:
            ExtractionResult with people, places, orgs, dates, keywords
        """
        if not text:
            return ExtractionResult()

        text_normalized = re.sub(r"\s+", " ", text)
        text_lower = text_normalized.lower()

        people = Counter()
        places = Counter()
        orgs = Counter()
        dates = Counter()

        # Layer 1: Gazetteer lookup
        for entity_type in ("people", "places", "orgs"):
            pattern = self._compiled.get(entity_type)
            if pattern is None:
                continue

            for match in pattern.finditer(text_lower):
                name = match.group()
                if entity_type == "people":
                    people[name.title()] += 1
                elif entity_type == "places":
                    places[name.title()] += 1
                else:
                    orgs[name.title()] += 1

        # Layer 2: Pattern-based extraction (dates)
        for pattern in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dates[match.group().strip()] += 1

        # Layer 3: Heuristic NER (capitalized multi-word sequences)
        self._extract_heuristic_names(text_normalized, text_lower, people)

        # Layer 4: Keyword extraction
        keywords = self._extract_keywords(text_normalized)

        return ExtractionResult(
            people=[{"name": k, "count": v} for k, v in people.most_common(100)],
            places=[{"name": k, "count": v} for k, v in places.most_common(50)],
            orgs=[{"name": k, "count": v} for k, v in orgs.most_common(50)],
            dates=[{"name": k, "count": v} for k, v in dates.most_common(50)],
            keywords=keywords,
        )

    def _extract_heuristic_names(
        self, text_normalized: str, text_lower: str, people: Counter
    ) -> None:
        """Find capitalized multi-word sequences that might be names."""
        pattern = (
            r"(?<!\.\s)(?<!\n)(?<!^)"
            r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,3})\b"
        )
        known_sets = [
            self._gazetteers["people"],
            self._gazetteers["places"],
            self._gazetteers["orgs"],
        ]

        for match in re.finditer(pattern, text_normalized):
            candidate = match.group().strip()
            candidate_lower = candidate.lower()

            # Skip if already in gazetteers
            if any(candidate_lower in s for s in known_sets):
                continue

            # Skip common false positives
            if self._is_false_positive(candidate_lower):
                continue

            # Heuristic: if preceded by a title, it's a person
            pre_context = text_normalized[max(0, match.start() - 15): match.start()]
            if re.search(self.TITLE_PATTERNS, pre_context):
                people[candidate] += 1
                continue

            # If 2 words and both capitalized, likely a person name
            words = candidate.split()
            if len(words) == 2 and all(w[0].isupper() for w in words):
                people[candidate] += 1

    def _is_false_positive(self, candidate: str) -> bool:
        """Check if a candidate is likely a false positive."""
        false_positives = {
            "the united states", "united states", "pursuant to",
            "direct examination", "cross examination", "legal counsel",
            "registered agent", "beneficial owner", "managing director",
            "general counsel", "chief executive", "executive director",
            "outside counsel", "special counsel", "investigation update",
            "case summary", "executive summary", "supplemental report",
            "witness statements", "next steps", "financial evidence",
            "banking relationships", "corporate entities",
            "corporate structure", "lobbying activities",
            "political contributions", "related entities",
        }
        if candidate in false_positives:
            return True

        skip_words = {
            "international", "island", "islands", "county", "country",
            "management", "department", "district", "corporation",
            "company", "holdings", "limited", "foundation", "institute",
            "university", "committee", "commission", "association",
            "trust", "group", "partners", "services", "solutions",
            "industries", "enterprises", "consulting", "advisors",
            "advisory", "capital", "ventures", "media", "labs",
            "records", "aircraft", "period", "details", "compensation",
            "principal", "filing", "disclosure", "profile",
            "engagement", "various", "status", "notes", "source",
            "modern", "general", "special", "national", "federal",
            "regional", "annual", "total", "summary", "report",
            "section", "chapter", "article", "schedule", "exhibit",
            "appendix",
        }
        return any(w in candidate for w in skip_words)

    def _extract_keywords(self, text: str, top_n: int = 50) -> list[dict[str, Any]]:
        """Extract significant keywords using term frequency."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        filtered = [w for w in words if w not in self.STOP_WORDS]
        counts = Counter(filtered)

        # Boost bigrams
        for i in range(len(filtered) - 1):
            bigram = f"{filtered[i]} {filtered[i + 1]}"
            if filtered[i] not in self.STOP_WORDS and filtered[i + 1] not in self.STOP_WORDS:
                counts[bigram] += 1

        return [{"word": k, "count": v} for k, v in counts.most_common(top_n)]


# ═══════════════════════════════════════════════════════════════════
# DOCUMENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

CATEGORY_SIGNALS: dict[str, list[str]] = {
    "correspondence": [
        "dear", "sincerely", "regards", "re:", "from:", "to:", "cc:",
        "memorandum", "memo", "letter",
    ],
    "report": [
        "incident report", "case number", "reporting officer",
        "investigation", "detective", "police report",
        "supplemental report", "field report",
    ],
    "legal": [
        "plaintiff", "defendant", "motion", "court order", "filed",
        "docket", "case no", "civil action", "complaint", "indictment",
    ],
    "email": [
        "subject:", "from:", "to:", "cc:", "date:", "message-id:",
        "mime-version", "content-type", "sent from my",
        "forwarded message", "original message",
    ],
    "technical": [
        "architecture", "design doc", "rfc", "api", "endpoint",
        "database", "schema", "migration", "deployment",
    ],
}


def classify_document(text: str, filename: str = "") -> str:
    """Classify a document into a category based on content signals.

    Args:
        text: Document content
        filename: Optional filename for additional signals

    Returns:
        Category name (or "other" if no strong signals)
    """
    from pathlib import Path

    text_lower = text[:5000].lower()
    filename_lower = filename.lower()

    scores: dict[str, int] = {}
    for category, signals in CATEGORY_SIGNALS.items():
        for signal in signals:
            count = text_lower.count(signal)
            scores[category] = scores.get(category, 0) + count
            if signal in filename_lower:
                scores[category] = scores.get(category, 0) + 5

    if not scores:
        return "other"

    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "other"


def generate_title(text: str, filename: str) -> str:
    """Generate a descriptive title from document content.

    Args:
        text: Document content
        filename: Original filename

    Returns:
        Generated title
    """
    from pathlib import Path

    lines = text.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        if 10 < len(line) < 120 and not line.startswith(("page", "Page", "#")):
            if len(line) < 80:
                return line

    stem = Path(filename).stem if filename else "Untitled"
    return stem.replace("_", " ").replace("-", " ").title()
