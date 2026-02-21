"""Knowledge domain classification and persona routing."""

from __future__ import annotations

from animus_bootstrap.personas.engine import PersonaProfile

# Simple keyword-to-domain mapping
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code",
        "programming",
        "function",
        "bug",
        "debug",
        "python",
        "rust",
        "javascript",
        "api",
        "git",
        "deploy",
    ],
    "architecture": [
        "architecture",
        "design",
        "pattern",
        "system",
        "scalab",
        "microservice",
        "database",
        "infrastructure",
    ],
    "devops": [
        "docker",
        "kubernetes",
        "ci/cd",
        "pipeline",
        "deploy",
        "terraform",
        "ansible",
        "monitoring",
    ],
    "writing": [
        "write",
        "essay",
        "story",
        "article",
        "blog",
        "draft",
        "edit",
        "proofread",
    ],
    "brainstorming": [
        "idea",
        "brainstorm",
        "creative",
        "innovate",
        "concept",
        "strategy",
    ],
    "art": [
        "draw",
        "paint",
        "design",
        "visual",
        "illustration",
        "graphic",
        "color",
        "style",
    ],
    "science": [
        "research",
        "experiment",
        "hypothesis",
        "data",
        "analysis",
        "scientific",
        "study",
    ],
    "health": [
        "health",
        "exercise",
        "diet",
        "sleep",
        "wellness",
        "medical",
        "symptom",
    ],
    "finance": [
        "money",
        "budget",
        "invest",
        "stock",
        "crypto",
        "tax",
        "financial",
        "savings",
    ],
    "general": [],  # Catch-all
}


class KnowledgeDomainRouter:
    """Routes questions to the persona with relevant domain expertise."""

    def classify_topic(self, text: str) -> list[str]:
        """Classify message text into topic domains via keyword matching."""
        text_lower = text.lower()
        matches: list[str] = []

        for domain, keywords in _DOMAIN_KEYWORDS.items():
            if domain == "general":
                continue
            for keyword in keywords:
                if keyword in text_lower:
                    if domain not in matches:
                        matches.append(domain)
                    break

        return matches if matches else ["general"]

    def find_best_persona(
        self, topics: list[str], personas: list[PersonaProfile]
    ) -> PersonaProfile | None:
        """Find the persona with the best domain match.

        Scores each persona by number of matching domains.
        Returns None if no persona has relevant domains.
        """
        best: PersonaProfile | None = None
        best_score = 0

        for persona in personas:
            if not persona.active:
                continue
            if not persona.knowledge_domains:
                continue

            # Check excluded topics first
            for topic in topics:
                if topic in persona.excluded_topics:
                    break
            else:
                score = sum(1 for t in topics if t in persona.knowledge_domains)
                if score > best_score:
                    best_score = score
                    best = persona

        return best

    def is_topic_excluded(self, topic: str, persona: PersonaProfile) -> bool:
        """Check if a topic is excluded by a persona."""
        return topic in persona.excluded_topics
