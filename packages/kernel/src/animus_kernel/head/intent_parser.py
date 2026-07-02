"""Intent parser for Animus Head — classifies user input into actionable intents.

Lightweight rule-based classifier (no model call). Maps natural language
patterns to structured intents so the REPL can decide whether to:
- Route directly to a known tool plan
- Ask the model to plan
- Ask the user for clarification
- Just chat
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    """Classification of user intent."""

    DIRECT_COMMAND = "direct_command"  # User explicitly names a tool/action
    VAGUE_REQUEST = "vague_request"  # Describes goal, no specific tool named
    CLARIFICATION_NEEDED = "clarification_needed"  # Ambiguous / multiple matches
    CONVERSATIONAL = "conversational"  # Chat, greeting, meta


@dataclass
class ParsedIntent:
    """Structured representation of parsed user intent."""

    intent_type: IntentType
    confidence: float  # 0.0 - 1.0
    suggested_tools: list[str] = field(default_factory=list)
    extracted_args: dict = field(default_factory=dict)
    reason: str = ""


class HeadIntentParser:
    """Rule-based intent parser.

    Uses keyword patterns and regex heuristics to classify input.
    No model call required — deterministic and fast.
    """

    # Patterns: (regex, intent_type, suggested_tools, arg_extractors)
    PATTERNS: list[tuple] = [
        # Direct commands — tool names explicitly mentioned
        (
            r"\bread\s+(?:file\s+)?['\"]?(.+?)['\"]?\s*$",
            IntentType.DIRECT_COMMAND,
            ["read_file"],
            {"path": 1},
        ),
        (
            r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:content\s+of\s+)?['\"]?(.+?)['\"]?\s*$",
            IntentType.DIRECT_COMMAND,
            ["read_file"],
            {"path": 1},
        ),
        (
            r"\b(cat|display|print)\s+['\"]?(.+?)['\"]?\s*$",
            IntentType.DIRECT_COMMAND,
            ["read_file"],
            {"path": 2},
        ),
        (
            r"\bwrite\s+(?:to\s+)?['\"]?(.+?)['\"]?\s*:?\s*(.+)$",
            IntentType.DIRECT_COMMAND,
            ["write_file"],
            {"path": 1, "content": 2},
        ),
        (
            r"\bcreate\s+(?:file\s+)?['\"]?(.+?)['\"]?\s*(?:with\s+)?(.+)?$",
            IntentType.DIRECT_COMMAND,
            ["write_file"],
            {"path": 1, "content": 2},
        ),
        (
            r"\bedit\s+['\"]?(.+?)['\"]?\s*(?:replace|change)\s+(.+?)\s+to\s+(.+)",
            IntentType.DIRECT_COMMAND,
            ["edit_file"],
            {"path": 1, "old_string": 2, "new_string": 3},
        ),
        (
            r"\brun\s+(?:test|tests)\b",
            IntentType.DIRECT_COMMAND,
            ["run_shell"],
            {"command": "pytest"},
        ),
        (
            r"\btest\s+(?:this|the|my)\s+(?:code|project|repo)",
            IntentType.DIRECT_COMMAND,
            ["run_shell"],
            {"command": "pytest"},
        ),
        (
            r"\b(git\s+status|status|what\s+changed)\b",
            IntentType.DIRECT_COMMAND,
            ["run_shell"],
            {"command": "git status"},
        ),
        (
            r"\b(git\s+log|show\s+commits|recent\s+commits)\b",
            IntentType.DIRECT_COMMAND,
            ["run_shell"],
            {"command": "git log --oneline -10"},
        ),
        (
            r"\b(search|find|grep)\s+(?:for\s+)?['\"]?(.+?)['\"]?\s*(?:in\s+(.+))?$",
            IntentType.DIRECT_COMMAND,
            ["search_code"],
            {"query": 2, "path": 3},
        ),
        (
            r"\blist\s+(?:all\s+)?(?:files?|directory|dir)\s*(?:in\s+)?['\"]?(.+?)['\"]?\s*$",
            IntentType.DIRECT_COMMAND,
            ["list_files"],
            {"path": 1},
        ),
        (
            r"\bls\s+['\"]?(.+?)['\"]?\s*$",
            IntentType.DIRECT_COMMAND,
            ["list_files"],
            {"path": 1},
        ),
        (
            r"\b(remember|save|store|note)\s+(?:that\s+)?(.+)$",
            IntentType.DIRECT_COMMAND,
            ["remember"],
            {"content": 2},
        ),
        (
            r"\b(recall|search\s+memory|find\s+memory|what\s+do\s+you\s+know)\s+(?:about\s+)?(.+)?$",
            IntentType.DIRECT_COMMAND,
            ["recall"],
            {"query": 2},
        ),
        (
            r"\b(add|create)\s+(?:a\s+)?task\s*(?:to\s+)?(.+)$",
            IntentType.DIRECT_COMMAND,
            ["create_task"],
            {"description": 2},
        ),
        (
            r"\b(list|show)\s+(?:my\s+)?tasks\b",
            IntentType.DIRECT_COMMAND,
            ["list_tasks"],
            {},
        ),

        # Vague requests — goal described but no specific tool
        (
            r"\b(check|audit|review|scan)\s+(?:for\s+)?(?:issues|problems|errors|bugs|lint)\b",
            IntentType.VAGUE_REQUEST,
            ["run_linter", "run_tests", "search_code"],
            {},
        ),
        (
            r"\b(what\s+is|explain|describe|tell\s+me\s+about)\s+(.+)",
            IntentType.VAGUE_REQUEST,
            ["read_file", "search_code", "recall"],
            {"query": 2},
        ),
        (
            r"\b(how\s+do\s+i|how\s+to|how\s+can\s+i)\b",
            IntentType.VAGUE_REQUEST,
            ["read_file", "search_code", "recall"],
            {},
        ),
        (
            r"\b(fix|debug|solve|resolve)\s+(?:the\s+)?(?:issue|error|problem|bug)\b",
            IntentType.VAGUE_REQUEST,
            ["search_code", "read_file", "run_tests"],
            {},
        ),
        (
            r"\b(setup|install|configure|init)\b",
            IntentType.VAGUE_REQUEST,
            ["run_shell", "read_file"],
            {},
        ),
        (
            r"\b(summary|summarize|overview|project\s+status)\b",
            IntentType.VAGUE_REQUEST,
            ["project_structure", "list_files", "read_file"],
            {},
        ),

        # Conversational / meta
        (
            r"^(hi|hello|hey|greetings|howdy)\b",
            IntentType.CONVERSATIONAL,
            [],
            {},
        ),
        (
            r"\b(thanks|thank\s+you|appreciate|grateful)\b",
            IntentType.CONVERSATIONAL,
            [],
            {},
        ),
        (
            r"\b(bye|goodbye|see\s+ya|cya|later)\b",
            IntentType.CONVERSATIONAL,
            [],
            {},
        ),
        (
            r"\b(who\s+are\s+you|what\s+can\s+you\s+do|your\s+capabilities)\b",
            IntentType.CONVERSATIONAL,
            [],
            {},
        ),
    ]

    def parse(self, user_input: str) -> ParsedIntent:
        """Parse user input into a structured intent.

        Returns:
            ParsedIntent with type, confidence, suggested tools, and args.
        """
        text = user_input.strip().lower()
        if not text:
            return ParsedIntent(
                intent_type=IntentType.CONVERSATIONAL,
                confidence=1.0,
                reason="empty input",
            )

        matches = []
        for pattern, intent_type, tools, arg_map in self.PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                extracted = {}
                for arg_name, group_idx in arg_map.items():
                    if isinstance(group_idx, str):
                        # Literal string value (e.g., {"command": "git status"})
                        extracted[arg_name] = group_idx
                    elif isinstance(group_idx, int):
                        if group_idx - 1 < len(groups) and groups[group_idx - 1]:
                            extracted[arg_name] = groups[group_idx - 1].strip()

                # Confidence based on match specificity
                confidence = self._compute_confidence(text, pattern, intent_type, extracted)
                matches.append(
                    ParsedIntent(
                        intent_type=intent_type,
                        confidence=confidence,
                        suggested_tools=tools,
                        extracted_args=extracted,
                        reason=f"matched pattern: {pattern[:40]}...",
                    )
                )

        if not matches:
            # Default to vague request — let the model figure it out
            return ParsedIntent(
                intent_type=IntentType.VAGUE_REQUEST,
                confidence=0.3,
                suggested_tools=[],
                reason="no pattern matched",
            )

        # Sort by confidence descending
        matches.sort(key=lambda x: x.confidence, reverse=True)
        best = matches[0]

        # If top match is vague but there are direct commands too, prefer direct
        direct_matches = [m for m in matches if m.intent_type == IntentType.DIRECT_COMMAND]
        if direct_matches:
            best = direct_matches[0]

        # If multiple patterns matched with similar confidence, might need clarification
        if len(matches) > 1 and matches[0].confidence - matches[1].confidence < 0.15:
            return ParsedIntent(
                intent_type=IntentType.CLARIFICATION_NEEDED,
                confidence=matches[0].confidence,
                suggested_tools=list({t for m in matches[:2] for t in m.suggested_tools}),
                reason="multiple ambiguous patterns matched",
            )

        return best

    @staticmethod
    def _compute_confidence(text: str, pattern: str, intent_type: IntentType, extracted: dict) -> float:
        """Compute confidence score for a match."""
        base = 0.7

        # Direct commands get higher confidence
        if intent_type == IntentType.DIRECT_COMMAND:
            base += 0.15

        # Longer regex patterns (more specific) get higher confidence
        pattern_len = len(pattern)
        if pattern_len > 40:
            base += 0.05

        # Extracted arguments increase confidence
        if extracted:
            base += 0.05 * min(len(extracted), 2)

        return min(0.99, base)
