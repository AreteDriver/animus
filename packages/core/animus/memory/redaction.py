"""Redaction for memory ingest and MCP egress.

Pure function module. No I/O. Returns `(redacted_text, list[RedactionHit])`.

Two pattern tiers:

1. **Universal**: API tokens (OpenAI/Anthropic/GitHub/AWS/Slack/Stripe), private
   key blocks, SSN, generic bearer tokens. Always-on, low false-positive rate.
2. **Personal**: Known emails, phone numbers, sensitive file paths. Driven by
   defaults reflecting ARETE PII, override via env vars
   (``ANIMUS_REDACT_EMAILS``, ``ANIMUS_REDACT_PHONES``, ``ANIMUS_REDACT_PATHS`` —
   comma-separated). Toggle off per-call with ``include_personal=False``.

``RedactionHit`` never carries the original value — only type, span, and length.
Logging a hit cannot re-leak the secret.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from re import Pattern

from animus_types.secrets import CREDENTIAL_PATTERNS

REDACTED_FMT = "[REDACTED:{type}]"


@dataclass(frozen=True)
class RedactionHit:
    """Record of a single redaction. Carries no original value."""

    type: str
    start: int
    end: int
    original_length: int


# Credential/key patterns are canonical in ``animus_types.secrets`` so the
# content-aware egress DLP and this redaction share ONE source (the same
# no-drift discipline that unified the egress policy). SSN is redaction-only
# PII — it stays here, not in the egress credential set, to avoid
# false-positive egress blocks on legitimate prose.
_UNIVERSAL_PATTERNS: dict[str, str] = {
    **CREDENTIAL_PATTERNS,
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

_UNIVERSAL_COMPILED: list[tuple[str, Pattern[str]]] = [
    (name, re.compile(pattern)) for name, pattern in _UNIVERSAL_PATTERNS.items()
]


DEFAULT_PERSONAL_EMAILS: tuple[str, ...] = (
    "aretedriver@gmail.com",
    "jamesyng79@gmail.com",
)

DEFAULT_PERSONAL_PHONES: tuple[str, ...] = (
    "503-449-8300",
    "5034498300",
    "(503) 449-8300",
    "503.449.8300",
)

DEFAULT_SENSITIVE_PATHS: tuple[str, ...] = (
    "/home/arete/Documents/WORK",
    "/home/arete/Documents/TIAID",
    "/home/arete/.local/share/animus/secrets.env",
    "/home/arete/.config/animus/secrets.env",
)


def _split_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or default


def _build_personal_patterns() -> list[tuple[str, Pattern[str]]]:
    emails = _split_env("ANIMUS_REDACT_EMAILS", DEFAULT_PERSONAL_EMAILS)
    phones = _split_env("ANIMUS_REDACT_PHONES", DEFAULT_PERSONAL_PHONES)
    paths = _split_env("ANIMUS_REDACT_PATHS", DEFAULT_SENSITIVE_PATHS)

    patterns: list[tuple[str, Pattern[str]]] = []
    for email in emails:
        patterns.append(("personal_email", re.compile(re.escape(email), re.IGNORECASE)))
    for phone in phones:
        patterns.append(("personal_phone", re.compile(re.escape(phone))))
    for path in paths:
        patterns.append(("sensitive_path", re.compile(re.escape(path) + r"(?:/[^\s\"']*)?")))
    return patterns


def redact(
    text: str,
    *,
    include_personal: bool = True,
) -> tuple[str, list[RedactionHit]]:
    """Redact secrets and PII from text.

    Args:
        text: Input text to scan.
        include_personal: When True (default), also apply personal-PII patterns.
            Set False for tests or contexts where universal-only is desired.

    Returns:
        Tuple of ``(redacted_text, list of RedactionHit records)`` in original
        left-to-right order. Spans in the hits refer to positions in the
        *original* text, not the redacted output.
    """
    if not text:
        return text, []

    patterns = list(_UNIVERSAL_COMPILED)
    if include_personal:
        patterns.extend(_build_personal_patterns())

    raw_spans: list[tuple[int, int, str]] = []
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            if match.start() == match.end():
                continue
            raw_spans.append((match.start(), match.end(), name))

    if not raw_spans:
        return text, []

    raw_spans.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, name in raw_spans:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_name = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_name)
        else:
            merged.append((start, end, name))

    hits = [
        RedactionHit(type=name, start=start, end=end, original_length=end - start)
        for start, end, name in merged
    ]

    out_parts: list[str] = []
    cursor = 0
    for start, end, name in merged:
        out_parts.append(text[cursor:start])
        out_parts.append(REDACTED_FMT.format(type=name))
        cursor = end
    out_parts.append(text[cursor:])

    return "".join(out_parts), hits


def has_secrets(text: str, *, include_personal: bool = True) -> bool:
    """Cheap boolean check — True if any pattern matches."""
    _, hits = redact(text, include_personal=include_personal)
    return bool(hits)
