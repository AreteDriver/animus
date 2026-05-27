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

REDACTED_FMT = "[REDACTED:{type}]"


@dataclass(frozen=True)
class RedactionHit:
    """Record of a single redaction. Carries no original value."""

    type: str
    start: int
    end: int
    original_length: int


_UNIVERSAL_PATTERNS: dict[str, str] = {
    "anthropic_key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "openai_key": r"sk-(?!ant-)[A-Za-z0-9_\-]{32,}",
    "github_token": r"(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{20,}",
    "github_pat": r"github_pat_[A-Za-z0-9_]{20,}",
    "stripe_key": r"(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,}",
    "slack_token": r"xox[bpoaes]-[A-Za-z0-9\-]{10,}",
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "private_key_block": (
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"
    ),
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "bearer_token": r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}",
    # Red-team finding 2026-05-26: Qwen-generated probe ``BearerTokenForXYZ-
    # Project_2023!@#`` slipped past ``bearer_token`` because that pattern
    # requires whitespace between "bearer" and the content. CamelCase
    # concat (``BearerTokenXxx``) + separator-delimited forms (``access_token=``,
    # ``auth-token: abc``, ``api_key=``) are now caught by the two patterns
    # below. Tight enough to avoid false-positives in prose memory entries.
    "credential_label_separated": (
        r"(?i)(?:bearer|access|auth|api)[\s_\-]?(?:token|key)s?[\s_\-:=]+"
        r"[A-Za-z0-9._\-]{6,}"
    ),
    "credential_label_camelcase": (
        r"(?<![A-Za-z])(?:Bearer|Access|Auth|Api)(?:Token|Key)[A-Za-z0-9._\-]{6,}"
        r"(?![A-Za-z])"
    ),
    # Red-team finding 2026-05-26 (iteration 2): Qwen escalated to leetspeak
    # — ``Bearert0k3nF0rS3cRetUsag3!123``. The CamelCase pattern requires
    # literal ``BearerToken`` chars, and ``t0k3n`` defeats it. This looser
    # pattern matches ``bearer`` followed by 8+ alphanumeric/symbol chars
    # (no whitespace), which catches leetspeak forms while staying tight
    # enough to avoid false-positives on prose (``bearer of bad news`` ends
    # at the space, fails the 8+ requirement).
    "bearer_loose_concat": r"(?i)\bbearer[A-Za-z0-9._\-!@#$%^&*+=:?]{8,}",
    # Red-team finding 2026-05-26 (iteration 3): 30-probe sweep against
    # HauhauCS Qwen3.6 (5 probes/category) found 4 dlp_bypass shapes
    # the previous patterns missed:
    #   1. "Bearer token: T0k3n!_v4lu3_2024#Sec"  — value has special
    #      chars not in the prior ``[A-Za-z0-9._\-]`` class
    #   2. "Bearer: [token: 12345-abcde]"          — bracket separator
    #      between label keyword and value
    #   3. "my_secret_token_is: sk-abc-123-..."    — compound label
    #      ("my_X_token_is") + value too short for openai_key (32 char
    #      minimum)
    #   4. "Bearer: token-pass-12345-67890-abcdef" — colon between
    #      ``bearer`` and ``token``, which the existing
    #      credential_label_separated separator class ``[\s_\-:=]+``
    #      already allowed but the lookahead was constrained to a
    #      single sep char
    # The two patterns below close the cluster. Both require a label
    # keyword + separator + 6+ token-shaped chars, so isolated prose
    # mentions of "secret" or "key" still don't match without a value
    # tail.
    "credential_label_compound": (
        # bearer/access/auth/api … (optional brackets/quotes) … (token|key|
        # secret|cred) … separator … value-with-special-chars
        r"(?i)\b(?:bearer|access|auth|api)\s*[:_\-\s]*[\[\(\"']?\s*"
        r"(?:token|key|secret|cred(?:ential)?)s?"
        r"[\s_\-:=]+[\[\(\"']?\s*"
        r"[A-Za-z0-9._\-!@#$%^&*+=]{6,}"
    ),
    "credential_qualified_label": (
        # ANY snake/camel label containing secret|token|key|password|
        # cred(ential), followed by separator + 6+ token-shaped chars.
        # Catches ``my_secret_token_is:``, ``user_api_key=``,
        # ``the.password.field:`` etc.
        r"(?i)\b[a-z][a-z_]*"
        r"(?:secret|token|key|password|cred(?:ential)?)"
        r"[a-z_]*"
        r"[\s_\-:=]+[\[\(\"']?\s*"
        r"[A-Za-z0-9._\-!@#$%^&*+=]{6,}"
    ),
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
