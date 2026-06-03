"""Canonical high-confidence credential patterns + a scanner.

One source of truth for "does this text contain a credential". Used by:
  - content-aware egress (``animus_types.egress``) — block accidental
    credential leaks to a cloud provider regardless of the declared tier;
  - Core's memory redaction (``animus.memory.redaction``) — redact on ingest.

Keeping the patterns here (zero-dependency package both Core and Forge import)
avoids the two-hand-synced-copies drift that the egress unification removed.
These are credential/key patterns only — NOT broad PII (SSN/email/phone),
which belong to redaction's own set to avoid false-positive egress blocks on
legitimate prose sent to an LLM.
"""

from __future__ import annotations

import re
from re import Pattern

# Red-team-hardened (2026-05-26) credential patterns. Names are stable; callers
# may key on them. Do not add broad PII here — see module docstring.
CREDENTIAL_PATTERNS: dict[str, str] = {
    "anthropic_key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "openai_key": r"sk-(?!ant-)[A-Za-z0-9_\-]{32,}",
    "github_token": r"(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{20,}",
    "github_pat": r"github_pat_[A-Za-z0-9_]{20,}",
    "google_api_key": r"\bAIza[0-9A-Za-z_\-]{35}\b",
    "stripe_key": r"(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,}",
    "slack_token": r"xox[bpoaes]-[A-Za-z0-9\-]{10,}",
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "private_key_block": (
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"
    ),
    "bearer_token": r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}",
    "credential_label_separated": (
        r"(?i)(?:bearer|access|auth|api)[\s_\-]?(?:token|key)s?[\s_\-:=]+[A-Za-z0-9._\-]{6,}"
    ),
    "credential_label_camelcase": (
        r"(?<![A-Za-z])(?:Bearer|Access|Auth|Api)(?:Token|Key)[A-Za-z0-9._\-]{6,}(?![A-Za-z])"
    ),
    "bearer_loose_concat": r"(?i)\bbearer[A-Za-z0-9._\-!@#$%^&*+=:?]{8,}",
}

_COMPILED: list[tuple[str, Pattern[str]]] = [
    (name, re.compile(p)) for name, p in CREDENTIAL_PATTERNS.items()
]


def scan_for_secrets(text: str) -> list[str]:
    """Return the names of any credential patterns found in ``text``.

    Empty list means no credential was detected. Names (not values) are
    returned so a caller can log/deny without echoing the secret.
    """
    if not text:
        return []
    return [name for name, rx in _COMPILED if rx.search(text)]
