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

import math
import re
from collections import Counter
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
    # Red-team sweep (HauhauCS Qwen3.6, 2026-05-26 iter 3, via #62) found 4 more
    # dlp_bypass shapes: special-char value tails, bracket separators, compound
    # labels (``my_secret_token_is:``), and colon-between-keyword forms. These
    # two close the cluster — both require a label keyword + separator + 6+
    # token-shaped chars, so bare prose ("secret"/"key") still won't match.
    "credential_label_compound": (
        r"(?i)\b(?:bearer|access|auth|api)\s*[:_\-\s]*[\[\(\"']?\s*"
        r"(?:token|key|secret|cred(?:ential)?)s?"
        r"[\s_\-:=]+[\[\(\"']?\s*"
        r"[A-Za-z0-9._\-!@#$%^&*+=]{6,}"
    ),
    "credential_qualified_label": (
        r"(?i)\b[a-z][a-z_]*"
        r"(?:secret|token|key|password|cred(?:ential)?)"
        r"[a-z_]*"
        r"[\s_\-:=]+[\[\(\"']?\s*"
        r"[A-Za-z0-9._\-!@#$%^&*+=]{6,}"
    ),
}

_COMPILED: list[tuple[str, Pattern[str]]] = [
    (name, re.compile(p)) for name, p in CREDENTIAL_PATTERNS.items()
]


# C1-6 — prefixless-secret heuristic. Every pattern above is prefix-anchored
# (sk-ant-, AKIA, AIza, ghp_…), so a high-entropy key with NO recognizable
# prefix (a raw 40-char API token, a base64 secret) slips through. The entropy
# detector below catches those. The bar is deliberately conservative to avoid
# false-positive egress blocks on legitimate content:
#   - length >= 32 (short tokens are too ambiguous)
#   - Shannon entropy >= 4.0 bits/char (excludes repetitive / low-variety runs)
#   - >= 3 character classes present (lower AND upper AND digit)
# The 3-class rule exempts the common benign high-entropy tokens that would
# otherwise false-positive: hex SHAs / git hashes (lower+digit only), UUIDs
# (hyphen-split, lower+digit), all-upper or all-lower IDs.
# RESIDUAL LIMIT: a base64 blob of binary (mixed case+digit) IS flagged — that
# is intentional (treat an unexplained high-entropy blob bound for a cloud LLM
# as possible exfil), but callers sending legitimate base64 should tag the
# request's sensitivity rather than rely on PUBLIC + content scan.
_HE_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{32,}")
_HE_MIN_ENTROPY = 4.0


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def find_high_entropy_tokens(text: str) -> list[str]:
    """Return tokens that look like prefixless secrets (see bar above)."""
    hits: list[str] = []
    for tok in _HE_TOKEN.findall(text):
        if _shannon_entropy(tok) < _HE_MIN_ENTROPY:
            continue
        classes = (
            any(c.islower() for c in tok)
            + any(c.isupper() for c in tok)
            + any(c.isdigit() for c in tok)
        )
        if classes >= 3:
            hits.append(tok)
    return hits


def scan_for_secrets(text: str, *, include_high_entropy: bool = True) -> list[str]:
    """Return the names of any credential patterns found in ``text``.

    Empty list means no credential was detected. Names (not values) are
    returned so a caller can log/deny without echoing the secret.

    ``include_high_entropy`` (C1-6, default on) also flags prefixless
    high-entropy tokens as ``"high_entropy_token"`` — set False for callers
    that legitimately carry high-entropy data and gate on the sensitivity tier
    instead.
    """
    if not text:
        return []
    names = [name for name, rx in _COMPILED if rx.search(text)]
    if include_high_entropy and find_high_entropy_tokens(text):
        names.append("high_entropy_token")
    return names
