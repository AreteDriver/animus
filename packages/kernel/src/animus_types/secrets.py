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
    "password_label": (
        r"(?i)\b(?:password|passwd|pwd)\s*[:=]+[\[\(\"']?\s*"
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


# ---------------------------------------------------------------------------
# Redaction helpers (SEC-06 — secret-safe logging and telemetry)
# ---------------------------------------------------------------------------

REDACTED_FMT = "[REDACTED:{name}]"


class _NoSecretMatch(Exception):
    """Internal sentinel used by ``redact`` to skip zero-width matches."""


def redact(text: str, *, include_high_entropy: bool = True) -> str:
    """Return a redacted copy of ``text``.

    Every credential pattern match and (by default) every prefixless
    high-entropy token is replaced with ``[REDACTED:<pattern_name>]``.  The
    placeholder contains no secret material and is safe to log or return.

    Correlation IDs, UUIDs, hex SHAs, and other low-entropy tokens are left
    untouched because the high-entropy bar requires three character classes.
    """
    if not text:
        return text

    # Collect spans in a single pass over all patterns.  Merge overlapping
    # spans so the longest match wins and placeholders never overlap.
    raw_spans: list[tuple[int, int, str]] = []
    for name, pattern in _COMPILED:
        for match in pattern.finditer(text):
            if match.start() == match.end():
                continue
            raw_spans.append((match.start(), match.end(), name))

    if include_high_entropy:
        for tok in find_high_entropy_tokens(text):
            idx = text.find(tok)
            while idx != -1:
                raw_spans.append((idx, idx + len(tok), "high_entropy_token"))
                idx = text.find(tok, idx + 1)

    if not raw_spans:
        return text

    raw_spans.sort()
    merged: list[tuple[int, int, str]] = []
    for start, end, name in raw_spans:
        if merged and start < merged[-1][1]:
            prev_start, prev_end, prev_name = merged[-1]
            # Prefer the earlier/larger span for determinism; type names are
            # stable, but overlapping spans are rare in practice.
            merged[-1] = (prev_start, max(prev_end, end), prev_name)
        else:
            merged.append((start, end, name))

    out_parts: list[str] = []
    cursor = 0
    for start, end, name in merged:
        out_parts.append(text[cursor:start])
        out_parts.append(REDACTED_FMT.format(name=name))
        cursor = end
    out_parts.append(text[cursor:])
    return "".join(out_parts)


def redact_exception(exc: Exception | None, *, include_high_entropy: bool = True) -> str:
    """Return a redacted string for an exception, never the raw args.

    This is the safe way to log an exception that may carry request details,
    headers, or payload snippets: the type name and redacted message are
    preserved, but credential-like substrings are masked.
    """
    if exc is None:
        return ""
    message = " ".join(str(a) for a in exc.args) if exc.args else type(exc).__name__
    return redact(message, include_high_entropy=include_high_entropy)


def _split_env_arg(arg: str) -> tuple[str, str] | None:
    """Parse ``-e KEY=VALUE`` or ``--env=KEY=VALUE`` and return ``(key, value)``.

    Returns ``None`` for flags without an inline value.
    """
    if arg.startswith("--env="):
        rest = arg[len("--env="):]
        if "=" in rest:
            key, value = rest.split("=", 1)
            return key, value
        return rest, ""
    if arg.startswith("-e") and not arg.startswith("-e="):
        rest = arg[2:]
        if not rest:
            return None
        if "=" in rest:
            key, value = rest.split("=", 1)
            return key, value
        return rest, ""
    return None


def mask_env_command_args(cmd: list[str]) -> list[str]:
    """Return a copy of a container/shell command with ``-e`` values masked.

    Every ``-e KEY=VALUE`` / ``--env=KEY=VALUE`` argument has its value replaced
    by ``[REDACTED]``.  Flags like ``-e KEY`` (value on next argv) are also
    handled: the next argv is replaced with ``KEY=[REDACTED]``.

    This prevents container command logging from emitting environment values
    while keeping the command structure useful for debugging.
    """
    out: list[str] = []
    skip_next_value = False
    for arg in cmd:
        if skip_next_value:
            # Previous arg was a bare -e/--env flag; this argv is the value.
            if "=" in arg:
                key, _ = arg.split("=", 1)
                out.append(f"{key}=[REDACTED]")
            else:
                out.append(f"{arg}=[REDACTED]")
            skip_next_value = False
            continue

        if arg in ("-e", "--env"):
            out.append(arg)
            skip_next_value = True
            continue

        parsed = _split_env_arg(arg)
        if parsed is not None:
            key, _ = parsed
            if arg.startswith("-e"):
                out.append(f"-e={key}=[REDACTED]")
            else:
                out.append(f"--env={key}=[REDACTED]")
            continue

        out.append(arg)

    return out
