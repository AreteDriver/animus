"""Standing red-team capability — periodic adversarial probing.

Builds on the one-shot ``animus.redteam.driver`` to give the hardening
gates a continuous, automated stream of fresh attacks. Designed to run
on a systemd user timer (weekly default) against the locally-hosted
uncensored Qwen so each sweep generates novel probes against the live
defense surface.

The find→fix→regression loop documented in ``docs/THREAT_MODEL.md``
("red-team-on-every-change") stays the same — this module just
keeps the loop running between feature commits.

**Two key signals**

1. **Novel findings** — a probe shape we haven't seen before that
   reached a gate. These are the high-priority items: the model
   generated an attack we didn't anticipate, and either the gate held
   (so we learn what defends against it) or didn't (so we fix
   immediately).
2. **Repeat findings** — a probe shape that matches a prior ledger
   entry by normalized hash. Repeats at HIGH/MEDIUM severity mean we
   *already knew about* this attack class and the fix didn't land or
   regressed. Often more actionable than a novel finding.

**Persistent ledger**

Every probe — OK or not — is appended to a JSONL ledger at
``<data_dir>/audit/standing-redteam-ledger.jsonl``. The ledger never
deletes; we tag each entry with the sweep_id, normalized_hash, and
is_novel flag so historical analysis is possible. Memory append-heavy
by design (matches the project's non-negotiable #3).

**Dashboard**

A human-readable markdown summary at ``<data_dir>/audit/standing-
redteam-dashboard.md``. Each sweep appends a section: timestamp,
totals, novel/repeat counts, top findings. Tail this file to see the
trend at a glance.

Run::

    python -m animus.redteam.standing
    python -m animus.redteam.standing --n-per-category 5
    python -m animus.redteam.standing --alert-on medium

Exit code is non-zero if any finding — novel OR repeat — at or above
``--alert-on`` severity lands, so it pairs cleanly with a systemd
``OnFailure=`` or a cron mail rule. Repeats count too: a known attack
class still reaching a gate is a regression, not noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from animus.redteam.driver import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_RED_TEAM_MODEL,
    RedTeamDriver,
    RedTeamReport,
    Severity,
)

logger = logging.getLogger("redteam.standing")


def _default_audit_dir() -> Path:
    """Match the driver's default audit location for consistency."""
    base = os.environ.get("ANIMUS_DATA_DIR")
    if base:
        return Path(base) / "audit"
    return Path.home() / ".animus" / "audit"


# ---------------------------------------------------------------------------
# Novel-vs-known classification
# ---------------------------------------------------------------------------


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_attack(text: str) -> str:
    """Strip text to lowercase alphanumerics for stable hashing.

    Aim: collapse trivial reformatting (case, whitespace, punctuation
    variations) so two probes that are 'the same attack' produce the
    same hash. The classifier is intentionally conservative — anything
    fancier (Levenshtein, embeddings) would push false-known matches
    and dampen the signal we want.
    """
    return _NORMALIZE_RE.sub("", text.lower())


def attack_hash(text: str) -> str:
    """SHA-256 over the normalized attack text. 16-hex prefix."""
    return hashlib.sha256(normalize_attack(text).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Persistent ledger
# ---------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    """One probe outcome — append-only record in the standing ledger."""

    ts: str
    sweep_id: str
    category: str
    severity: str
    attack_input: str
    observed: str
    normalized_hash: str
    is_novel: bool
    model: str


class FindingsLedger:
    """JSONL-backed append-only ledger of every probe across every sweep."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._known_hashes: set[str] | None = None

    def known_hashes(self) -> set[str]:
        """Lazy-load the set of normalized hashes in the ledger.

        Cached for the lifetime of this instance. If you append entries
        and then need a fresh view, instantiate a new ``FindingsLedger``.
        """
        if self._known_hashes is not None:
            return self._known_hashes
        hashes: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    h = row.get("normalized_hash")
                    if h:
                        hashes.add(h)
                except json.JSONDecodeError:
                    # Don't let a corrupt line block the whole ledger.
                    continue
        self._known_hashes = hashes
        return hashes

    def append(self, entry: LedgerEntry) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        if self._known_hashes is not None:
            self._known_hashes.add(entry.normalized_hash)


# ---------------------------------------------------------------------------
# Standing sweep orchestrator
# ---------------------------------------------------------------------------


@dataclass
class StandingSweepResult:
    """Aggregate result of one standing sweep — written to the dashboard."""

    sweep_id: str
    ts: str
    model: str
    total_probes: int
    novel_findings: int = 0
    repeat_findings: int = 0
    new_high: int = 0
    new_medium: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    findings_detail: list[dict] = field(default_factory=list)


def run_standing_sweep(
    *,
    model: str = DEFAULT_RED_TEAM_MODEL,
    base_url: str | None = None,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    n_per_category: int = 3,
    categories: list[str] | None = None,
    ledger_path: Path | None = None,
    audit_dir: Path | None = None,
    timeout_seconds: float = 120.0,
) -> StandingSweepResult:
    """Execute one sweep, append every probe to the ledger, return summary.

    Does NOT mutate the dashboard — caller is responsible for rendering
    the summary. Keeps the orchestrator pure-ish (single side-effect:
    the ledger append) for testing.
    """
    audit_dir = audit_dir or _default_audit_dir()
    ledger_path = ledger_path or (audit_dir / "standing-redteam-ledger.jsonl")
    ledger = FindingsLedger(ledger_path)
    known = ledger.known_hashes()

    driver = RedTeamDriver(
        model=model,
        ollama_host=ollama_host,
        base_url=base_url,
        n_per_category=n_per_category,
        quick=False,
        timeout_seconds=timeout_seconds,
    )
    report: RedTeamReport = driver.run(categories=categories)

    sweep_id = uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc).isoformat()

    result = StandingSweepResult(
        sweep_id=sweep_id,
        ts=ts,
        model=report.model,
        total_probes=report.total_probes,
        by_severity=dict(report.by_severity),
    )

    for probe in report.probes:
        h = attack_hash(probe.attack_input)
        is_novel = h not in known
        entry = LedgerEntry(
            ts=ts,
            sweep_id=sweep_id,
            category=probe.category,
            severity=probe.severity.value,
            attack_input=probe.attack_input,
            observed=probe.observed,
            normalized_hash=h,
            is_novel=is_novel,
            model=report.model,
        )
        ledger.append(entry)

        # Findings = anything more severe than OK
        if probe.severity in {Severity.HIGH, Severity.MEDIUM}:
            if is_novel:
                result.novel_findings += 1
                if probe.severity == Severity.HIGH:
                    result.new_high += 1
                elif probe.severity == Severity.MEDIUM:
                    result.new_medium += 1
            else:
                result.repeat_findings += 1
            result.findings_detail.append(
                {
                    "category": probe.category,
                    "severity": probe.severity.value,
                    "is_novel": is_novel,
                    "attack_input": probe.attack_input[:200],
                    "observed": probe.observed[:200],
                }
            )

    return result


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------


def render_dashboard_section(result: StandingSweepResult) -> str:
    """Markdown section for one sweep. Appended to the dashboard file.

    Tail-friendly: each sweep is a self-contained block with a
    ``##`` heading and a fixed structure so ``grep -A 20 'sweep_id'``
    extracts a clean record.
    """
    lines: list[str] = []
    lines.append(f"## Sweep {result.sweep_id} — {result.ts}")
    lines.append("")
    lines.append(f"- model: `{result.model}`")
    lines.append(f"- total probes: {result.total_probes}")
    sev = result.by_severity
    lines.append(
        f"- by severity: ok={sev.get('ok', 0)} medium={sev.get('medium', 0)} "
        f"high={sev.get('high', 0)}"
    )
    lines.append(
        f"- **NEW**: {result.new_high} HIGH, {result.new_medium} MEDIUM "
        f"(novel: {result.novel_findings}, repeat: {result.repeat_findings})"
    )
    lines.append("")
    if result.findings_detail:
        lines.append("### Findings")
        lines.append("")
        for f in result.findings_detail:
            marker = "🆕" if f["is_novel"] else "🔁"
            lines.append(
                f"- {marker} [{f['severity'].upper()}] {f['category']}: `{f['attack_input'][:120]}`"
            )
        lines.append("")
    else:
        lines.append("- (no findings at OK threshold)")
        lines.append("")
    return "\n".join(lines) + "\n"


def append_dashboard(result: StandingSweepResult, path: Path) -> None:
    """Append the markdown section to the dashboard file. Creates if absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with path.open("a") as f:
        if header_needed:
            f.write("# Standing Red-Team Dashboard\n\n")
            f.write(
                "Auto-generated by ``python -m animus.redteam.standing``. "
                "Each sweep is appended as a new section. Tail this file to "
                "see the trend; 🆕 = novel finding, 🔁 = repeat (same probe "
                "shape seen before).\n\n"
            )
        f.write(render_dashboard_section(result))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _severity_meets(level: Severity | str, threshold: str) -> bool:
    rank = {"low": 1, "medium": 2, "high": 3}
    lvl = level.value if isinstance(level, Severity) else level
    return rank.get(lvl, 0) >= rank.get(threshold, 3)


def gate_fires(result: StandingSweepResult, alert_on: str) -> bool:
    """Return True if the sweep should exit non-zero.

    Fires on ANY finding — novel OR repeat — at or above ``alert_on``
    severity. A *repeat* HIGH means a known attack class is STILL
    reaching a gate: the fix regressed or never landed. That is a
    regression and must break the build exactly like a novel HIGH. The
    dashboard keeps the novel-vs-repeat distinction for triage; the
    gate deliberately does not.
    """
    return any(_severity_meets(f["severity"], alert_on) for f in result.findings_detail)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Standing red-team sweep — runs the driver, appends to a "
            "persistent ledger, classifies novel vs known."
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANIMUS_REDTEAM_MODEL", DEFAULT_RED_TEAM_MODEL),
        help="Model name (env: ANIMUS_REDTEAM_MODEL).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ANIMUS_REDTEAM_BASE_URL"),
        help="OpenAI-compat backend root URL (env: ANIMUS_REDTEAM_BASE_URL).",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        help="Ollama host fallback when --base-url is unset.",
    )
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=3,
        help="Probes per category (default: 3).",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Filter to one category (repeatable).",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=None,
        help="Override the JSONL ledger location.",
    )
    parser.add_argument(
        "--dashboard-path",
        type=Path,
        default=None,
        help="Override the markdown dashboard location.",
    )
    parser.add_argument(
        "--alert-on",
        choices=["high", "medium", "low"],
        default="high",
        help="Exit non-zero if any finding (novel OR repeat) meets this severity.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Override the audit directory base (default: ~/.animus/audit).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    audit_dir = args.audit_dir or _default_audit_dir()
    ledger_path = args.ledger_path or (audit_dir / "standing-redteam-ledger.jsonl")
    dashboard_path = args.dashboard_path or (audit_dir / "standing-redteam-dashboard.md")

    result = run_standing_sweep(
        model=args.model,
        base_url=args.base_url,
        ollama_host=args.ollama_host,
        n_per_category=args.n_per_category,
        categories=args.category,
        ledger_path=ledger_path,
        audit_dir=audit_dir,
    )
    append_dashboard(result, dashboard_path)

    print(f"\nStanding sweep: {result.sweep_id}")
    print(f"  Model: {result.model}")
    print(f"  Total probes: {result.total_probes}")
    print(f"  By severity: {result.by_severity}")
    print(
        f"  NEW: {result.new_high} HIGH, {result.new_medium} MEDIUM "
        f"(novel={result.novel_findings}, repeat={result.repeat_findings})"
    )
    print(f"  Ledger: {ledger_path}")
    print(f"  Dashboard: {dashboard_path}")

    # Exit non-zero on ANY finding — novel OR repeat — at or above the
    # --alert-on threshold. A repeat HIGH is a known attack class still
    # reaching a gate (the fix regressed or never landed), so it must
    # break the build just like a novel one.
    fired = gate_fires(result, args.alert_on)
    print(f"  Gate (--alert-on {args.alert_on}): {'FAIL (exit 1)' if fired else 'pass'}")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
