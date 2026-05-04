# Detection Framework Patterns — Extracted from Monolith

**Source**: `monolith` repo, archived 2026-04-25
**Source state at extraction**: v0.5.0, 728 tests, 84% coverage, 18 sync checkers + 1 async, deployed Fly.io
**Extraction reason**: Monolith was a Frontier-specific tool. The detection rules, the galaxy map, and the chain-poll loops do not generalize. The patterns below do — and they belong in arete-evals (the eval layer pattern especially) and in any future drift-monitor work that needs typed anomaly events.

This doc captures the architecture that survives the cull, with enough code to reconstruct it elsewhere without re-reading the monolith repo. Where useful, it cites the monolith source path so the original implementation can be retrieved from git history.

---

## 1. Anomaly schema with provenance chain

**The pattern.** Detection events are typed dataclasses, not dicts. Every detection carries a `provenance` array — an ordered list of source records that justify the detection. Each provenance entry has four fields: `source_type`, `source_id`, `timestamp`, `derivation`. That's the "who/what/when/why" of every anomaly, machine-readable.

**Why it matters.** Two reasons:

1. **Audit trail.** When a stakeholder asks "why did we flag this?", the answer is a JSON array, not a chase through logs. The same array is what a reviewer needs to confirm or dismiss the finding.
2. **Eval ground-truth alignment.** Provenance entries are stable identifiers. If you replay a checker against historical data, you can compare provenance arrays for equivalence — much more reliable than comparing free-text descriptions.

**Canonical shape** (from `monolith/backend/detection/base.py`):

```python
from dataclasses import dataclass, field

@dataclass
class ProvenanceEntry:
    source_type: str   # event | snapshot | rpc_query | rule | upstream_alert
    source_id: str     # event_id, snapshot_id, tx_hash, rule reference
    timestamp: int     # when the source data was produced
    derivation: str    # human-readable: how this source contributed

@dataclass
class Anomaly:
    anomaly_type: str
    rule_id: str
    detector: str
    object_id: str
    evidence: dict = field(default_factory=dict)
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    severity: str = ""    # CRITICAL | HIGH | MEDIUM | LOW
    category: str = ""    # set by classify_anomaly(rule_id) if blank
    anomaly_id: str = ""  # generated: PREFIX-YYYYMMDD-NNNN
    detected_at: int = 0
```

**Generalizing.** The schema is domain-agnostic. Replace `object_id` with whatever the unit-of-detection is (request_id, model_run_id, customer_id). Replace `system_id` (omitted above) with whatever scoping dimension matters.

**Where this goes.** Use as the standard event shape for any detector authored under arete-evals. drift-monitor's drift events should adopt the same `provenance: list[ProvenanceEntry]` field — currently drift events have a flat `metadata` dict, which loses the ordered audit chain.

---

## 2. Deterministic rubric for LLM output evaluation

**The pattern.** When you generate text with an LLM (narration, summaries, recommendations), evaluate it with a rule-based scorer, not another LLM. The scorer reads the LLM's output and the underlying evidence, then emits a multi-dimension score.

**Why it matters.** LLM-grades-LLM has three problems: it's expensive, it's noisy (judge-judge disagreement), and it conceals failure modes (a judge that doesn't know what hallucination looks like will miss it). A deterministic rubric is cheap, deterministic across runs, and **catches the failure modes you specifically care about** because you encoded them in the rubric.

**Canonical four-dimension shape** (from `monolith/eval/narration_eval.py`):

| Dimension | What it measures | How it's scored |
|---|---|---|
| `factual_grounding` | Does narration reference evidence IDs/values that actually appear in the source? | Extract identifiable tokens from evidence (long alphanumerics, tx hashes), count how many appear in narration, normalize. |
| `severity_alignment` | Does narration's language match the declared severity? | Vocabulary lookup. CRITICAL severity expects "urgent/severe/immediate"; LOW expects "routine/minor". Opposing vocabulary is a strong negative. |
| `actionability` | Does narration tell someone what to do? | Count action verbs ("investigate", "review", "escalate", "verify"). Cap at 1.0. |
| `hallucination_flag` | Did the LLM invent specific claims? | Extract 4+ digit numbers and tx hashes from narration. Anything not present in the evidence block is a fabrication. |

The composite score averages the first three and applies a 0.5× penalty if `hallucination_flag` fires. Pass threshold: composite ≥ 0.70 AND hallucination_rate ≤ 0.10 across the run.

**The hallucination check is the load-bearing piece.** It's the most reliable signal because it's a string-membership test, not a judgment call. Generalize it by deciding what kinds of "specifics" matter for your domain (numbers, hashes, citations, function names, file paths) and checking each against the evidence block.

**Generalizing.** Swap the vocabulary lists for your domain. Frontier-flavored words ("compromised", "monitor") become whatever your operators say. The dimensions stay: grounding, alignment-with-declared-state, actionability, hallucination. These four cover most narration-evaluation needs.

**Where this goes.** This is the headline pattern for arete-evals. The animus-forge `briefing-quality` rubric already encodes a related vocabulary; this extends it with the hallucination-flag mechanic, which forge does not currently implement.

---

## 3. DB-backed dedup that survives process restarts

**The pattern.** Don't dedupe by in-memory set or LRU cache. Persist the dedup state in the same database that holds the events. When the process restarts, the dedup state is intact.

**Why it matters.** Most anomaly systems lose dedup state on restart and re-fire alerts that fired ten minutes earlier. This burns operator trust faster than false positives do, because the operator can see that the system has no memory.

**Canonical shape** (from `monolith/backend/alerts/github_issues.py` and the `(anomaly_type, object_id)` window in `engine.py`):

- Dedup key: a tuple of (event_type, target_id) plus a time window.
- Dedup table: `filed_issues (anomaly_id, github_issue_url, filed_at)`.
- Check: `SELECT 1 FROM filed_issues f JOIN anomalies a ON f.anomaly_id = a.anomaly_id WHERE a.anomaly_type = ? AND a.object_id = ? AND a.detected_at > ?`. If any row matches, suppress.
- Window: 24 hours in monolith. Tunable.

**Generalizing.** Any alerting system. The key insight is: **the dedup table joins to the events table**, so dedup logic and event lifecycle stay in sync. If an event is purged, its dedup record purges with it via foreign key cascade.

---

## 4. Webhook subscription dispatch with severity/type filters

**The pattern.** Operators register webhooks with two filter dimensions: a severity allowlist and an event-type allowlist. Empty list = match everything. Dispatcher iterates active subscriptions, filters per-anomaly, sends in parallel via async HTTP client.

**Why it matters.** Most webhook systems either fire-everything (operator drowns in noise) or require complex routing rules upfront (operator never configures it). Two allowlists is the right amount of expressiveness for >80% of operator needs.

**Canonical shape** (from `monolith/backend/api/subscriptions.py` and `monolith/backend/alerts/subscription_dispatch.py`):

```python
# Schema
CREATE TABLE subscriptions (
    sub_id TEXT UNIQUE NOT NULL,
    webhook_url TEXT NOT NULL,
    severity_filter TEXT NOT NULL DEFAULT '[]',  -- JSON array
    event_types TEXT NOT NULL DEFAULT '[]',      -- JSON array
    created_at INTEGER NOT NULL,
    active INTEGER DEFAULT 1
);

# Filter logic — the only logic that matters
def matches(anomaly, severity_filter, event_types):
    if severity_filter and anomaly["severity"] not in severity_filter:
        return False
    if event_types and anomaly["anomaly_type"] not in event_types:
        return False
    return True

# Dispatch — async, per-subscription error isolation
async with httpx.AsyncClient(timeout=10.0) as client:
    for sub in active_subs:
        if not matches(anomaly, sub.severity_filter, sub.event_types):
            continue
        try:
            await client.post(sub.webhook_url, json=embed_payload)
        except Exception:
            logger.exception("dispatch failed for %s", sub.sub_id)
```

**What's missing in the monolith implementation** (and what arete-evals should add): HMAC-SHA256 signature on the payload using a per-subscription secret. Monolith referenced this in CLAUDE.md but the dispatch code above doesn't sign. The pattern is right; the security hardening was incomplete. If you adopt this, sign the payload.

**Generalizing.** Direct transplant into any FastAPI service. Replace `_build_embed()` (Discord-specific) with whatever payload shape your consumers expect.

---

## 5. Detection cycle telemetry as a first-class table

**The pattern.** Every detection cycle writes its own row to a `detection_cycles` table: started_at, finished_at, anomalies_found, errors. This is operational data, not application data, but it lives in the same database.

**Why it matters.** Two things become easy:
1. **P50/P95 cycle latency.** Querying `(finished_at - started_at)` percentiles tells you whether the system is keeping up with input rate.
2. **Drift detection on detector behavior.** If your CRITICAL-anomaly rate suddenly triples, that's either a real surge or a regression in the rules. The `anomalies_found` column lets you bisect against deploys.

**Generalizing.** Any pipeline with a recurrent cycle (cron job, scheduled detector, batch processor). Record the cycle as a row, query the row stream for ops dashboards.

---

## What did NOT generalize

For honesty: most of monolith doesn't port. The 36 detection rules are specific to Sui chain event shapes, EVE Frontier object types, and tribe-membership semantics. The 3D galaxy map renders 24K Frontier solar systems via Three.js. The Warden polls Sui RPC to verify anomalies. None of these reach beyond Frontier without rewriting against a different event shape.

The five patterns above are the part that does port. The line between portable and Frontier-specific is roughly: **architecture and schemas port; rules and rendering don't**.

---

## Provenance

| Pattern | Source path in monolith | Notes |
|---|---|---|
| Anomaly schema | `backend/detection/base.py` | `ProvenanceEntry` + `Anomaly` dataclasses |
| Deterministic rubric | `eval/narration_eval.py` | 471 lines, 4 dimensions + composite |
| DB-backed dedup | `backend/alerts/github_issues.py` + `backend/detection/engine.py` | Two-table JOIN pattern |
| Webhook subscriptions | `backend/api/subscriptions.py` + `backend/alerts/subscription_dispatch.py` | 144 + 150 lines |
| Cycle telemetry | `backend/db/database.py` (`detection_cycles` table) | Schema + cycle-write hook in engine |

Repo will be archived 2026-04-25. Git history preserved for reference.
