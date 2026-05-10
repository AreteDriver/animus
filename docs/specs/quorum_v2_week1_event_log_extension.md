# Spec: Quorum v2 Week 1 — EventLog Bitemporal Extension + Signal Bus Bridge

**Status:** Ready to build
**Mode:** /specification
**Date:** 2026-05-10
**Effort:** 2 days (revised from roadmap's 5 days — most infrastructure already exists)
**Roadmap:** `docs/ROADMAP_quorum_v2.md` Week 1
**Supersedes:** Roadmap's "TickEvent append-only log" framing — `EventLog` already exists; this is an extension.

---

## 0. Premise (do not re-argue)

`convergent.event_log.EventLog` exists and is mature:
- Append-only SQLite, WAL mode, 4 indexes
- 10 event types covering all coordination mutations
- `record()` / `query()` / `count()` API
- Zero production deps (stdlib only)

What is missing:
1. Bitemporal fields (`valid_from`, `recorded_at`) — needed to distinguish *when a fact became true* from *when we recorded it*
2. Bridge to `signal_bus` — `record()` writes to SQLite but does not publish a `Signal`, so subscribers cannot react in real-time
3. Three mutation sites do not currently emit events (gap audit below)

This spec closes those three gaps. No new ontology, no new event types.

---

## 1. Goal

Complete `EventLog`'s coverage of Quorum coordination mutations and make the stream live-subscribable, while adding bitemporal-lite fields ported from memboot.

---

## 2. Out of scope

| Item | Reason |
|---|---|
| New event types beyond the existing 10 | Existing types cover all 5 mutation sites |
| Replay engine | Week 1 is the log; replay deferred until demand-side justification |
| Cross-machine event sync | Single-machine Animus only |
| Schema versioning beyond a single `schema_version` int | YAGNI for v1 |
| Migrating existing rows to populate `valid_from` / `recorded_at` | Defaults handle backfill (see §6.4) |
| Removing the existing `timestamp` field | Backwards compatibility — `timestamp` becomes alias for `recorded_at` |
| Constitutional signing of events (P5) | Confirmed not required; only Intent creation is signed (per `packages/quorum/CLAUDE.md` Protocol Invariants) |

---

## 3. Current-state baseline

### 3.1 EventLog (already mature)

`packages/quorum/python/convergent/event_log.py`:
- `CoordinationEvent` dataclass (line 54-72): `event_id`, `event_type`, `agent_id`, `timestamp`, `payload`, `correlation_id`
- `EventLog.record(...)` (line 109): writes to SQLite, returns event
- `EventLog.query(...)` (line 160): filter by type/agent/correlation/time-range, ordered ASC
- `EventLog.count(...)` (line 215): cardinality
- `event_timeline(events)` (line 249): pretty-printer

### 3.2 SignalBus (already mature)

`packages/quorum/python/convergent/signal_bus.py`:
- `SignalBus.publish(signal: Signal) -> None` (line 92)
- `SignalBus.subscribe(signal_type, callback, agent_id=None)` (line 106)
- `Signal` envelope: `signal_type`, `source_agent`, `target_agent`, `payload`, `timestamp`
- Backends: `FilesystemSignalBackend` (default), `SQLiteSignalBackend` (cross-process ACID)

### 3.3 The 5 mutation sites — current emission status

| # | Mutation | File:Line | EventType | Currently calls `record()`? |
|---|---|---|---|---|
| 1 | Intent publish | `resolver.py:49` `PythonGraphBackend.publish(intent)` | `INTENT_PUBLISHED` | **No** — only DEBUG logs |
| 2 | Stability update | `intent.py:169` `Intent.add_evidence(evidence)` | `SCORE_UPDATED` | **No** — no logging |
| 3 | Vote commit | `triumvirate.py:88` `Triumvirate.submit_vote(request_id, vote)` | `VOTE_CAST` | **No** — only INFO logs |
| 4 | Stigmergy mark | `stigmergy.py:69` `StigmergyField.leave_marker(agent_id, marker_type, target, content, strength=1.0, expires_at=None)` | `MARKER_LEFT` | **No** — only INFO logs |
| 5 | Intent resolution outcome | (no direct method — happens via `min_stability` filter at `resolver.py:61` and via `Triumvirate` decision evaluation at `triumvirate.py:114`) | `INTENT_RESOLVED` / `DECISION_MADE` | **No** |

**All 5 sites are currently dark.** Constitutional Principle P3 (Transparency) mandates these emit. Week 1 wires them.

### 3.4 memboot bitemporal-lite reference

`~/projects/memboot/src/memboot/models.py:73-76`:
```python
valid_from: str | None = None      # When fact became true in the world (ISO 8601)
recorded_at: str | None = None     # When system learned it (ISO 8601)
```
Defaults: both fall back to `created_at` on write if caller omits.

---

## 4. Acceptance criteria (measurable)

- [ ] **AC1** — `CoordinationEvent` has `valid_from: str | None` and `recorded_at: str | None`. Existing `timestamp` field preserved; on new writes it is set equal to `recorded_at`.
- [ ] **AC2** — Schema migration: existing rows get `valid_from = timestamp`, `recorded_at = timestamp` (backfill). Migration is idempotent (safe to run twice).
- [ ] **AC3** — All 5 mutation sites call `EventLog.record(...)`. Verified by integration test that exercises each site and queries the resulting event.
- [ ] **AC4** — `EventLog.record()` optionally publishes a `Signal` to a `SignalBus`. Wiring is opt-in via constructor (`EventLog(db_path, signal_bus=bus)`); when `signal_bus` is `None`, behavior is unchanged from today.
- [ ] **AC5** — Published signal has `signal_type = f"event.{event_type.value}"`, `source_agent = event.agent_id`, `target_agent = None` (broadcast), `payload = json.dumps(event_dict)`, `timestamp = event.recorded_at`.
- [ ] **AC6** — Throughput: `record()` sustains ≥200 calls/sec on local SQLite (single-thread benchmark, `:memory:` and disk-backed both tested). Signal bus emit adds <2ms p99 overhead per call.
- [ ] **AC7** — `query(valid_from_since=..., valid_from_until=...)` returns rows by world-time range; existing `since` / `until` continue to filter by `recorded_at`. Both ranges composable.
- [ ] **AC8** — Existing 926 Quorum tests pass unchanged.
- [ ] **AC9** — Zero new production dependencies. Verified by `grep dependencies pyproject.toml` showing `dependencies = []` after change.
- [ ] **AC10** — All new code passes `ruff check`, `ruff format --check`, `mypy`. Line length stays under p95 = 76 chars (per package convention).

---

## 5. Implementation

### 5.1 Schema extension

Update `_SCHEMA` in `event_log.py`:

```sql
CREATE TABLE IF NOT EXISTS coordination_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    valid_from TEXT,
    recorded_at TEXT,
    payload TEXT NOT NULL,
    correlation_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_valid_from
    ON coordination_events(valid_from);
CREATE INDEX IF NOT EXISTS idx_events_recorded_at
    ON coordination_events(recorded_at);
-- existing indexes unchanged
```

Add idempotent migration in `__init__`:

```python
def _migrate(self) -> None:
    cur = self._conn.execute("PRAGMA table_info(coordination_events)")
    cols = {row["name"] for row in cur.fetchall()}
    if "valid_from" not in cols:
        self._conn.execute(
            "ALTER TABLE coordination_events ADD COLUMN valid_from TEXT"
        )
        self._conn.execute(
            "UPDATE coordination_events SET valid_from = timestamp"
        )
    if "recorded_at" not in cols:
        self._conn.execute(
            "ALTER TABLE coordination_events ADD COLUMN recorded_at TEXT"
        )
        self._conn.execute(
            "UPDATE coordination_events SET recorded_at = timestamp"
        )
    self._conn.commit()
```

### 5.2 CoordinationEvent extension

```python
@dataclass(frozen=True)
class CoordinationEvent:
    event_id: str
    event_type: EventType
    agent_id: str
    timestamp: str          # alias for recorded_at, kept for back-compat
    payload: dict
    correlation_id: str | None = None
    valid_from: str | None = None
    recorded_at: str | None = None
```

### 5.3 record() signature

```python
def record(
    self,
    event_type: EventType,
    agent_id: str,
    payload: dict | None = None,
    correlation_id: str | None = None,
    timestamp: str | None = None,
    valid_from: str | None = None,
    recorded_at: str | None = None,
) -> CoordinationEvent:
    """Record a coordination event.

    Bitemporal semantics (ported from memboot):
        valid_from: When the fact became true in the world.
            Defaults to recorded_at if omitted.
        recorded_at: When this system observed the fact.
            Defaults to now (UTC) if omitted.
        timestamp: Back-compat alias. Defaults to recorded_at.

    Caller must supply valid_from explicitly when ingesting historical
    data (e.g., importing a past coordination trace).
    """
    now = datetime.now(timezone.utc).isoformat()
    recorded_at = recorded_at or timestamp or now
    valid_from = valid_from or recorded_at
    timestamp = timestamp or recorded_at
    # ... existing logic with new fields persisted ...
    if self._signal_bus is not None:
        self._publish_signal(event)
    return event
```

### 5.4 Signal bridge

```python
def __init__(
    self,
    db_path: str = ":memory:",
    signal_bus: "SignalBus | None" = None,
) -> None:
    # ... existing init ...
    self._signal_bus = signal_bus
    self._migrate()

def _publish_signal(self, event: CoordinationEvent) -> None:
    """Publish a Signal mirroring the event. Failures are swallowed
    and logged — signal bus is best-effort observability, not the
    source of truth."""
    try:
        from convergent.protocol import Signal
        signal = Signal(
            signal_type=f"event.{event.event_type.value}",
            source_agent=event.agent_id,
            target_agent=None,
            payload=json.dumps({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "valid_from": event.valid_from,
                "recorded_at": event.recorded_at,
                "correlation_id": event.correlation_id,
                "payload": event.payload,
            }),
            timestamp=event.recorded_at or event.timestamp,
        )
        self._signal_bus.publish(signal)
    except Exception as exc:
        logger.warning("signal bridge failed: %s", exc)
```

Import is local to keep `EventLog`'s zero-deps profile clean (no top-level coupling to signal_bus module).

### 5.5 Wiring the 5 mutation sites

For each site, the diff is the same shape: accept an optional `event_log: EventLog | None = None` constructor arg, store it, and call `record()` at the mutation point. Default `None` preserves existing behavior; tests that need event verification pass an `EventLog`.

Per-site detail:

| Site | Change |
|---|---|
| `PythonGraphBackend.publish` (resolver.py:49) | `record(INTENT_PUBLISHED, agent_id=intent.agent_id, payload={"intent_id": intent.id, "intent_type": intent.intent_type}, correlation_id=intent.id)` |
| `Intent.add_evidence` (intent.py:169) | `record(SCORE_UPDATED, agent_id=intent.agent_id, payload={"intent_id": intent.id, "old_score": prior, "new_score": new, "evidence_kind": evidence.kind}, correlation_id=intent.id)` — requires plumbing event_log into Intent. **Alternative**: emit from a wrapper at the resolver layer to keep `Intent` pure. **Decision: emit from resolver** to preserve `Intent` as a value object. |
| `Triumvirate.submit_vote` (triumvirate.py:88) | `record(VOTE_CAST, agent_id=vote.voter_id, payload={"request_id": request_id, "decision": vote.decision, "weight": vote.weight}, correlation_id=request_id)` |
| `StigmergyField.leave_marker` (stigmergy.py:69) | `record(MARKER_LEFT, agent_id=agent_id, payload={"marker_type": marker_type, "target": target, "strength": strength, "expires_at": expires_at}, correlation_id=target)` |
| Resolution outcome (resolver.py + triumvirate.py:114) | When stability crosses min threshold downward emit `INTENT_RESOLVED` with payload `{"intent_id": ..., "final_score": ..., "outcome": "demoted"}`. When Triumvirate evaluates a decision emit `DECISION_MADE` with `{"request_id": ..., "outcome": vote_result}` |

---

## 6. Test plan

Match existing pattern (`packages/quorum/tests/test_signal_bus.py:29-56`): factory helpers + class-grouped tests + `tmp_path`.

### 6.1 Unit (new file: `tests/test_event_log_bitemporal.py`)

- `test_record_defaults_recorded_at_to_now`
- `test_record_defaults_valid_from_to_recorded_at`
- `test_record_explicit_valid_from_persisted`
- `test_query_by_valid_from_range`
- `test_query_by_recorded_at_range_back_compat` (existing `since` / `until`)
- `test_query_combined_world_and_record_time`
- `test_migration_idempotent` (run `_migrate()` twice on disk-backed db, no error, no duplicate columns)
- `test_migration_backfills_existing_rows` (insert row with raw SQL pre-migration, run migration, verify `valid_from = recorded_at = timestamp`)

### 6.2 Signal bridge (new file: `tests/test_event_log_signal_bridge.py`)

- `test_record_with_no_bus_does_not_publish` (no-op when bus=None)
- `test_record_with_bus_publishes_signal`
- `test_signal_payload_round_trips_event_fields`
- `test_signal_type_format` (asserts `"event.intent_published"` etc.)
- `test_signal_bus_failure_does_not_break_record` (bus that raises → record still succeeds, warning logged)

### 6.3 Mutation site integration (new file: `tests/test_mutation_sites_emit.py`)

For each of the 5 sites:
- Create EventLog + relevant Quorum subsystem with `event_log=` kwarg
- Trigger the mutation
- Assert one event of the expected type with the expected `correlation_id`
- Assert `valid_from` and `recorded_at` populated

### 6.4 Throughput benchmark (new file: `tests/test_event_log_throughput.py`, marked `@pytest.mark.benchmark`)

- `test_record_throughput_no_bus` — assert ≥200/sec on `:memory:` and disk
- `test_record_throughput_with_bus` — assert ≥150/sec (signal bridge overhead budget)
- `test_signal_emit_p99_under_2ms` — pytest-benchmark stats

### 6.5 Regression

Run full Quorum suite: `cd packages/quorum && PYTHONPATH=python pytest tests/ -v`. All 926 existing tests pass.

---

## 7. File-by-file change list

| File | Change | Estimated LOC delta |
|---|---|---|
| `packages/quorum/python/convergent/event_log.py` | Schema migration, bitemporal fields, signal bridge | +80 |
| `packages/quorum/python/convergent/resolver.py` | Optional `event_log` ctor arg, emit at publish + at resolution | +25 |
| `packages/quorum/python/convergent/triumvirate.py` | Optional `event_log` ctor arg, emit on vote + decision | +20 |
| `packages/quorum/python/convergent/stigmergy.py` | Optional `event_log` ctor arg, emit on leave_marker | +10 |
| `packages/quorum/tests/test_event_log_bitemporal.py` | New | +120 |
| `packages/quorum/tests/test_event_log_signal_bridge.py` | New | +80 |
| `packages/quorum/tests/test_mutation_sites_emit.py` | New | +150 |
| `packages/quorum/tests/test_event_log_throughput.py` | New | +60 |
| `packages/quorum/CHANGELOG.md` | Note 1.3.0 entry | +10 |
| **Total** | | **~555 LOC** |

No new files in production tree beyond what already exists. No new dependencies. No public API removals.

---

## 8. Rollout

1. Land schema migration + bitemporal fields + signal bridge as one PR.
2. Verify on a development checkout: existing 926 tests green, new tests green.
3. Land mutation site wiring as a second PR (separable from schema work).
4. Bump Quorum to 1.3.0 (minor: backwards-compatible additive change).
5. Update root CLAUDE.md test count: 926 → ~926 + new test count.
6. Log ADL entry: `ADL-20260510-001` class TOOLING — "Quorum EventLog completed P3 coverage with bitemporal + signal bridge."

---

## 9. Constitutional alignment

| Principle | How this spec honors it |
|---|---|
| **P1 Sovereignty** | All event data stays in local SQLite. No telemetry. |
| **P2 Continuity** | EventLog is append-only by design. Migration adds columns, never drops or rewrites data. |
| **P3 Transparency** | Closes 5 dark mutation sites — every coordination mutation now leaves an audit trail. This is the principle's explicit demand. |
| **P5 Signed writes (Quorum-specific)** | Confirmed not required for events; only Intent creation needs ed25519 signature. Events about Intents inherit auditability through `correlation_id = intent.id`. |
| **Non-negotiable #4** ("Audit log is sacred") | This spec strengthens the audit log; never weakens it. |

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Schema migration corrupts existing data | Idempotent migration, ALTER TABLE only adds columns, original `timestamp` preserved. Tested in §6.1. |
| Signal bridge becomes a hot path bottleneck | Bus is optional (None by default), bridge wraps publish in try/except so failures never break record(). Throughput AC6 sets 2ms p99 budget. |
| Mutation site wiring breaks existing tests | All wiring is via optional ctor arg (`event_log=None` default). Existing call sites unchanged. AC8 enforces. |
| Rust core (PyO3) needs equivalent change | Out of scope for week 1 — Python `EventLog` is the audit truth; Rust hot path can stay event-emission-free for v1. Re-evaluate at re-eval gate. |
| Forge / Bootstrap subscribers consume the new signals incorrectly | Week 1 ships the producer only. Subscribers are downstream work (Week 2 Watchdog, Week 5 Coupling) and are specced separately. |

---

## 11. What this unblocks

- **Week 2 (LivenessWatchdog):** subscribes to `event.*` signals to detect dead zones.
- **Week 5 (Coupling dashboard):** subscribes to `event.marker_left` for MI computation over stigmergy stream.
- **Forge replay debugging:** `EventLog.query(correlation_id=task_id)` returns full coordination history of a task.
- **Bootstrap dashboard:** `event_timeline()` already renders human-readable output; drop into `/coordination` tile.

---

## 12. Open questions (defer to build)

1. Should `record()` write `payload` as JSON or BLOB? Currently TEXT/JSON — keep, no need to change.
2. WAL checkpoint cadence — leave at SQLite default; benchmark in AC6 will surface if it matters.
3. `event_timeline()` — extend to optionally show `valid_from` separately from `recorded_at`? Only if a real debugging session demands it.

---

*End of spec.*
