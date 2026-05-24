# Spec: Quorum v2 Week 2 — LivenessWatchdog

**Status:** Ready to build (depends on Week 1)
**Mode:** /specification
**Date:** 2026-05-10
**Effort:** 2 days
**Roadmap:** `docs/ROADMAP_quorum_v2.md` Week 2
**Depends on:** Week 1 spec — `docs/specs/quorum_v2_week1_event_log_extension.md`

---

## 0. Premise (do not re-argue)

After Week 1, every Quorum coordination mutation emits a `Signal` of type `event.<event_type>`. That gives us a live heartbeat surface keyed by `agent_id`. Week 2 turns that stream into a liveness watchdog.

Discord delivery is **not** in Quorum (zero-deps invariant). The watchdog emits `ESCALATION_TRIGGERED` events with severity in payload; downstream adapters (Bootstrap dashboard, fleet-monitor pattern) subscribe to those events and deliver alerts.

No new event types. Severity travels in payload. Reuse `ESCALATION_TRIGGERED`.

---

## 1. Goal

Detect Quorum agents that have stopped emitting coordination events and surface them as `ESCALATION_TRIGGERED` events with severity `warn` / `stalled` / `dead`. Alert-only — never auto-act on a dead agent in Week 2.

---

## 2. Out of scope

| Item | Reason |
|---|---|
| Auto-respawn dead agents | Forge or operator owns respawn decisions |
| Adaptive expected_period | Manual per-subject_kind config; learned cadence is post-re-eval-gate |
| Discord webhook code in Quorum | Quorum is zero-deps; adapter is downstream |
| Per-correlation_id liveness (per-task watchdog) | v1 is per-agent_id only; per-task added if re-eval gate justifies |
| Liveness for signal-bus subscribers | Producer-side only — we watch agents that emit, not consumers that listen |
| New event type | `ESCALATION_TRIGGERED` already covers this; severity in payload |

---

## 3. Current-state baseline

### 3.1 What Week 1 provides

- `convergent.event_log.EventLog` with optional `signal_bus` ctor arg
- On every `record()`, a `Signal(signal_type=f"event.{event_type.value}", source_agent=agent_id, ...)` is published
- All 5 mutation sites wired: `INTENT_PUBLISHED`, `SCORE_UPDATED`, `VOTE_CAST`, `MARKER_LEFT`, `INTENT_RESOLVED` / `DECISION_MADE`

### 3.2 SignalBus subscribe API

`packages/quorum/python/convergent/signal_bus.py:106`:
```python
SignalBus.subscribe(
    signal_type: str,
    callback: Callable[[Signal], None],
    agent_id: str | None = None,
) -> None
```
- Sync callback, exceptions swallowed and logged
- Wildcards: not supported (must subscribe per signal_type)

### 3.3 Sweep loop reference (Bootstrap proactive engine)

`packages/bootstrap/.../intelligence/proactive/engine.py`:
- `asyncio.Task` with `_running` flag
- SQLite WAL with idempotent migration via `PRAGMA table_info`
- Quiet hours (we do NOT need this — watchdog runs always)

### 3.4 Discord webhook reference (out of Quorum scope)

`fleet-monitor/src/fleet_monitor/bot.py:33-108`:
- `DISCORD_WEBHOOK_URL` env var
- `discord.Embed` for formatting
- Cooldown via `ALERT_COOLDOWN`
- POST via `httpx.AsyncClient`

This pattern is for the **adapter**, not the watchdog. Documented here so the Bootstrap-side adapter follow-up has a clear template.

---

## 4. Acceptance criteria (measurable)

- [ ] **AC1** — `LivenessWatchdog` in `convergent/liveness/watchdog.py`. Constructor takes `signal_bus: SignalBus`, `event_log: EventLog`, `policies: dict[str, LivenessPolicy]`, `sweep_interval_s: float = 5.0`.
- [ ] **AC2** — `LivenessPolicy` dataclass: `subject_kind: str`, `expected_period_ms: int`, `warn_multiple: float = 2.0`, `stalled_multiple: float = 5.0`, `dead_multiple: float = 30.0`, `failure_threshold: int = 5`.
- [ ] **AC3** — On `start()`, watchdog subscribes to all 10 `event.<event_type>` signal types and begins the sweep loop. On `stop()`, sweep loop cancels cleanly within 1 second.
- [ ] **AC4** — Heartbeat tracking: in-memory dict `last_seen: dict[str, datetime]` keyed by `agent_id`, updated on every received signal. No SQLite read in the hot path.
- [ ] **AC5** — Failure tracking: in-memory `failure_counts: dict[str, int]` incremented via `report_failure(agent_id, exception)` (called by other Quorum subsystems on transition errors). Reset to 0 on successful heartbeat.
- [ ] **AC6** — Sweep emits `ESCALATION_TRIGGERED` event when a tracked agent crosses a threshold:
  - `payload = {"severity": "warn"|"stalled"|"dead", "subject_kind": ..., "agent_id": ..., "elapsed_ms": ..., "expected_ms": ..., "failure_count": ...}`
  - Dedup: at most one event per (agent_id, severity) until the agent heartbeats again or transitions to a higher severity.
- [ ] **AC7** — Resolution: when a previously-alerted agent heartbeats, watchdog emits one `ESCALATION_TRIGGERED` with `payload.severity = "resolved"` and clears the dedup state.
- [ ] **AC8** — Synthetic test (asyncio + freezegun-style time control via injected clock):
  - Start watchdog with policy `expected_period_ms=100`
  - Heartbeat once, advance clock to 250ms → assert `warn` fires
  - Advance to 600ms (no further heartbeat) → assert `stalled` fires
  - Advance to 3500ms → assert `dead` fires
  - Heartbeat → assert `resolved` fires; subsequent advance does not re-fire dead
- [ ] **AC9** — Failure test: call `report_failure(agent_id, ...)` 5× within `expected_period_ms` → assert `dead` fires regardless of heartbeat timing.
- [ ] **AC10** — Stability test: 1-hour synthetic run with 50 agents heartbeating at expected cadence → zero false positives.
- [ ] **AC11** — Zero new production dependencies. Confirmed by `pyproject.toml` showing `dependencies = []`.
- [ ] **AC12** — Existing 926+ Quorum tests pass; new watchdog tests pass; ruff + mypy clean; line length ≤76 p95.

---

## 5. Implementation

### 5.1 Module layout

```
packages/quorum/python/convergent/liveness/
├── __init__.py        # re-exports
├── watchdog.py        # LivenessWatchdog class
├── policy.py          # LivenessPolicy dataclass
└── clock.py           # Clock protocol (testability)
```

### 5.2 Clock protocol (testability)

```python
# liveness/clock.py
from datetime import datetime, timezone
from typing import Protocol

class Clock(Protocol):
    def now(self) -> datetime: ...

class WallClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)
```

Watchdog accepts an optional `clock: Clock = WallClock()`. Tests inject a fake.

### 5.3 LivenessPolicy

```python
# liveness/policy.py
from dataclasses import dataclass

@dataclass(frozen=True)
class LivenessPolicy:
    subject_kind: str
    expected_period_ms: int
    warn_multiple: float = 2.0
    stalled_multiple: float = 5.0
    dead_multiple: float = 30.0
    failure_threshold: int = 5

    def severity_for(
        self,
        elapsed_ms: float,
        failure_count: int,
    ) -> str | None:
        if failure_count >= self.failure_threshold:
            return "dead"
        if elapsed_ms > self.dead_multiple * self.expected_period_ms:
            return "dead"
        if elapsed_ms > self.stalled_multiple * self.expected_period_ms:
            return "stalled"
        if elapsed_ms > self.warn_multiple * self.expected_period_ms:
            return "warn"
        return None
```

Pure function — easy to unit test.

### 5.4 Watchdog skeleton

```python
# liveness/watchdog.py
import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from convergent.event_log import EventLog, EventType
from convergent.liveness.clock import Clock, WallClock
from convergent.liveness.policy import LivenessPolicy
from convergent.protocol import Signal
from convergent.signal_bus import SignalBus

logger = logging.getLogger(__name__)

# Severity ordering for dedup transitions.
_RANK = {"warn": 1, "stalled": 2, "dead": 3}


class LivenessWatchdog:
    """Detects coordination subjects that stop emitting events.

    Subscribes to event.* signals to track per-agent heartbeat in
    memory. Sweep loop emits ESCALATION_TRIGGERED events when a
    subject crosses a severity threshold. Alert-only.
    """

    def __init__(
        self,
        signal_bus: SignalBus,
        event_log: EventLog,
        policies: dict[str, LivenessPolicy],
        sweep_interval_s: float = 5.0,
        clock: Clock | None = None,
    ) -> None:
        self._bus = signal_bus
        self._log = event_log
        self._policies = policies
        self._sweep_interval_s = sweep_interval_s
        self._clock: Clock = clock or WallClock()
        self._last_seen: dict[str, datetime] = {}
        self._subject_kind: dict[str, str] = {}
        self._failure_counts: dict[str, int] = defaultdict(int)
        self._alerted: dict[str, str] = {}  # agent_id -> last severity
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        for et in EventType:
            self._bus.subscribe(
                f"event.{et.value}",
                self._on_event_signal,
            )
        self._task = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def report_failure(
        self,
        agent_id: str,
        exc: BaseException,
    ) -> None:
        self._failure_counts[agent_id] += 1

    def _on_event_signal(self, signal: Signal) -> None:
        agent_id = signal.source_agent
        self._last_seen[agent_id] = self._clock.now()
        # Subject kind inferred from signal_type prefix
        # event.intent_published -> kind from policies map
        # For v1 we map all event types to a single 'agent' kind
        # unless explicit kind hint exists in payload.
        kind = self._infer_subject_kind(signal)
        self._subject_kind[agent_id] = kind
        # Heartbeat clears failure count and alerted state
        if self._failure_counts[agent_id]:
            self._failure_counts[agent_id] = 0
        if agent_id in self._alerted:
            self._emit_resolution(agent_id)

    def _infer_subject_kind(self, signal: Signal) -> str:
        # v1: derive from signal_type when payload doesn't hint.
        # All 10 event types map to "agent" by default.
        # Stigmergy markers can override via payload.subject_kind.
        return "agent"

    async def _sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._sweep_interval_s)
                self._sweep_once()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.exception("watchdog sweep error: %s", exc)

    def _sweep_once(self) -> None:
        now = self._clock.now()
        for agent_id, last in list(self._last_seen.items()):
            kind = self._subject_kind.get(agent_id, "agent")
            policy = self._policies.get(kind)
            if policy is None:
                continue
            elapsed_ms = (now - last).total_seconds() * 1000
            severity = policy.severity_for(
                elapsed_ms,
                self._failure_counts[agent_id],
            )
            if severity is None:
                continue
            prior = self._alerted.get(agent_id)
            if prior == severity:
                continue
            if prior and _RANK[severity] <= _RANK[prior]:
                continue
            self._emit_alert(
                agent_id=agent_id,
                kind=kind,
                severity=severity,
                elapsed_ms=elapsed_ms,
                policy=policy,
            )
            self._alerted[agent_id] = severity

    def _emit_alert(
        self,
        agent_id: str,
        kind: str,
        severity: str,
        elapsed_ms: float,
        policy: LivenessPolicy,
    ) -> None:
        self._log.record(
            event_type=EventType.ESCALATION_TRIGGERED,
            agent_id=agent_id,
            payload={
                "severity": severity,
                "subject_kind": kind,
                "agent_id": agent_id,
                "elapsed_ms": elapsed_ms,
                "expected_ms": policy.expected_period_ms,
                "failure_count": self._failure_counts[agent_id],
            },
            correlation_id=f"liveness:{agent_id}",
        )

    def _emit_resolution(self, agent_id: str) -> None:
        kind = self._subject_kind.get(agent_id, "agent")
        self._log.record(
            event_type=EventType.ESCALATION_TRIGGERED,
            agent_id=agent_id,
            payload={
                "severity": "resolved",
                "subject_kind": kind,
                "agent_id": agent_id,
            },
            correlation_id=f"liveness:{agent_id}",
        )
        del self._alerted[agent_id]
```

### 5.5 Wiring in production

The watchdog is **not** auto-started by Quorum. Bootstrap (or whatever process owns Quorum runtime) constructs and starts it explicitly:

```python
# Bootstrap-side composition (not in Quorum)
event_log = EventLog("animus.db", signal_bus=bus)
watchdog = LivenessWatchdog(
    signal_bus=bus,
    event_log=event_log,
    policies={
        "agent": LivenessPolicy(
            subject_kind="agent",
            expected_period_ms=30_000,
        ),
    },
)
await watchdog.start()
```

Subject_kind hand-tuning: `expected_period_ms=30_000` is conservative; coordination agents that tick faster (e.g., active stigmergy collaborators) get policies with smaller periods.

### 5.6 Discord adapter (out of Quorum)

A separate Bootstrap-side subscriber, modeled on `fleet-monitor/src/fleet_monitor/bot.py:78-108`:

```python
# Pseudocode — lives in Bootstrap, not Quorum
def on_escalation_signal(signal: Signal) -> None:
    payload = json.loads(signal.payload)
    inner = payload["payload"]
    if inner.get("severity") == "warn":
        return  # Discord only on stalled+ to avoid noise
    embed = build_discord_embed(inner)
    httpx.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]})

bus.subscribe("event.escalation_triggered", on_escalation_signal)
```

Adapter lives in Bootstrap intelligence layer; Discord cooldown reuses `ALERT_COOLDOWN` env var.

---

## 6. Test plan

Match Quorum convention: `tests/test_*.py`, factory helpers, class-grouped, `tmp_path`, sync where possible.

### 6.1 Unit (new file: `tests/test_liveness_policy.py`)

- `test_severity_returns_none_under_warn_threshold`
- `test_severity_warn_at_2x`
- `test_severity_stalled_at_5x`
- `test_severity_dead_at_30x`
- `test_failure_count_overrides_timing`

### 6.2 Watchdog (new file: `tests/test_liveness_watchdog.py`)

Use a `FakeClock` and a pumped `asyncio` event loop (no `freezegun` — keep zero deps).

- `test_start_subscribes_to_all_event_types`
- `test_heartbeat_updates_last_seen`
- `test_sweep_emits_warn_at_2x` (advance clock, run `_sweep_once()` manually)
- `test_sweep_does_not_re_emit_same_severity`
- `test_sweep_escalates_warn_to_stalled_to_dead`
- `test_resolution_emitted_on_heartbeat_after_alert`
- `test_resolution_clears_alerted_state`
- `test_failure_count_triggers_dead_directly`
- `test_failure_count_resets_on_heartbeat`
- `test_no_policy_for_kind_skips_subject` (no crash, no alert)
- `test_stop_cancels_sweep_within_one_second`

### 6.3 Stability (new file: `tests/test_liveness_stability.py`, marked `@pytest.mark.slow`)

- `test_one_hour_healthy_run_zero_false_positives` — 50 simulated agents, heartbeats at policy cadence ± jitter, run for synthetic 1-hour clock advancement, assert `_alerted == {}` throughout.

### 6.4 Integration with Week 1 (new file: `tests/test_liveness_integration.py`)

- Real `EventLog` + `SignalBus` (in-memory)
- Trigger `intent.publish(...)` → assert watchdog `_last_seen` updated
- Stop emitting; sweep advances clock; assert ESCALATION_TRIGGERED row appears in EventLog
- Resume emitting; assert resolution event appears

### 6.5 Regression

`cd packages/quorum && PYTHONPATH=python pytest tests/ -v` — all existing 926+ Week-1 tests pass.

---

## 7. File-by-file change list

| File | Change | LOC |
|---|---|---|
| `packages/quorum/python/convergent/liveness/__init__.py` | New | +10 |
| `packages/quorum/python/convergent/liveness/clock.py` | New | +20 |
| `packages/quorum/python/convergent/liveness/policy.py` | New | +30 |
| `packages/quorum/python/convergent/liveness/watchdog.py` | New | +180 |
| `packages/quorum/tests/test_liveness_policy.py` | New | +60 |
| `packages/quorum/tests/test_liveness_watchdog.py` | New | +220 |
| `packages/quorum/tests/test_liveness_stability.py` | New | +80 |
| `packages/quorum/tests/test_liveness_integration.py` | New | +90 |
| `packages/quorum/CHANGELOG.md` | Note 1.4.0 entry | +5 |
| **Total** | | **~695 LOC** |

No production dep changes. No mutations to existing files (watchdog is purely additive).

---

## 8. Rollout

1. Merge after Week 1 (1.3.0) is live.
2. Bump Quorum to 1.4.0 (minor, additive).
3. Add a sample policy preset to Bootstrap composition (non-blocking — preset is operational config).
4. Discord adapter ships as a separate Bootstrap PR, tracked outside this spec.
5. ADL `ADL-20260517-001` class TOOLING — "Liveness watchdog over coordination event stream."

---

## 9. Constitutional alignment

| Principle | How |
|---|---|
| **P1 Sovereignty** | All state in-memory or local SQLite. No external calls from watchdog itself. |
| **P2 Continuity** | ESCALATION_TRIGGERED events are append-only via existing EventLog. |
| **P3 Transparency** | Liveness state changes are themselves audit events — meta-observability. |
| **Non-negotiable #6** ("No unregistered background threads") | Watchdog uses single `asyncio.Task`, exposed via `start()`/`stop()`, cleanly cancellable. |

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sweep loop hot-spins if clock drifts | `asyncio.sleep(sweep_interval_s)` is monotonic; no risk of negative sleep. |
| In-memory state lost on process restart | Acceptable for v1 — restart resets liveness tracking; no persistent claim that any agent is dead. Re-eval at gate. |
| `report_failure` race with sweep | Sweep reads `failure_counts[agent_id]` atomically (Python int read is GIL-protected). Increment is also atomic. No lock needed. |
| Subscribed callbacks are sync; slow signal handlers block bus | `_on_event_signal` is O(1) dict updates only. No I/O. |
| Discord adapter sends storm during legitimate degraded period | Adapter (not watchdog) handles cooldown — out of Quorum scope. |
| Subject_kind always "agent" hides per-task patterns | Documented limitation. v2 adds per-correlation_id subjects only if re-eval gate justifies. |

---

## 11. What this unblocks

- **Bootstrap dashboard `/coordination` tile** showing live alert state.
- **Forge respawn policies** — Forge subscribes to `event.escalation_triggered` with `severity=dead` and decides whether to respawn the agent.
- **fleet-monitor analog for coordination** — same mental model, different layer.

---

## 12. Open questions (defer to build)

1. Should `report_failure` accept a `correlation_id` so failures can be scoped to a task instead of an agent? Defer — track at agent level for v1, add task scoping if Forge actually needs it.
2. Should the watchdog write last_seen to SQLite for restart recovery? Defer — in-memory is fine until a real restart-detection scenario emerges.
3. Should resolution events carry the recovered-from severity? Useful for dashboard color states. Add if Bootstrap dashboard requests it.

---

*End of spec.*
