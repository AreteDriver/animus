# Spec: Quorum v2 Week 5 — Coupling MI Dashboard

**Status:** Ready to build (depends on Week 1)
**Mode:** /specification
**Date:** 2026-05-10
**Effort:** 4 days
**Roadmap:** `docs/ROADMAP_quorum_v2.md` Week 5
**Depends on:** Week 1 spec — `docs/specs/quorum_v2_week1_event_log_extension.md`

---

## 0. Premise (do not re-argue)

Coupling is **observability, never actuation**. The roadmap explicitly rejects the spec's ScheduleController. This week ships a sliding-window mutual-information estimator over the `MARKER_LEFT` event stream, surfaced as a heatmap tile in the Bootstrap dashboard. **It does not write anything back to Quorum state.**

If a future case argues for closing the loop (coupling → schedule), reopen the discussion at the re-evaluation gate, not in this spec.

---

## 1. Goal

Compute pairwise mutual information between agents' stigmergy marker streams, partitioned by `correlation_id` (proxy for task), surface as a heatmap on the Bootstrap dashboard. Operator gets a diagnostic for "which agents are over/under-coupling on this task."

---

## 2. Out of scope

| Item | Reason |
|---|---|
| Transfer entropy | MI alone is enough for the dashboard signal. TE was for the (rejected) controller. |
| Coupling-driven scheduling or rate adjustment | Roadmap rejection — read-only by design |
| Learned embeddings for state discretization | Stigmergy markers are already low-cardinality; hash-mod-K suffices |
| Real-time push updates to the heatmap | Polling at 5s on the dashboard is acceptable; no SSE/WebSocket complexity |
| Persistence of historical CouplingEdge values | In-memory window; if dashboard wants history, snapshot on read and store in dashboard's own SQLite |
| Cross-task coupling | Strict per-`correlation_id` partition; no global coupling |
| Coupling between non-stigmergy event types | v1 is `MARKER_LEFT` only; extend later if a real diagnostic case emerges |

---

## 3. Current-state baseline

### 3.1 What Week 1 provides

`MARKER_LEFT` event from `StigmergyField.leave_marker(...)` carries:
- `agent_id` = source agent
- `correlation_id` = target (per Week 1 spec §5.5)
- `payload = {"marker_type", "target", "strength", "expires_at"}`
- `valid_from`, `recorded_at` (bitemporal)
- Published as `Signal(signal_type="event.marker_left", source_agent=..., payload=json.dumps(envelope))`

Coupling monitor subscribes to `event.marker_left` only.

### 3.2 Bootstrap dashboard pattern

`packages/bootstrap/src/animus_bootstrap/dashboard/routers/automations.py:1-30`:
- `APIRouter` per page
- `runtime = request.app.state.runtime` for shared services
- Jinja template via `request.app.state.templates`
- Runtime exposes a service (`runtime.automation_engine`); router calls `.list_rules()`, `.get_history()`

Coupling follows the same pattern: `runtime.coupling_monitor.heatmap(correlation_id=...)` → dict → render.

### 3.3 Adjacency restriction

Quorum has a graph backend (PythonGraphBackend in `resolver.py`). Two agents are "adjacent" if they share at least one common intent's `provides`/`requires` interface, or appear together in any active correlation. v1 simplification: **two agents are adjacent if they have both emitted `MARKER_LEFT` events under the same `correlation_id` within the window.** This is implicit adjacency from co-participation; cheap and correct for the dashboard signal.

---

## 4. Acceptance criteria (measurable)

- [ ] **AC1** — `CouplingMonitor` in `convergent/coupling/monitor.py`. Constructor: `signal_bus`, `window_size: int = 1000`, `alphabet_size: int = 8`, `min_samples: int = 30`.
- [ ] **AC2** — On `start()`, subscribes to `event.marker_left` only. On `stop()`, unsubscribes (or accepts that signal_bus has no unsubscribe; document as known limitation).
- [ ] **AC3** — Per-(correlation_id, agent_id) rolling buffer of discretized symbols, max length = `window_size`. Symbol = `int(sha256(marker_type + target).hexdigest(), 16) % alphabet_size`.
- [ ] **AC4** — `heatmap(correlation_id: str | None = None) -> CouplingHeatmap` returns dict-shaped payload: `{"correlation_id": ..., "agents": list[str], "matrix": list[list[float]], "sample_counts": list[list[int]], "computed_at": ISO8601}`. None correlation_id → all-tasks aggregate.
- [ ] **AC5** — MI computation uses Miller-Madow corrected estimator: `MI_MM = MI_naive - (R_x + R_y - R_xy - 1) / (2 * n * ln(2))` where `R_*` is observed support size.
- [ ] **AC6** — Pair iteration restricted to agents co-emitting under same `correlation_id` within window. Skip pairs with `n < min_samples`.
- [ ] **AC7** — Synthetic 3-agent ring test: agents A→B copy B→C copy with deterministic mapping; A independent of C. After 1000 ticks: `MI(A,B) ≥ 1.5`, `MI(B,C) ≥ 1.5`, `MI(A,C) ≤ 0.2`. Tolerance ±10%.
- [ ] **AC8** — Performance: heatmap computation for 50 agents × 1 correlation_id < 200ms p95. Bench in `tests/test_coupling_perf.py`.
- [ ] **AC9** — Dashboard route `/coordination/coupling` renders heatmap as HTML table (cells colored by MI value). Template in `packages/bootstrap/.../templates/coupling.html`.
- [ ] **AC10** — Dashboard polls `/coordination/coupling/data?correlation_id=...` every 5s via HTMX `hx-trigger="every 5s"`. JSON endpoint returns `heatmap()` dict.
- [ ] **AC11** — Read-only: zero writes back into Quorum or EventLog from CouplingMonitor. Verified by inspection — no `record()` or `publish()` calls in `monitor.py`.
- [ ] **AC12** — Zero new production deps in Quorum (`pyproject.toml`). Bootstrap may add nothing new (FastAPI + Jinja already present).
- [ ] **AC13** — Existing 926+ Quorum tests + Bootstrap test suite pass; new tests pass; ruff + mypy clean.

---

## 5. Implementation

### 5.1 Module layout (Quorum)

```
packages/quorum/python/convergent/coupling/
├── __init__.py
├── monitor.py        # CouplingMonitor
├── mi.py             # Miller-Madow MI estimator
└── types.py          # CouplingHeatmap dataclass
```

### 5.2 Monitor skeleton

```python
# coupling/monitor.py
import hashlib
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from convergent.coupling.mi import mutual_information_mm
from convergent.coupling.types import CouplingHeatmap
from convergent.protocol import Signal
from convergent.signal_bus import SignalBus

logger = logging.getLogger(__name__)


class CouplingMonitor:
    """Sliding-window MI estimator over MARKER_LEFT events.

    Read-only diagnostic. Subscribes to event.marker_left signals,
    maintains per-(correlation_id, agent_id) symbol buffer, computes
    pairwise MI on demand.
    """

    def __init__(
        self,
        signal_bus: SignalBus,
        window_size: int = 1000,
        alphabet_size: int = 8,
        min_samples: int = 30,
    ) -> None:
        self._bus = signal_bus
        self._window_size = window_size
        self._K = alphabet_size
        self._min_samples = min_samples
        # buffers[correlation_id][agent_id] -> deque[int]
        self._buffers: dict[
            str, dict[str, deque[int]]
        ] = defaultdict(lambda: defaultdict(self._new_buffer))

    def _new_buffer(self) -> deque[int]:
        return deque(maxlen=self._window_size)

    def start(self) -> None:
        self._bus.subscribe(
            "event.marker_left",
            self._on_marker,
        )

    def _on_marker(self, signal: Signal) -> None:
        try:
            envelope = json.loads(signal.payload)
            inner = envelope.get("payload", {})
            correlation_id = envelope.get("correlation_id") or "_global"
            agent_id = signal.source_agent
            marker_type = str(inner.get("marker_type", ""))
            target = str(inner.get("target", ""))
            symbol = self._discretize(marker_type, target)
            self._buffers[correlation_id][agent_id].append(symbol)
        except Exception as exc:
            logger.warning("coupling _on_marker: %s", exc)

    def _discretize(self, marker_type: str, target: str) -> int:
        digest = hashlib.sha256(
            f"{marker_type}|{target}".encode()
        ).hexdigest()
        return int(digest, 16) % self._K

    def heatmap(
        self,
        correlation_id: str | None = None,
    ) -> CouplingHeatmap:
        if correlation_id is not None:
            buffers = {
                correlation_id: self._buffers.get(correlation_id, {})
            }
        else:
            buffers = self._buffers

        # Aggregate agents present in scope
        agents: list[str] = sorted({
            a for buf_map in buffers.values() for a in buf_map.keys()
        })
        n = len(agents)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        counts: list[list[int]] = [[0] * n for _ in range(n)]

        for cid, buf_map in buffers.items():
            local_agents = list(buf_map.keys())
            for i, a in enumerate(local_agents):
                xs = list(buf_map[a])
                gi = agents.index(a)
                for b in local_agents[i + 1:]:
                    ys = list(buf_map[b])
                    m = min(len(xs), len(ys))
                    if m < self._min_samples:
                        continue
                    mi = mutual_information_mm(
                        xs[-m:], ys[-m:], self._K
                    )
                    gj = agents.index(b)
                    # Aggregate across correlation_ids by max
                    if mi > matrix[gi][gj]:
                        matrix[gi][gj] = mi
                        matrix[gj][gi] = mi
                        counts[gi][gj] = m
                        counts[gj][gi] = m

        return CouplingHeatmap(
            correlation_id=correlation_id,
            agents=agents,
            matrix=matrix,
            sample_counts=counts,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
```

### 5.3 Miller-Madow MI estimator

```python
# coupling/mi.py
import math
from collections import defaultdict


def mutual_information_mm(
    xs: list[int],
    ys: list[int],
    alphabet_size: int,
) -> float:
    """Miller-Madow corrected mutual information in bits.

    Naive plug-in MI is biased upward at small n. Miller-Madow
    subtracts (R_x + R_y - R_xy - 1) / (2 n ln 2) where R_* are
    observed support sizes.
    """
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0

    pxy: dict[tuple[int, int], int] = defaultdict(int)
    px: dict[int, int] = defaultdict(int)
    py: dict[int, int] = defaultdict(int)
    for x, y in zip(xs, ys):
        pxy[(x, y)] += 1
        px[x] += 1
        py[y] += 1

    mi = 0.0
    for (x, y), c in pxy.items():
        p_xy = c / n
        p_x = px[x] / n
        p_y = py[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))

    rx = len(px)
    ry = len(py)
    rxy = len(pxy)
    correction = (rx + ry - rxy - 1) / (2 * n * math.log(2))
    return max(0.0, mi - correction)
```

### 5.4 CouplingHeatmap

```python
# coupling/types.py
from dataclasses import dataclass

@dataclass(frozen=True)
class CouplingHeatmap:
    correlation_id: str | None
    agents: list[str]
    matrix: list[list[float]]
    sample_counts: list[list[int]]
    computed_at: str

    def to_dict(self) -> dict:
        return {
            "correlation_id": self.correlation_id,
            "agents": self.agents,
            "matrix": self.matrix,
            "sample_counts": self.sample_counts,
            "computed_at": self.computed_at,
        }
```

### 5.5 Bootstrap dashboard router

```python
# packages/bootstrap/.../dashboard/routers/coupling.py
from fastapi import APIRouter, Request

router = APIRouter()


def _get_monitor(request: Request) -> object | None:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return None
    return getattr(runtime, "coupling_monitor", None)


@router.get("/coordination/coupling")
async def coupling_page(
    request: Request,
    correlation_id: str | None = None,
) -> object:
    templates = request.app.state.templates
    monitor = _get_monitor(request)
    heatmap_data: dict = {}
    if monitor is not None:
        heatmap_data = monitor.heatmap(
            correlation_id=correlation_id
        ).to_dict()
    return templates.TemplateResponse(
        request,
        "coupling.html",
        {
            "heatmap": heatmap_data,
            "correlation_id": correlation_id or "",
        },
    )


@router.get("/coordination/coupling/data")
async def coupling_data(
    request: Request,
    correlation_id: str | None = None,
) -> dict:
    monitor = _get_monitor(request)
    if monitor is None:
        return {"agents": [], "matrix": [], "sample_counts": []}
    return monitor.heatmap(
        correlation_id=correlation_id
    ).to_dict()
```

### 5.6 Template (sketch)

```html
<!-- packages/bootstrap/.../templates/coupling.html -->
{% extends "base.html" %}
{% block content %}
<h1>Coordination Coupling</h1>
<form method="get">
  <label>Task (correlation_id):
    <input name="correlation_id" value="{{ correlation_id }}">
  </label>
  <button>Filter</button>
</form>
<div id="heatmap"
     hx-get="/coordination/coupling/data?correlation_id={{ correlation_id }}"
     hx-trigger="every 5s"
     hx-swap="innerHTML">
  {% include "_coupling_table.html" %}
</div>
<p class="muted">
  MI in bits, Miller-Madow corrected, K={{ 8 }} alphabet.
  Updated {{ heatmap.computed_at }}.
</p>
{% endblock %}
```

`_coupling_table.html` renders the matrix with cell background opacity proportional to MI value (CSS `rgba(...)`).

### 5.7 Runtime composition

Bootstrap startup adds:

```python
from convergent.coupling.monitor import CouplingMonitor

self.coupling_monitor = CouplingMonitor(
    signal_bus=self.signal_bus,
    window_size=1000,
    alphabet_size=8,
)
self.coupling_monitor.start()
```

---

## 6. Test plan

### 6.1 MI estimator unit (new: `tests/test_coupling_mi.py`)

- `test_mi_zero_for_independent_uniform` — synthetic xs, ys uniform random K=8, n=10000 → MI ≤ 0.05
- `test_mi_max_for_identical` — xs == ys → MI ≈ log2(K) (within Miller-Madow correction)
- `test_mi_intermediate_for_partial_dependence` — ys = xs with 30% noise → MI between 0.5 and log2(K)*0.7
- `test_mi_handles_empty_input`
- `test_mi_handles_mismatched_lengths_returns_zero`
- `test_miller_madow_correction_reduces_bias_at_small_n` — compare naive MI vs MM at n=50; MM strictly lower

### 6.2 Monitor unit (new: `tests/test_coupling_monitor.py`)

- `test_subscribes_to_marker_left_only`
- `test_buffer_partitioned_by_correlation_id`
- `test_buffer_capped_at_window_size`
- `test_discretize_deterministic`
- `test_discretize_distributes_uniformly` — 10000 distinct (marker_type, target) pairs → χ² test for uniform distribution over K buckets at p>0.01
- `test_heatmap_skips_pairs_below_min_samples`
- `test_heatmap_no_writes_back_to_quorum` — pass mock signal_bus that records publishes; assert zero publishes

### 6.3 Synthetic ring (new: `tests/test_coupling_ring.py`)

- `test_three_agent_ring_recovers_known_coupling` (AC7)
  - Generate 1000 marker events for agent A
  - Agent B emits markers deterministically derived from A's
  - Agent C emits markers deterministically derived from B's
  - A and C have no direct dependency
  - Run all 3000 events through monitor under same correlation_id
  - Assert `heatmap().matrix` shows MI(A,B) ≥ 1.5, MI(B,C) ≥ 1.5, MI(A,C) ≤ 0.2

### 6.4 Performance (new: `tests/test_coupling_perf.py`, marked `@pytest.mark.benchmark`)

- `test_heatmap_50_agents_under_200ms` — synthetic 50 agents × 1 correlation_id × 1000 events each, assert wall time

### 6.5 Bootstrap router (new: `packages/bootstrap/tests/test_coupling_router.py`)

- `test_coupling_page_renders_with_no_monitor`
- `test_coupling_page_renders_with_monitor`
- `test_coupling_data_returns_json`
- `test_correlation_id_filter_passed_through`

### 6.6 Regression

`cd packages/quorum && PYTHONPATH=python pytest tests/ -v` and `cd packages/bootstrap && pytest tests/ -v` — full suites green.

---

## 7. File-by-file change list

| File | Change | LOC |
|---|---|---|
| `packages/quorum/python/convergent/coupling/__init__.py` | New | +10 |
| `packages/quorum/python/convergent/coupling/monitor.py` | New | +120 |
| `packages/quorum/python/convergent/coupling/mi.py` | New | +50 |
| `packages/quorum/python/convergent/coupling/types.py` | New | +30 |
| `packages/quorum/tests/test_coupling_mi.py` | New | +100 |
| `packages/quorum/tests/test_coupling_monitor.py` | New | +120 |
| `packages/quorum/tests/test_coupling_ring.py` | New | +80 |
| `packages/quorum/tests/test_coupling_perf.py` | New | +50 |
| `packages/bootstrap/.../dashboard/routers/coupling.py` | New | +60 |
| `packages/bootstrap/.../dashboard/templates/coupling.html` | New | +40 |
| `packages/bootstrap/.../dashboard/templates/_coupling_table.html` | New | +30 |
| `packages/bootstrap/.../runtime.py` (or composition module) | Add `coupling_monitor` field + start | +15 |
| `packages/bootstrap/tests/test_coupling_router.py` | New | +100 |
| `packages/quorum/CHANGELOG.md` | 1.6.0 entry | +5 |
| `packages/bootstrap/CHANGELOG.md` | Coupling tile entry | +5 |
| **Total** | | **~815 LOC** |

---

## 8. Rollout

1. PR1: `convergent.coupling` package + tests + ring synthetic. Merge as Quorum 1.6.0.
2. PR2: Bootstrap dashboard router + template + runtime wiring. Merge as Bootstrap minor bump.
3. Visual smoke test: load `/coordination/coupling` in dev, verify heatmap renders.
4. Re-evaluation gate: run new infrastructure for 2 weeks. Did the heatmap surface anything actionable, or is it pretty wallpaper?
5. ADL `ADL-202605XX-002` class TOOLING — "Coupling MI dashboard, read-only diagnostic."

---

## 9. Constitutional alignment

| Principle | How |
|---|---|
| **P1 Sovereignty** | All computation in-process; no network calls; no telemetry |
| **P2 Continuity** | Read-only over EventLog stream; never deletes or modifies events |
| **P3 Transparency** | Surfaces an otherwise-invisible coordination property as a visualization |
| **Non-negotiable #4** ("Audit log is sacred") | Coupling never writes back to EventLog; reads via signal bus only |
| **Non-negotiable #6** ("No unregistered background threads") | Monitor uses signal_bus subscription (callback), not its own thread; no background work outside the bus's existing dispatch |

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MI estimator wrong → misleading dashboard | AC7 synthetic ring is the regression test; Miller-Madow correction handles small-n bias |
| Hash discretization collisions hide signal | K=8 with sha256 mod K is well-distributed; `test_discretize_distributes_uniformly` asserts uniform via χ² |
| Memory grows unbounded with correlation_ids | `defaultdict` of buffers grows. Add LRU cap on correlation_ids (max 100 active) only if a real load problem appears; v1 ships uncapped |
| Heatmap O(N²) cost at scale | AC8 sets a 200ms p95 budget at N=50. If real load exceeds, restrict to top-K busiest agents in the dashboard render layer |
| Signal bus has no unsubscribe | Acknowledged limitation. Monitor lives for runtime lifetime; that's fine for v1 |
| Operator misreads heatmap as causation | Render-time text on the dashboard: "MI is symmetric, does not imply direction. Diagnostic only." |
| Coupling tile invites someone to wire it to control | Template comment + docstring + this spec all flag it. If review catches a controller PR, reject it citing this section |

---

## 11. What this unblocks

- Operator diagnostic: "agents X and Y are over-coupling on task T → maybe their roles overlap"
- Reverse signal: low MI on a task that should be collaborative → "agent C is decoupled, may be stalled even if Liveness Watchdog hasn't flagged"
- Future feature: if re-evaluation gate justifies, coupling history could feed a learned priorities module — explicitly out of scope for v1

---

## 12. Open questions (defer to build)

1. Should heatmap aggregate across correlation_ids by **max** (current §5.2 design) or by **mean weighted by sample count**? Max is more legible at a glance; mean is more statistically honest. Default to max for v1; revisit if a debugging session demands the other.
2. Should the dashboard support task-list filtering (dropdown of active correlation_ids)? Add if operator finds the text input painful.
3. Should we expose a CLI command (`convergent coupling heatmap --task=<id>`) for headless inspection? Cheap to add; defer to first ask.

---

*End of spec.*
