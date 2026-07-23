# Citizen Integration Plan — Operations Center Phase 5

**Goal:** Wire the Animus Phase 0 Citizens into the Operations Center dashboard, turning it from a runtime monitor into a **sovereign AI operating environment control plane**.

**Scope:** This is a scoped integration — we bridge existing citizen infrastructure (in `packages/core/`) to the dashboard (in `packages/bootstrap/`) without reimplementing citizen logic.

---

## Current State

| Component | Location | Status |
|---|---|---|
| Architect Citizen | `packages/core/animus/citizens/architect.py` | ✅ Implemented — produces `ImprovementProposal` |
| Conversation Designer | `packages/core/animus/citizens/conversation_designer.py` | ✅ Implemented — detects correction loops, vague requests |
| Knowledge Curator | `packages/core/animus/citizens/knowledge_curator.py` | ✅ Implemented — finds stale references, contradictions |
| Test Oracle | `packages/core/animus/citizens/test_oracle.py` | ✅ Implemented — analyzes test health, coverage trends |
| Session Steward | `packages/core/animus/citizens/session_steward.py` | ✅ Implemented — retrospective session audits |
| Citizen Council | `packages/core/animus/citizens/citizen_council.py` | ✅ Implemented — ranks backlog from all citizens |
| Forge Commissioner | `packages/core/animus/citizens/commissioner.py` | ✅ Implemented — commissions Forge on approved proposals |
| Proposal Schema | `packages/core/animus/citizens/proposal.py` | ✅ Implemented — `ImprovementProposal` dataclass |
| Dashboard Proposals | `packages/bootstrap/dashboard/routers/proposals.py` | ✅ Exists — but only for **identity proposals**, not Architect proposals |
| Event Ledger | `packages/bootstrap/intelligence/event_ledger.py` | ✅ Ready — records all operational events |

**Gap:** The dashboard has no visibility into citizen activity. Architect proposals go to memory; operators must query memory via CLI or MCP to see them. There is no unified citizen control surface.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Operations Center Dashboard                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Home    │  │ Citizens │  │ Proposals│  │  Citizen Detail  │  │
│  │ (status) │  │ (status) │  │ (backlog)│  │  (per-citizen)   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘  │
│       │             │             │                  │            │
│  ┌────┴─────────────┴─────────────┴──────────────────┴────────┐  │
│  │              Citizen Bridge (bootstrap adapter)                │  │
│  │  - Queries core memory for proposals                          │  │
│  │  - Surfaces citizen status (idle/observing/proposing)         │  │
│  │  - Provides approve/reject/commission endpoints               │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           │                                        │
│  ┌────────────────────────┴──────────────────────────────────────┐  │
│  │              Core Animus Citizens (packages/core/)             │  │
│  │  Architect → ImprovementProposal → Memory → Council → Forge   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Work Breakdown

### Task 5.1 — Citizen Bridge Module
**File:** `packages/bootstrap/src/animus_bootstrap/intelligence/citizen_bridge.py`

Create a thin adapter that exposes core citizen data to the bootstrap dashboard without importing heavy core dependencies at module level (lazy import).

**Responsibilities:**
- `list_citizen_proposals(limit=50)` — Query memory for `ImprovementProposal` objects from all citizens
- `get_citizen_status()` — Return citizen activity state (idle / observing / proposing / error)
- `get_citizen_backlog()` — Use `CitizenCouncil` to produce ranked backlog
- `approve_proposal(proposal_id)` — Update proposal status → `APPROVED`
- `reject_proposal(proposal_id)` — Update proposal status → `REJECTED`
- `commission_proposal(proposal_id)` — Hand approved proposal to `ForgeCommissioner`

**Design constraint:** Use lazy imports (`importlib.import_module`) so bootstrap can run without core installed. If core is absent, bridge returns empty lists gracefully.

---

### Task 5.2 — Citizens Dashboard Router
**File:** `packages/bootstrap/src/animus_bootstrap/dashboard/routers/citizens_page.py`

**Endpoints:**
- `GET /citizens` — Main citizens page
  - Citizen status cards (5 citizens, color-coded: green=idle, yellow=observing, blue=proposing, red=error)
  - Recent proposal count per citizen
  - Council backlog summary (total proposals, top-ranked)
- `GET /citizens/proposals` — Architect proposals list
  - Table: ID, Title, Confidence, Effort, Status, Source Citizen
  - Filter by status (draft / approved / rejected / commissioned / complete)
- `POST /citizens/proposals/{id}/approve` — Approve with CSRF protection
- `POST /citizens/proposals/{id}/reject` — Reject with CSRF protection
- `POST /citizens/proposals/{id}/commission` — Commission to Forge (only if status=approved)
- `GET /citizens/{name}` — Per-citizen detail page
  - Recent observations / findings
  - Last scan timestamp
  - Proposal history

**Template:** `citizens.html` + `fragments/citizen_proposal_row.html` (HTMX for approve/reject/commission)

---

### Task 5.3 — Wire Event Ledger
Every citizen action must record to the Event Ledger:

| Event Type | Source | Payload |
|---|---|---|
| `citizen_scan_started` | `citizens.{name}` | `{citizen_name, focus}` |
| `citizen_scan_completed` | `citizens.{name}` | `{citizen_name, observations_count, proposal_id}` |
| `citizen_proposal_approved` | `dashboard` | `{proposal_id, citizen_name}` |
| `citizen_proposal_rejected` | `dashboard` | `{proposal_id, citizen_name}` |
| `citizen_proposal_commissioned` | `dashboard` | `{proposal_id, forge_status}` |
| `citizen_error` | `citizens.{name}` | `{citizen_name, error}` |

---

### Task 5.4 — Dashboard Navigation
- Add "Citizens" nav link in `base.html` (between Events and Tasks)
- Keyboard shortcut: `g` `c` → Citizens page (already exists for Conversations; update to `g` `z` or use `g` `ci`)

---

### Task 5.5 — Tests
**File:** `packages/bootstrap/tests/test_dashboard_citizens.py`

- Test citizen page renders with mock bridge
- Test proposal list filters by status
- Test approve/reject/commission endpoints require CSRF
- Test commission rejected when proposal not approved
- Test event ledger records citizen actions
- Test graceful degradation when core not installed

---

### Task 5.6 — Documentation Update
Update `docs/dashboard/operations-center.md`:
- Add Citizens section to architecture diagram
- Document citizen event types
- Document approval → commission → evidence pipeline

---

## Execution Order

1. **Task 5.1** — Citizen Bridge (foundation; all other tasks depend on it)
2. **Task 5.2** — Citizens Router + Templates (the visible surface)
3. **Task 5.4** — Navigation wiring (small, can parallel with 5.2)
4. **Task 5.3** — Event Ledger integration (can be done during 5.2)
5. **Task 5.5** — Tests (after 5.2 is stable)
6. **Task 5.6** — Documentation (last)

---

## Acceptance Criteria

- [ ] `/citizens` page loads and shows 5 citizen status cards
- [ ] `/citizens/proposals` lists Architect proposals with approve/reject/commission buttons
- [ ] Approving a proposal changes its status and records an event
- [ ] Commissioning an approved proposal triggers ForgeCommissioner (or simulates if Forge unavailable)
- [ ] All new POST endpoints have CSRF protection (Phase 0 does not regress)
- [ ] Event Ledger records every citizen action
- [ ] Dashboard works when core package is not installed (graceful degradation)
- [ ] 10+ new tests green
- [ ] Final regression: all 77+ dashboard tests pass

---

## Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Core package import failures break bootstrap | Lazy imports in bridge; graceful fallback to empty state |
| Proposal schema drift between core and bootstrap | Bridge uses `to_dict()` / `from_dict()` serialization; no direct dataclass dependency |
| ForgeCommissioner requires env vars not set in dashboard | Simulate commission result when Forge unavailable (same as CLI behavior) |
| Memory queries are slow | Cap at 50 proposals; cache for 30s in bridge |
| Citizen Council ranking is CPU-heavy | Run ranking server-side, paginate results |

---

## Estimated Effort

| Task | Estimated Time |
|---|---|
| 5.1 Citizen Bridge | 45 min |
| 5.2 Citizens Router + Templates | 90 min |
| 5.3 Event Ledger Wiring | 15 min |
| 5.4 Navigation | 10 min |
| 5.5 Tests | 45 min |
| 5.6 Documentation | 20 min |
| **Total** | **~4.5 hours** |

---

## Success Narrative

> "The Operations Center now shows not just what the runtime is doing, but what the citizens are thinking. I can see the Architect observing the codebase, the Conversation Designer detecting correction loops, the Test Oracle flagging coverage gaps — all in one place. When the Architect produces a proposal, I review it on the Citizens page, approve it with one click, and commission Forge to implement it — without leaving the dashboard. The full loop: Observe → Propose → Approve → Commission → Evidence → Merge — is now visible and operable from a single surface."
