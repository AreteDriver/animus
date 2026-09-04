# Animus Rework Spec

> **Purpose:** Unblock Animus by killing the hardware-gating pattern and shipping a working v0.1 on existing infrastructure. Reposition Animus as the private reference implementation of the verifiable-AI-decision primitive stack.

---

## Part 1: Strategy

### 1.1 Current state

Animus is a three-layer personal AI architecture:

- **Core** — operating environment UI, identity anchor, signed memory
- **Forge** — orchestration engine (formerly Gorgon)
- **Quorum** — stigmergic coordination protocol (formerly Convergent)

Signing is Ed25519. Memory writes are append-only. The Zorya Triumvirate (Utrennyaya, Vechernyaya, Polunochnaya) is a three-model ensemble intended to run locally on Mac Studio M4 Ultra hardware. `IDENTITY.md` and `budget.py` are named as first-build priorities.

Nothing is shipped. The project is gated on hardware that is not acquired, under an architecture that is more ambitious than the current need requires.

### 1.2 The diagnosis

This is perfectionism presented as engineering rigor. The triumvirate is a v2 feature being held up as a v1 requirement. The hardware gate is real money being waited on to justify work that does not need the hardware to start.

The symptom is that Animus has been a named project for months and produced zero running code reachable from outside your head. That is not a roadmap problem. That is a shipping problem, and it compounds.

### 1.3 The repositioning

Animus becomes **the private reference implementation of the Arete primitive stack.** Every primitive (P1-P7 from the pattern-reuse playbook) ships in Animus first. Your personal operating environment is how you dogfood the studio's shared infrastructure.

This reframes the question. You are not building a personal AI system and also building a venture studio. You are building the venture studio's infrastructure and running the first instance of it on yourself. Animus is v0.1 of everything.

That framing removes the hardware dependency immediately. If Animus is the reference implementation, it needs to run on what the studio's products will run on — FastAPI, Postgres, Claude API, standard cloud. The triumvirate becomes optional and later.

### 1.4 The spine

**Animus v0.1 is a signed-memory operating environment with budgeted Claude access and an identity anchor.** One model. No local inference. No triumvirate. Append-only log. Running in a month.

Everything beyond that is v2.

### 1.5 What this unlocks

- Ledger (P1) gets its reference implementation
- Context Bundle (P5) gets its reference implementation
- Arete-context-mcp stops being a standalone repo and becomes the external-facing view onto Animus memory
- The budget-gate pattern (budget.py) becomes P1-adjacent infrastructure every product can reuse
- You start having actual usage data on your own system, which is the only honest way to iterate

---

## Part 2: Build Blueprint

### 2.1 v0.1 scope

```
Animus v0.1
├── identity/
│   └── IDENTITY.md              # Source of truth for who the system is
├── memory/
│   ├── ledger.py                # Append-only signed log (P1)
│   ├── keypair.py               # Ed25519 signing
│   └── schema.sql               # Postgres schema
├── budget/
│   └── budget.py                # Token + dollar budget gate
├── orchestration/
│   └── forge.py                 # Single-model Claude wrapper
├── api/
│   └── server.py                # FastAPI, minimal surface
└── ui/
    └── [Next.js, deferred until API works]
```

That is the whole system. Eight files. Maybe 600 lines total.

### 2.2 Component specs

**`IDENTITY.md`** (not code, but load-bearing)

Plain markdown. Defines who Animus is, what it remembers, what it refuses, what its relationship to you is. Signed with your key and hashed into every memory write as the anchor. If `IDENTITY.md` changes, that is a versioned event in the ledger.

This is the one piece that is philosophical, not technical. Write it once, carefully. Revisions are rare and explicit.

**`memory/ledger.py`**

```python
def record(
    event_type: str,
    payload: dict,
    identity_hash: str,
    signer: Ed25519PrivateKey,
) -> SignedEntry:
    """Append a signed entry to the ledger. Never mutates existing entries."""
```

Observable behavior:

- Entries are append-only
- Each entry includes: timestamp, event_type, payload, identity_hash, signature
- Signature covers the full entry contents including timestamp
- Verification is a separate function that reads without modifying

Non-goals: no garbage collection, no compaction, no replication. v0.1 just appends.

**`budget/budget.py`**

```python
def check_and_reserve(
    intended_tokens: int,
    intended_dollars: float,
    operation: str,
) -> BudgetDecision:
    """Return allow/deny with remaining budget. Records reservation."""
```

Observable behavior:

- Daily and monthly caps enforced
- Reservations are recorded in the ledger
- Actual spend (vs. reservation) is reconciled after the operation completes
- Budget overrun requires an explicit override flag

This is the pattern every studio product reuses. Build it well here once.

**`orchestration/forge.py`**

```python
def query(
    prompt: str,
    context: list[dict] | None = None,
) -> Response:
    """Query Claude with budget check + ledger write."""
```

Single model. Claude API. Budget-gated. Every call logged to the ledger with prompt hash, response hash, token count, dollar cost.

The triumvirate is a later refactor where `query` routes across three models and returns an ensemble result. v0.1 is the wrapper that makes that refactor trivial later.

**`api/server.py`**

FastAPI. Minimal endpoints:

- `POST /query` — authenticated query through Forge
- `GET /memory/recent` — read recent ledger entries
- `GET /memory/search` — text search over ledger
- `GET /budget/status` — current budget state

No UI yet. Use curl or the API docs page. UI is v0.2.

### 2.3 What v0.1 explicitly does NOT include

- Triumvirate (three-model ensemble)
- Local inference of any kind
- Quorum / stigmergic coordination
- Web UI
- Mobile access
- External integrations (Slack, email, calendar)
- Multi-user support
- Replication or backup beyond Postgres dumps

Every one of these is a v0.2+ feature. Shipping v0.1 without them is the point.

### 2.4 Build order (four-week target)

**Week 1: Foundation**
- Write `IDENTITY.md`
- Implement Ed25519 keypair management
- Implement `ledger.py` with Postgres schema
- Tests for append-only invariants

**Week 2: Budget + Orchestration**
- Implement `budget.py`
- Implement `forge.py` with single Claude model
- Wire budget check into every forge call
- Tests for budget enforcement

**Week 3: API**
- Implement FastAPI server
- Auth (API key, single user)
- Connect all endpoints
- Local deployment via Docker Compose

**Week 4: Dogfood + Document**
- Use it. Daily. For real queries.
- Document what breaks
- Write the v0.2 roadmap based on actual usage gaps
- Decide whether UI or triumvirate is the next unlock

### 2.5 Migration from current state

Assuming the current Animus is largely in planning/early-code form:

1. **Freeze current branch.** Anything in progress, commit and tag as `pre-v0.1-archive`.
2. **Start `v0.1` branch from scratch.** Do not port old code; the architecture is different enough that porting is slower than rewriting.
3. **First commit: `IDENTITY.md`.** Literally the first file.
4. **No feature additions during v0.1 build.** If an idea arrives, write it to a `v0.2_ideas.md` file and keep moving.

### 2.6 Integration with primitive stack

Animus v0.1 produces the reference implementations of:

- **P1 (Signed Decision Ledger):** `memory/ledger.py` is extracted as `arete-ledger` package
- **P5 (Context Bundle):** `arete-context-mcp` becomes the external-facing MCP server that reads from Animus memory
- **Budget gate pattern:** `budget/budget.py` is extracted as `arete-budget` utility

The extraction happens in v0.2. v0.1 keeps everything in one repo for velocity.

### 2.7 Risk: scope creep during build

The biggest risk to v0.1 is you, during the build. The triumvirate is interesting. Stigmergic coordination is interesting. The philosophy of the operating environment is interesting. None of them ship v0.1.

Mitigation: Put a physical reminder somewhere you work that says "v0.1 is the whole point." When a v0.2 idea arrives, write it down and keep moving. The ideas are not going anywhere. The shipped system is the only thing that enables the ideas to matter.

### 2.8 Success criteria

v0.1 succeeds if:

1. You run a query through the system daily for four weeks without it breaking
2. You can verify any memory entry's signature from the command line
3. The budget gate has prevented at least one unintended cost overrun
4. `IDENTITY.md` has not been edited more than twice
5. The v0.2 roadmap is written from real usage data, not speculation

If any of those fail, v0.1 is not done. If all of them pass, extract the primitives and move on to v0.2.

---

## Part 3: What to do this week

1. **Write `IDENTITY.md` today.** One sitting, one hour. Do not polish; just commit the first version. Revisions come with use.
2. **Create the v0.1 branch.** Archive current work as noted above. Start clean.
3. **Spec the ledger schema in SQL.** This is the second load-bearing artifact after IDENTITY.md. Get it right on paper before writing migration code.
4. **Commit to no hardware purchases for Animus this quarter.** The triumvirate is a future problem. Write that decision down somewhere visible.
5. **Pick the dogfood query you will run first.** "What did I work on yesterday" is a good candidate. Having the first real use case in mind focuses the API surface.
