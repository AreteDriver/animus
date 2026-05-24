# Token-optimization patterns lifted from GitHub Agentic Workflows (2026-05)

Source: [Improving token efficiency in GitHub agentic workflows](https://github.blog/ai-and-ml/github-copilot/improving-token-efficiency-in-github-agentic-workflows/) (GitHub Blog, May 2026). They reported sustained **19–62% reductions** across five production agentic workflows (Auto-Triage Issues −62%, Smoke Claude −59%, Security Guard −43%, Community Attribution −37%, Daily Compiler Quality −19%) after rebuilding their cost-attribution and pruning two specific kinds of waste.

This doc is the Forge roadmap for adopting the lift-able pieces. Each pattern: **what GitHub did → where Forge does the equivalent today → the concrete change.**

---

## 1. Cost-axis: Effective Tokens (DONE)

`ET = m × (1.0·I + 0.1·C + 4.0·O)` — input × 1.0, cache-read × 0.1, output × 4.0, scaled by a per-model tier multiplier `m`. Lets a single number rank workflows across Haiku/Sonnet/Opus and across input/cache/output mixes.

**Forge today:** [`src/animus_forge/budget/manager.py`](../../src/animus_forge/budget/manager.py) — `effective_tokens()`, `UsageRecord` carries `input_tokens` / `output_tokens` / `cache_read_tokens` / `model`; `BudgetManager.total_effective_tokens()` + `.effective_tokens_by_agent()`. `BudgetConfig.model_multipliers` overrides the default Haiku 0.08 / Sonnet 1.0 / Opus 5.0 table.

**Status:** shipped in [PR #41](https://github.com/AreteDriver/animus/pull/41). Follow-on: surface ET on the dashboard (`/dashboard/budget`) and in `dashboard/stats` so workflows can be ranked on it.

---

## 2. MCP-tool pruning (8–12 KB / call)

**What GitHub does:** every MCP tool registered with a workflow ships its schema in the system prompt; unused tools are pure context overhead. A Daily Token Optimizer cross-references each workflow's *manifest* against the tools *actually called* and proposes a pruned tool-set. Measured savings: **8–12 KB of context per call** from removing unused MCP tools.

**Forge today:** `WorkflowExecutor` (`workflow/executor_core.py`) loads the configured MCP tool-set per workflow. There's no pruning step — workflows register everything they *might* use. The `mcp_loader` and `mcp/` mixin do schema fetch + registration, not selection.

**Concrete change (queued):**
1. Add `WorkflowExecutor`-side instrumentation that records, per step / per run, *which MCP tool names actually got called* into a per-workflow log (extend the existing audit log or a new `mcp_tool_usage` table).
2. After N runs (default 10), have a `forge analyze mcp-usage <workflow>` CLI emit the actually-used set + a `tools:` config block suggestion (or a `mcp_tool_allowlist:`).
3. Self-improve source: `self_improve/sources/mcp_tool_pruner.py` that watches the usage log and proposes pruned configs as `ImprovementSuggestion`s through the standard orchestrator (`approval → safety → sandbox → rollback → PR`). Same shape as the existing analyzer; YAML-only changes — fits inside `better.md`'s constraint.
4. Treat MCP server *registration order* as a Tier-2 lever: lower-frequency-used tools last (some clients allocate less context to lower-priority tools).

**Read-through to the MCP servers we just built:** `aurora-query` exposes 11 tools, `eve-character` exposes 12 — small surfaces, low absolute waste *for them*. The win is for workflows that pull from the *registry of all configured MCP servers* (which can be 50+ tools combined).

---

## 3. Substitute `gh` CLI for MCP data-fetches

**What GitHub does:** when a workflow needs raw data (PR diffs, file contents, issue bodies), invoking an MCP tool means the LLM has to *reason about* the call. Replacing it with a `gh` CLI invocation in the workflow YAML — either a pre-agentic download stage or a transparent HTTP proxy that routes CLI traffic without exposing credentials — drops the reasoning overhead entirely. The data lands in the workflow as a plain artifact; no tool-call round-trip.

**Forge today:** every workflow that needs external data does it through an MCP tool or a `step.kind: http_get` / similar. There's no convention for "pre-fetch as raw artifact; pass to the agent step."

**Concrete change (queued):**
1. New step kind `cli_fetch` (or extend `bash` / `shell` step if one exists) that runs a CLI command, captures stdout/stderr, and attaches the result to the workflow context as a named artifact (e.g. `${artifacts.pr_diff}`).
2. A YAML-level convention: when a step needs a fixed, predictable piece of data (`pr_diff`, `file_contents`, `issue_body`), prefer the `cli_fetch` artifact over a tool call. Document in `workflows/examples/`.
3. The same idea generalises beyond `gh` — `curl`, `flyctl logs`, `kubectl get`, `psql -c` are all data-fetch substitutes that bypass tool reasoning.

**Risk:** caller-managed args + plain CLI = wider attack surface than typed MCP tool calls. Gate behind workflow approval/safety like any other shell step.

---

## 4. Self-monitoring agentic workflows over `token-usage.jsonl`

**What GitHub does:** the observability layer is just structured log emission — an API proxy capturing every LLM call's `input` / `output` / `cache_read` token counts into a JSONL artifact per workflow run. Then **two agentic workflows monitor** the JSONL: a *Daily Token Usage Auditor* (flags anomalous/expensive workflows) and a *Daily Token Optimizer* (identifies inefficiencies and proposes patches). Both are themselves agentic workflows — the system optimises itself with the same primitive it observes.

**Forge today:** `BudgetManager` writes `budget_session_usage` to the configured backend (SQLite/Postgres). There's no daily-cron auditor; the proactive engine has self-heal checks for tool failures/slow/errors but not for cost regressions.

**Concrete change (queued):**
1. Daily proactive check `cost_audit` (`packages/core/animus/...`/`packages/forge/.../monitoring/`): query the budget table for the last 24 h, compute ET-per-run and ET-per-agent, compare to the trailing 7-day baseline, surface anomalies (anything > 2σ or > 1.5× baseline) to the existing notification path.
2. Self-improve source `self_improve/sources/cost_optimizer.py` that reads the same data and proposes:
   - tool-set prunes (from §2),
   - `cli_fetch` substitutions (from §3),
   - model-tier downshifts (if a step's outputs are short and structured, can it run on Haiku?).
3. Expose ET-per-workflow in the dashboard so a human can spot the same anomalies the auditor flags.

---

## Adoption sequence (recommended)

1. ✅ ET metric (§1) — [PR #41](https://github.com/AreteDriver/animus/pull/41).
2. Dashboard surfacing of ET (small).
3. MCP-tool usage instrumentation (§2 step 1) — needed before the pruner can do anything.
4. `cli_fetch` step kind (§3) — independent; can land in parallel with §3.
5. `cost_audit` proactive check (§4 step 1) — needs the ET dashboard or at least ET aggregation in the backend.
6. The two `self_improve/sources/` modules (§2 step 3, §4 step 2) — last; they consume everything above.

---

## What we deliberately *aren't* lifting

- GitHub's specific `gh-aw` framework for declaring agentic workflows in Actions. Forge's YAML workflow model is the equivalent surface; we'd duplicate ourselves.
- The exact `m` values for non-Anthropic / non-OpenAI providers. Default table covers Haiku/Sonnet/Opus/GPT-4o/GPT-5/Ollama; everything else falls back to 1.0 and the user overrides via `BudgetConfig.model_multipliers`.
- A direct HTTP proxy for MCP-credential routing. We have `core/security` + audit logging; building a transparent proxy adds an attack surface for a small per-call win.
