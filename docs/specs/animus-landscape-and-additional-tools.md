# Animus — Competitive Landscape, Differentiation & Additional Tools
**Date:** 2026-03-04  
**Purpose:** Honest positioning analysis + spec for three additional Arete Tools  
**For:** Strategic planning, Claude Code, Animus self-directed development

---

## Part 1: Where Animus Stands in the Current Landscape

### The Honest Map

There are four distinct categories of things Animus touches. It has overlap in each — but no single competitor occupies the same intersection.

```
CATEGORY A — Agent Orchestration Frameworks
LangChain / LangGraph, CrewAI, AutoGen, Semantic Kernel
→ Animus Forge overlaps here

CATEGORY B — Personal AI Assistants / Exocortex
OpenClaw, ai.com, Freysa, EXO
→ Animus Core overlaps here

CATEGORY C — Agent Observability / Debugging
LangSmith, Langfuse, Arize, AgentOps
→ Arete Tools (Signal, Autopsy) overlap here

CATEGORY D — Local / Sovereign AI
Ollama ecosystem, local-AI hobbyist movement
→ Animus's runtime overlaps here
```

Animus is the **only project that spans all four categories in a unified architecture.** That's not marketing — that's a structural fact that emerges from the three-layer design.

---

### Category A: Agent Orchestration — What Overlaps, What Doesn't

**What exists:**

<br>

| Framework | Architecture | Differentiation |
|-----------|-------------|-----------------|
| LangGraph | Graph-based state machine | Best for complex branching workflows |
| CrewAI | Role-based crews, YAML config | Best for declarative multi-agent |
| AutoGen → Microsoft Agent Framework | Conversation-based multi-agent | Enterprise, Azure-native, in maintenance mode |
| OpenAI Agents SDK | Managed runtime, first-party tools | Lowest friction, highest lock-in |

**Where Forge overlaps:**
Forge's YAML-declarative workflow style is closest to CrewAI. Both define agents with roles and tasks in config files rather than code. Both support multi-agent coordination.

**Where Forge diverges — genuinely:**

1. **Budget-first design.** No framework has token/cost budgets as a first-class primitive. Forge's `max_cost_usd` per workflow is a hard ceiling, not a monitoring afterthought. Every framework tracks cost as an observability concern. Forge treats it as an execution constraint.

2. **Stigmergic coordination (Quorum).** Every framework — LangGraph, CrewAI, AutoGen — uses some form of centralized coordination: a graph controller, a crew manager, a group chat orchestrator. Quorum has no coordinator. Agents read and write shared environment state. This is architecturally novel and eliminates the single point of failure that kills production multi-agent systems.

3. **Self-improving runtime.** No framework has a reflection loop that modifies the system's own configuration based on observed outcomes. Animus does. This isn't an agent that improves outputs — it's a system that improves itself.

4. **Human-first identity layer.** Frameworks are headless execution environments. Animus Core is a persistent cognitive companion with identity files, episodic memory, and a daily reflection loop. Frameworks have no concept of the human operating them.

**Honest overlap risk:** CrewAI is the closest architectural cousin. If CrewAI ships budget enforcement and YAML-native workflows improve, Forge's surface-level differentiation shrinks. The moat is Quorum + Core + the self-improvement loop — none of which CrewAI is building.

---

### Category B: Personal AI Assistants — The OpenClaw Problem

**OpenClaw is the biggest competitive signal to watch.**

<br>

OpenClaw (formerly Clawdbot) went viral in late 2025 — MIT-licensed, open-source, personal AI assistant that executes real actions: clears inboxes, manages calendars, runs shell commands, controls browsers, reads/writes files. It's the closest thing to what Animus Core is building.

**Key differences:**

| Dimension | OpenClaw | Animus Core |
|-----------|----------|-------------|
| Architecture | Messaging-first (WhatsApp, Telegram, etc.) | CLI + local dashboard |
| Memory | Session-based | Persistent episodic/semantic/procedural |
| Identity | None — generic assistant | Six identity files, user-sovereign |
| Self-improvement | None | Reflection loop → LEARNED.md |
| Privacy concern | Security researchers alarmed — personal data exposure | Local-first, no cloud by default |
| Coordination | Single agent | Three-layer: Core + Forge + Quorum |

**The OpenClaw security story is Animus's opening.** MIT Technology Review published concerns about OpenClaw's security vulnerabilities. The Chinese government issued a public warning. This creates a trust gap that Animus's sovereignty-first design fills directly.

**Honest risk:** OpenClaw has massive GitHub traction. If they fix the security issues and add persistent memory, they eat a significant portion of Animus Core's addressable market. Ship Core before that happens.

**ai.com** is a Super Bowl-level funded consumer play — personal agents for non-technical users. They're not building for sovereignty or developers. No overlap on target user.

**Freysa** is crypto-native digital twins. Interesting concept, different market entirely.

---

### Category C: Observability — Where Arete Tools Live

**The existing tools:**

LangSmith, Langfuse, Arize Phoenix, AgentOps, Helicone — all observability tools. They show what happened. None classify why failure occurred, generate fix recommendations, or score epistemic quality of outputs.

**Arete Tools (Signal, Autopsy) are not observability tools.** They're a forensics and quality layer that sits on top of observability data. The positioning must be explicit: "not another LangSmith — a layer above LangSmith."

**Honest competitive gap:** This space is moving fast. Maxim AI is building toward comprehensive end-to-end quality platforms. Watch them. If they add failure classification and output quality scoring, Autopsy's standalone case weakens. The moat is speed — ship before they expand scope.

---

### Category D: Local AI / Sovereignty Movement

A genuine cultural movement is forming around local AI. The "from cloud to couch" narrative is gaining traction among technically capable users tired of paying $132/month for rented intelligence. Ollama is the runtime. The missing piece is a cognitive layer on top of it.

**Animus is the cognitive layer the Ollama ecosystem is missing.** Every local AI user runs Ollama and then manually manages their prompts, contexts, and sessions. Animus gives them persistent memory, identity, self-improvement, and orchestration. This is the natural distribution channel for Core: Ollama community, r/LocalLLaMA, HN local-AI posts.

---

## Part 2: The Differentiation Stack — What Makes Animus Genuinely Unique

In order of defensibility:

### 1. Stigmergic Coordination via Intent Graph (Quorum) — Most Defensible
No production multi-agent framework uses stigmergy. Every competitor has a coordinator or message-passing. Quorum's mechanism: agents write IntentNodes to a shared SQLite-backed graph before and after decisions. The StabilityScorer replaces pheromone concentration — intents strengthen when consumed by other agents, decay when unvalidated. The IntentResolver gives every agent local awareness of high-stability decisions in its semantic neighborhood before it acts. Three boid rules (separation, alignment, cohesion) applied through the resolver produce coordinated output without any agent ever talking directly to another. This is architecturally novel, grounded in forty years of biological research (Grassé 1959, Reynolds 1987), and impossible to retrofit into LangGraph or CrewAI without replacing their core coordination assumptions.

### 2. Self-Improving Runtime (Core Phase 1b) — Very Defensible
The reflection loop + identity file writes + proposal system creates a system that compounds intelligence over time. No framework, no personal assistant, no observability tool does this. The longer Animus runs, the more it diverges from any competitor in its understanding of its user.

### 3. Budget-as-Execution-Constraint (Forge) — Moderately Defensible
Cost as a hard ceiling rather than a monitoring concern. Easy to copy once someone notices it. First-mover advantage here is real but time-limited.

### 4. Arete Tools as Self-Instrumentation — Uniquely Compound
Animus running Signal on its own outputs, Autopsy on its own failures, and Verdict on its own decisions creates a feedback loop no standalone tool can replicate. The tools instrument themselves through the system that builds them. This is the moat that takes 6-12 months to emerge but becomes impossible to copy once established.

### 5. Sovereignty + Local-First (Core) — Culturally Differentiating
The OpenClaw security panic validates this positioning. "Yours, not rented" resonates strongly with the emerging local-AI movement. Not a technical moat, but a strong trust and community moat.

---

## Part 3: Similarity Summary (Honest)

```
Animus Core      ≈ OpenClaw (with persistent memory + identity + sovereignty)
Animus Forge     ≈ CrewAI (with budget enforcement + Quorum coordination)
Animus Quorum     ≈ nothing shipping in production (stigmergy is novel)
Arete Tools      ≈ above LangSmith (forensics layer, not observability layer)
Full Animus      ≈ nothing — the combination doesn't exist
```

The pitch isn't "we're different from X." The pitch is: **"We're the only system where the orchestration layer, the personal memory layer, the coordination layer, and the quality instrumentation layer are built as one unified architecture that improves itself."**

That sentence describes nothing currently shipping.

---

## Part 4: Additional Arete Tools Specifications

Three tools identified as high-value, currently unspecced. All follow the same pattern: standalone CLI first, Forge integration second, Core extension third.

---

### Tool A: Context Hygiene
**AI conversation quality monitor**  
**Repo:** `arete-tools/context-hygiene`  
**Priority:** Build after Signal ships

#### Problem
AI conversations degrade as context accumulates. Old constraints contradict new ones. Resolved threads still consume window space. Stale goals influence current responses. Developers and power users don't notice the degradation — outputs get subtly worse without a clear signal. By the time the problem is obvious, the conversation is unsalvageable.

Nobody has built a tool that tells you *when* a conversation context has gone stale and *exactly what* to prune to restore response quality.

#### What it does
Analyzes an active AI conversation's context window and returns:
- **Staleness score** — percentage of context that is outdated or resolved
- **Contradiction map** — instructions or constraints that conflict with each other
- **Compression candidates** — threads that can be summarized without losing value
- **Dead weight** — content with zero influence on current task
- **Pruning recommendation** — exactly what to remove, in priority order

Works on exported conversation JSON (Claude, ChatGPT, any OpenAI-compatible format).

#### Why it doesn't exist
Observability tools track what agents do. Nobody tracks the quality of the context they're reasoning from. It's the difference between monitoring a car's engine vs. monitoring the quality of fuel in the tank.

#### Animus integration
Context Hygiene runs as a **Core maintenance process** — periodically audits Animus's own active context and proposes pruning via the proposal system. Animus's context quality compounds over time because it monitors itself.

```python
# core/self/context_hygiene.py
class ContextHygieneMonitor:
    def audit(self, conversation_history: list[dict]) -> HygieneReport:
        """Analyze context quality. Returns staleness score + pruning plan."""

    def auto_prune(self, report: HygieneReport, threshold: float = 0.7) -> list[dict]:
        """Remove dead weight above staleness threshold. Returns cleaned history."""
```

#### Directory Structure
```
context-hygiene/
├── README.md
├── pyproject.toml
├── context_hygiene/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── parsers/
│   │   ├── claude.py          ← parse claude.ai export format
│   │   ├── openai.py          ← parse ChatGPT/OpenAI format
│   │   └── generic.py         ← generic message list format
│   ├── analyzer/
│   │   ├── staleness.py       ← score how outdated each context segment is
│   │   ├── contradictions.py  ← find conflicting instructions
│   │   ├── compression.py     ← identify compressible threads
│   │   └── deadweight.py      ← find zero-influence content
│   └── report/
│       └── formatter.py
├── prompts/
│   ├── score_staleness.md
│   ├── find_contradictions.md
│   ├── identify_compression.md
│   └── generate_pruning_plan.md
└── tests/
    └── fixtures/
        ├── fresh_context.json
        ├── stale_context.json
        └── contradictory_context.json
```

#### Prompts

**score_staleness.md**
```
You receive an AI conversation history.
Score each segment (group of related messages) for staleness:
- 0.0 = fully relevant to current task
- 0.5 = partially relevant, some outdated elements
- 1.0 = completely stale, no current relevance

A segment is stale if:
- The task it addressed is complete or abandoned
- The instructions it established have been superseded
- The context it provided is no longer relevant to active goals
- It contains resolved errors or debugging that's no longer needed

Return JSON:
{
  "segments": [
    {
      "segment_id": integer,
      "messages": [start_index, end_index],
      "staleness_score": 0.0-1.0,
      "reason": "why this is stale or fresh"
    }
  ],
  "overall_staleness": 0.0-1.0,
  "fresh_percentage": 0-100
}
```

**find_contradictions.md**
```
You receive an AI conversation history.
Find instructions, constraints, or facts that contradict each other.

A contradiction is when:
- An earlier instruction says "always do X" and a later one says "never do X"
- An earlier fact is later corrected but both versions remain in context
- Competing constraints that cannot both be satisfied exist simultaneously

Return JSON:
{
  "contradictions": [
    {
      "type": "instruction|fact|constraint",
      "first_occurrence": message_index,
      "second_occurrence": message_index,
      "description": "plain English description of the conflict",
      "resolution": "which version should win and why"
    }
  ]
}
```

**generate_pruning_plan.md**
```
You receive a conversation context analysis including staleness scores,
contradictions, and dead weight identification.

Generate a specific, ordered pruning plan:
1. What to remove entirely (stale > 0.8)
2. What to compress into a summary (stale 0.4-0.8)
3. What contradiction to resolve (which version to keep)
4. What to keep unchanged (stale < 0.4)

For each action: specify exact message indices and the action to take.
Estimate context reduction: what percentage of tokens will be saved.

Return JSON:
{
  "actions": [
    {
      "priority": integer,
      "action": "remove|compress|resolve|keep",
      "messages": [start_index, end_index],
      "description": "what to do and why"
    }
  ],
  "estimated_reduction_pct": integer,
  "estimated_quality_improvement": "low|medium|high"
}
```

#### CLI Interface
```bash
context-hygiene audit conversation.json         # full audit report
context-hygiene audit conversation.json --prune # audit + generate cleaned version
context-hygiene score conversation.json         # staleness score only (fast)
context-hygiene clean conversation.json --threshold 0.7  # auto-prune above threshold
context-hygiene watch --model claude            # monitor active session (Pro)
```

#### Revenue Model
- Free: 10 audits/month, JSON input
- Pro: $12/mo — unlimited, live session monitoring, API
- API: $0.02/audit — for agent pipelines that self-monitor context quality

#### Distribution
- r/LocalLLaMA: "I built a tool that tells you when your AI conversation has gone stale and exactly what to prune"
- Claude/ChatGPT power user communities
- HN: "Context window quality is the hidden variable in AI productivity"

#### Claude Code Build Instructions
```
1. Initialize arete-tools/context-hygiene
2. Install: anthropic typer pydantic-settings rich
3. Create test fixtures: fresh_context.json, stale_context.json, contradictory_context.json
4. Build parsers/claude.py — parse claude.ai conversations.json export
5. Build analyzer/staleness.py — test against stale fixture, verify score > 0.7
6. Build analyzer/contradictions.py — test against contradictory fixture
7. Build analyzer/compression.py — identify threads that summarize to <20% original length
8. Build analyzer/deadweight.py — find messages with zero forward influence
9. Build report/formatter.py — rich terminal output with pruning priority list
10. Wire cli.py: audit, score, clean commands
11. Test full pipeline: stale conversation → audit → clean → verify quality improvement
```

---

### Tool B: Prompt Debt Tracker
**Version control and outcome tracking for prompts**  
**Repo:** `arete-tools/prompt-debt`  
**Priority:** Build after Autopsy ships

#### Problem
Prompts are the most important engineering artifact in an AI system — and the least managed. There is no git for prompts. Developers change a system prompt, things break, they can't remember what it said before. They run A/B tests mentally without any structure. They have no idea which prompt version produced which quality of output. Accumulated prompt changes create "prompt debt" — an invisible accumulation of technical debt that degrades system performance over time.

#### What it does
Tracks prompt versions the way git tracks code:
- Version every prompt change with a hash, timestamp, and commit message
- Link prompt versions to outcome quality scores (manual or automated)
- Diff any two prompt versions side by side
- Roll back to any previous version in one command
- Show performance trends across prompt versions
- Identify when a prompt change caused a quality regression

Additionally surfaces "prompt debt" — patterns of accumulated changes that create fragility:
- Contradictory instructions added over time
- Patches applied to work around earlier bad decisions
- Unused instructions that bloat the prompt
- Prompts that have drifted from their original intent

#### Why it doesn't exist
LangSmith has basic prompt versioning but no outcome correlation, no debt detection, and no rollback. It's a storage tool, not a management tool. No one has built the git-for-prompts that developers actually need.

#### Animus integration
Prompt Debt runs as a **Forge native tool** — every Forge prompt is automatically versioned. Quality gate scores from Signal feed back as outcome metrics per prompt version. Animus can detect when a prompt change in Forge caused a quality regression and propose a rollback.

```yaml
# forge/config.yaml
prompt_versioning:
  enabled: true
  backend: prompt-debt
  auto_score_from: signal  # use Signal scores as outcome metrics
  regression_threshold: -0.2  # flag if score drops more than 20%
```

#### Directory Structure
```
prompt-debt/
├── README.md
├── pyproject.toml
├── prompt_debt/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── store/
│   │   ├── versions.py        ← SQLite: prompt version history
│   │   ├── outcomes.py        ← SQLite: quality scores per version
│   │   └── diff.py            ← generate human-readable prompt diffs
│   ├── analyzer/
│   │   ├── debt.py            ← detect prompt debt patterns
│   │   ├── regression.py      ← identify quality-degrading changes
│   │   └── trends.py          ← performance trends across versions
│   └── report/
│       └── formatter.py
├── prompts/
│   ├── detect_debt.md
│   └── analyze_regression.md
└── tests/
```

#### Prompts

**detect_debt.md**
```
You receive the full version history of a prompt, including all changes
made over time and their associated quality scores.

Identify prompt debt — patterns of accumulated changes that create fragility:

CONTRADICTION_DEBT: Instructions added at different times that conflict
PATCH_DEBT: Changes applied to work around earlier poor decisions rather than fixing them
BLOAT_DEBT: Instructions that are never triggered or no longer relevant
DRIFT_DEBT: The prompt has changed so much from its original intent it needs a rewrite

Return JSON:
{
  "debt_items": [
    {
      "type": "CONTRADICTION|PATCH|BLOAT|DRIFT",
      "severity": "low|medium|high",
      "description": "plain English description",
      "location": "which part of the prompt",
      "recommendation": "specific fix"
    }
  ],
  "total_debt_score": 0.0-1.0,
  "recommended_action": "monitor|refactor|rewrite"
}
```

**analyze_regression.md**
```
You receive two prompt versions and their associated quality scores.
The quality dropped between version A and version B.

Identify: what change between these versions most likely caused the quality drop?

Consider:
- Instructions removed that were load-bearing
- Instructions added that conflict with existing ones
- Tone or style changes that reduced clarity
- Scope changes that confused the model's focus

Return JSON:
{
  "likely_cause": "plain English description of what caused regression",
  "culprit_change": "the specific text change most likely responsible",
  "confidence": 0.0-1.0,
  "fix": "specific recommendation to restore quality"
}
```

#### CLI Interface
```bash
# Version management
prompt-debt init my-prompt.md              # start tracking a prompt
prompt-debt commit my-prompt.md -m "add tone instruction"
prompt-debt log my-prompt.md               # version history
prompt-debt diff my-prompt.md v1 v3        # diff two versions
prompt-debt rollback my-prompt.md v2       # restore previous version

# Outcome tracking
prompt-debt score my-prompt.md --version v3 --score 8.2
prompt-debt score my-prompt.md --from-signal output.json  # auto-score via Signal

# Analysis
prompt-debt debt my-prompt.md              # analyze prompt debt
prompt-debt trends my-prompt.md            # quality trends across versions
prompt-debt regression my-prompt.md        # identify which change caused drop
```

#### Revenue Model
- Free: 3 prompts tracked, 30-day history, CLI only
- Pro: $15/mo — unlimited prompts, 1yr history, team sharing, API
- Team: $20/seat/mo — org-wide prompt registry, regression alerts

#### Distribution
- LangChain/CrewAI communities (they already think about prompt management)
- HN: "I built git for prompts — versions, diffs, rollbacks, and debt detection"
- Dev.to technical post: "How prompt debt silently kills your AI system quality"

#### Claude Code Build Instructions
```
1. Initialize arete-tools/prompt-debt
2. Install: anthropic typer rich pydantic-settings
3. Build store/versions.py — SQLite schema: prompts, versions, commits
4. Build store/outcomes.py — quality scores linked to version hashes
5. Build store/diff.py — line-level diff with color output via rich
6. Build analyzer/debt.py — test against a prompt with known debt patterns
7. Build analyzer/regression.py — test with two versions where quality dropped
8. Build analyzer/trends.py — chart quality over version history
9. Wire cli.py: init, commit, log, diff, rollback, score, debt, trends, regression
10. Integration test: init → commit 3 versions → score each → detect regression
11. Build Signal integration: prompt-debt score --from-signal reads Signal JSON output
```

---

### Tool C: Agent Specification Linter
**Static analysis for agent configs before deployment**  
**Repo:** `arete-tools/agentlint`  
**Priority:** Build after Autopsy ships (similar buyer)

#### Problem
Developers deploy agent configurations without any static analysis. Common failure patterns — goal definitions that guarantee necrosis, context structures that guarantee overflow, tool configurations that guarantee hallucination — are invisible until the agent fails in production. No tool catches these before they run.

This is different from Scaffold (which is a specification language — a research problem). Agentlint is a linter — it takes existing configs in any format and finds problems using known failure patterns from the Autopsy taxonomy.

#### What it does
Analyzes agent system prompts, workflow configs, and tool definitions before deployment:
- Detects configuration patterns that correlate with each Autopsy failure type
- Flags ambiguous goal definitions (goal_necrosis risk)
- Identifies context structure issues (context_overflow risk)
- Catches tool definition problems (tool_hallucination risk)
- Scores overall deployment readiness (0-10)
- Generates specific pre-deployment fixes

Works on raw system prompts, LangChain configs, CrewAI YAML, Forge YAML.

#### Why it doesn't exist
Autopsy is post-mortem (after failure). Agentlint is pre-mortem (before deployment). The failure taxonomy from Autopsy becomes the rule set for Agentlint. They share core logic — Agentlint is Autopsy applied proactively.

#### Animus integration
Agentlint runs as a **Forge pre-deployment gate** — every new workflow YAML is linted before its first execution. Animus cannot deploy a workflow that fails the linter above a configurable severity threshold.

```yaml
# forge/gates/agentlint.yaml
name: pre-deployment-lint
trigger: before_first_execution
analyzer:
  module: tools.agentlint.core.analyzer
  config:
    fail_on_severity: high     # block deployment on high-severity issues
    warn_on_severity: medium   # warn but allow on medium
budget:
  max_tokens: 800
  max_cost_usd: 0.02
```

#### Directory Structure
```
agentlint/
├── README.md
├── pyproject.toml
├── agentlint/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── parsers/
│   │   ├── system_prompt.py   ← raw text system prompt
│   │   ├── langchain.py       ← LangChain agent config
│   │   ├── crewai.py          ← CrewAI YAML config
│   │   └── forge.py           ← Animus Forge YAML config
│   ├── rules/
│   │   ├── goal_rules.py      ← goal_necrosis, goal_cancer patterns
│   │   ├── context_rules.py   ← context_overflow, context_poisoning patterns
│   │   ├── tool_rules.py      ← tool_hallucination, tool_loop patterns
│   │   ├── reasoning_rules.py ← overconfidence, premature_termination patterns
│   │   └── security_rules.py  ← injection_risk, identity_risk patterns
│   ├── core/
│   │   ├── analyzer.py        ← orchestrates all rule checks
│   │   └── scorer.py          ← deployment readiness score
│   └── report/
│       └── formatter.py
├── prompts/
│   ├── analyze_goal_definition.md
│   ├── analyze_context_structure.md
│   ├── analyze_tool_definitions.md
│   └── generate_lint_report.md
└── tests/
    └── fixtures/
        ├── goal_necrosis_risk.yaml
        ├── context_overflow_risk.yaml
        ├── tool_hallucination_risk.yaml
        └── clean_config.yaml
```

#### Rule Set (maps directly to Autopsy taxonomy + security layer)

```python
# rules/goal_rules.py

GOAL_NECROSIS_PATTERNS = [
    "goal has no completion criteria",        # open-ended without stopping condition
    "goal references external state that may become invalid",
    "goal has no timeout or iteration limit",
    "success condition is unmeasurable",
]

GOAL_CANCER_PATTERNS = [
    "subgoal has no resource ceiling",
    "recursive goal definition without base case",
    "goal expansion not bounded by parent goal scope",
]

# rules/security_rules.py

INJECTION_RISK_PATTERNS = [
    "agent ingests web content without sanitization",   # prompt injection vector
    "agent reads user-supplied files directly into context",
    "external API response inserted into system prompt",
    "no content scanning before agent context ingestion",
    "agent can be instructed via external document",
]

IDENTITY_RISK_PATTERNS = [
    "agent uses shared API key with other agents",      # no individual identity
    "agent-to-agent writes are unsigned",               # no write verification
    "agent has write access outside its declared domain",
    "no anomaly detection on agent write frequency",
]
```

#### Prompts

**analyze_goal_definition.md**
```
You receive an agent goal definition (from system prompt or config).
Identify patterns that predict goal-related failures:

GOAL_NECROSIS risk: Does the goal have clear completion criteria?
Can the agent know when it's done? Does it reference state that
could become invalid (URLs, APIs, file paths, external data)?

GOAL_CANCER risk: Could any subgoal expand to consume all resources?
Are there recursive patterns without base cases? Are there unbounded
search or iteration operations?

GOAL_AUTOIMMUNITY risk: Could the goal definition conflict with
the agent's operational context? Are there constraints that might
make the goal unachievable by design?

Return JSON:
{
  "risks": [
    {
      "failure_type": "goal_necrosis|goal_cancer|goal_autoimmunity",
      "severity": "low|medium|high",
      "description": "specific pattern found",
      "location": "which part of the goal definition",
      "fix": "specific change to make"
    }
  ]
}
```

**generate_lint_report.md**
```
You receive analysis results from goal, context, tool, and reasoning checks.
Generate a deployment readiness report.

DEPLOYMENT READINESS: X/10
[Score explanation]

BLOCKING ISSUES (must fix before deployment)
[High severity items with specific fixes]

WARNINGS (should fix, won't block)
[Medium severity items]

NOTES (low severity, consider improving)
[Low severity items]

SUMMARY: ready_to_deploy | fix_required | rewrite_recommended
```

#### CLI Interface
```bash
agentlint check system-prompt.txt          # lint a raw system prompt
agentlint check agent-config.yaml          # lint a config file
agentlint check --framework crewai crew.yaml
agentlint check --framework forge workflow.yaml
agentlint check --fail-on high             # exit 1 on high severity (CI/CD)
agentlint report agent-config.yaml        # full report with fixes
agentlint rules                            # show all lint rules and descriptions
```

#### Revenue Model
- Free: 10 lints/month, terminal output
- Pro: $15/mo — unlimited, custom rules, API, CI/CD integration
- API: $0.03/lint — for deployment pipelines

#### Distribution
- Same audience as Autopsy — LangChain/CrewAI communities
- Natural pairing: "Ship Autopsy for post-mortem, Agentlint for pre-mortem"
- HN: "I built a linter that catches agent failure patterns before you deploy"
- Bundle with Autopsy: "Agentlint → deploy → Autopsy" is a complete failure management workflow

#### Claude Code Build Instructions
```
1. Initialize arete-tools/agentlint
2. Install: anthropic typer pydantic-settings rich
3. Import Autopsy failure taxonomy as rule definitions (shared core logic)
4. Create test fixtures: one per failure type risk (goal_necrosis, etc.)
5. Build rules/goal_rules.py — test against goal_necrosis_risk.yaml fixture
6. Build rules/context_rules.py — test against context_overflow_risk.yaml
7. Build rules/tool_rules.py — test against tool_hallucination_risk.yaml
8. Build core/analyzer.py — orchestrate all rule checks, collect results
9. Build core/scorer.py — 0-10 deployment readiness score from results
10. Build report/formatter.py — blocking / warnings / notes structure
11. Wire cli.py: check, report, rules commands with --framework and --fail-on flags
12. Validate: clean_config.yaml scores 9+, each risk fixture scores < 6
13. Build Autopsy bridge: agentlint check can import Autopsy pattern library
```

---

## Part 5: Complete Tool Portfolio Summary

| Tool | Type | Status | Priority | Revenue Path |
|------|------|--------|----------|--------------|
| Verdict | Standalone + Core | Scaffolded | 1 | Free → $8/mo |
| Signal | Standalone + Forge | Spec complete | 2 | Free → $15/mo |
| Autopsy | Standalone + Forge | Spec complete | 3 | Free → $20/mo |
| Context Hygiene | Standalone + Core | This doc | 4 | Free → $12/mo |
| Prompt Debt | Standalone + Forge | This doc | 5 | Free → $15/mo |
| Agentlint | Standalone + Forge | This doc | 6 | Free → $15/mo |
| Calibrate | Standalone + Core | Spec complete | 7 | Free → $12/mo |
| Provenance | Standalone + Core | Spec complete | 8 | Free → $25/mo |
| Tenure | Platform | Future | 9 | $30/seat/mo |

### Suite Tiers (when 4+ tools live)

**Developer:** $39/mo — all standalone tools, unlimited usage  
**Team:** $25/seat/mo — all tools + team features + shared dashboards  
**Animus Pro:** $49/mo — all tools natively integrated in Animus, self-monitoring

---

## Part 6: The Honest One-Line Answer

**Where Animus stands:** At the intersection of four markets (agent orchestration, personal AI, observability tooling, local/sovereign AI) with no direct competitor in that intersection.

**Where it's similar:** Forge resembles CrewAI. Core resembles OpenClaw with better privacy.

**Where it's unique:** Stigmergic coordination (Quorum), self-improving runtime (Phase 1b), and the Arete Tools feedback loop — none of these exist in any shipping product.

**The risk:** OpenClaw has momentum. CrewAI is closing feature gaps. Speed of execution is the moat right now. Ship Core before OpenClaw fixes its security problems.
