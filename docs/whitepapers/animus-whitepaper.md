# Animus: A Three-Layer Architecture for Personal AI Sovereignty

**Whitepaper v2.0 — February 2026**
**Author: ARETE (AreteDriver)**

---

## Abstract

Animus is a personal AI system designed around three principles: persistent identity across sessions and devices, cost-visible multi-agent orchestration, and supervisor-free parallel coordination. Unlike chatbot wrappers that add memory to a language model, Animus separates the concerns of user interaction, workflow execution, and agent coordination into three distinct layers — Core, Forge, and Swarm — each independently useful and collectively powerful.

This paper describes the architecture, the coordination protocol based on biological stigmergy, the budget-first execution model drawn from lean manufacturing, and the initial production deployment: an autonomous media engine producing ~480 videos/month across three YouTube channels in eight languages.

---

## 1. The Problem

Current personal AI tools suffer from three architectural failures:

**Amnesia.** Language models have no persistent memory. Every conversation starts from zero. Users repeat context, re-explain preferences, and re-establish rapport. Solutions like ChatGPT's memory and Claude's memory are steps forward but remain shallow — they store facts, not understanding. There is no episodic memory (what happened), no procedural memory (how to do things), and no semantic memory (what things mean in your specific context).

**Invisible cost.** Multi-agent frameworks like LangChain, CrewAI, and AutoGen execute workflows without surfacing what they cost until the bill arrives. Agents spin up, consume tokens, retry on failure, and burn budget with no visibility. In manufacturing, this would be unacceptable — you don't run a production line without knowing the cost per unit. AI workflows deserve the same discipline.

**Supervisor bottleneck.** The default pattern for coordinating multiple agents is a supervisor: one agent watches all others, decides what runs next, resolves conflicts. This is expensive (the supervisor consumes tokens proportional to the work it monitors) and fragile (if the supervisor fails, everything stops). It's also architecturally lazy — it centralizes intelligence rather than distributing it.

Animus addresses all three with a layered architecture where each layer solves exactly one problem.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                     ANIMUS CORE                           │
│                                                           │
│  Persistent memory (episodic, semantic, procedural)       │
│  User identity and preferences                            │
│  Multi-device interface (CLI, voice, desktop, mobile)     │
│  Browser automation adapter                               │
│                                                           │
│  The layer you talk to.                                   │
├──────────────────────────────────────────────────────────┤
│                     ANIMUS FORGE                          │
│                                                           │
│  Declarative YAML workflows                               │
│  Agent archetypes (researcher, writer, reviewer, etc.)    │
│  Token budget management with hard ceilings               │
│  Quality gates with human-in-loop option                  │
│  SQLite checkpoint/resume                                 │
│  Provider-agnostic LLM interface                          │
│                                                           │
│  The layer that gets things done.                         │
├──────────────────────────────────────────────────────────┤
│                     ANIMUS SWARM                          │
│                                                           │
│  Shared intent graph                                      │
│  Stability scoring (0.0–1.0)                              │
│  Per-agent intent resolver                                │
│  Conflict resolution via stability precedence             │
│  No inter-agent messaging                                 │
│                                                           │
│  The layer that keeps agents aligned.                     │
└──────────────────────────────────────────────────────────┘
```

Each layer communicates only with its adjacent layers. Core never talks to Swarm directly. Swarm never receives user input. Forge is the sole bridge between user intent and agent coordination.

---

## 3. Animus Core: Identity, Memory, Interface

Core is the user-facing layer. It has three responsibilities:

### 3.1 Persistent Memory

Three memory types, inspired by cognitive science:

**Episodic memory** stores what happened — conversations, events, decisions, outcomes. Implemented in SQLite with full-text search. "Last Tuesday we decided to use Redis for caching" is an episodic memory.

**Semantic memory** stores what things mean in the user's context — project knowledge, domain relationships, technical preferences. Implemented as vector embeddings in ChromaDB (or SQLite + numpy for the lightweight variant). "Forge uses YAML configs" is a semantic memory.

**Procedural memory** stores how to do things — learned workflows, repeated patterns, command sequences. Implemented as structured records in SQLite with pattern matching. "When the user asks to deploy, run tests first, then build, then push to Railway" is a procedural memory.

All memory is local-first. Data stays on the user's hardware unless explicitly configured for cloud sync.

### 3.2 User Identity

Core maintains a persistent user profile: name, preferences, communication style, active projects, current goals. This profile informs every interaction — Core knows to be precise and direct, to use TPS analogies, to push back on bad ideas.

### 3.3 Multi-Device Interface

Core presents the same identity and memory across devices:

| Interface | Implementation | Use Case |
|-----------|---------------|----------|
| CLI | Python Click app | Primary development interface |
| Voice | Whisper (STT) + TTS | Hands-free, CarPlay |
| Desktop | Tauri or Electron | Dashboard, notification center |
| Mobile | React Native or PWA | On-the-go queries |
| Browser | Playwright adapter | Web automation for Forge agents |

All interfaces call the same Core API. Memory and identity are shared across all.

### 3.4 Browser Adapter

An interface adapter that gives Forge agents the ability to interact with web pages: navigate, fill forms, upload files, solve CAPTCHAs (via external services), and extract content. Built on Playwright with persistent session management and anti-detection measures. Used as a fallback when platform APIs are unavailable, and for interacting with web-only interfaces like some 3D printer dashboards.

---

## 4. Animus Forge: Orchestration

Forge is the execution layer. It receives structured task definitions from Core and coordinates agents to complete them.

### 4.1 Situation-Driven Design

Forge has no hardcoded pipelines. Every workflow is defined in a YAML situation config:

```yaml
name: story_fire_episode
description: Produce one Story Fire folklore episode

agents:
  - name: researcher
    archetype: researcher
    budget_tokens: 5000
    outputs: [content_brief]

  - name: scriptwriter
    archetype: writer
    budget_tokens: 8000
    inputs: [researcher.content_brief]
    outputs: [script]

  - name: voice
    archetype: producer
    budget_tokens: 1000
    inputs: [scriptwriter.script]
    outputs: [audio_file]
    tool: tts_engine

  - name: reviewer
    archetype: reviewer
    budget_tokens: 3000
    inputs: [scriptwriter.script, voice.audio_file]
    outputs: [review_decision]

gates:
  - name: quality_check
    after: reviewer
    type: automated
    pass_condition: review_decision.score >= 0.8
    on_fail: revise  # Send back to scriptwriter with feedback
```

Adding a new workflow means writing a new YAML file. No new code.

### 4.2 Budget-First Execution

Every agent has a token budget. Every workflow has a cost ceiling. This is non-negotiable.

The inspiration is the Toyota Production System's concept of making waste visible. In a lean manufacturing line, every station tracks cycle time, defect rate, and material cost. Nobody runs a process without knowing what it costs per unit. Forge applies the same discipline to AI workflows.

```
BUDGET TRACKING (real-time during execution)
├── researcher:   2,340 / 5,000 tokens (47%)     $0.035
├── scriptwriter: 6,120 / 8,000 tokens (77%)     $0.092
├── voice:           0 / 1,000 tokens  (0%)      $0.000
├── reviewer:        0 / 3,000 tokens  (0%)      $0.000
├── TOTAL:        8,460 / 17,000 tokens (50%)     $0.127
└── CEILING:                                      $0.500
```

If an agent exhausts its budget, Forge stops it. If the workflow ceiling is reached, Forge halts the entire pipeline and reports what completed and what didn't. No surprise bills.

### 4.3 Quality Gates

Checkpoints between pipeline stages where output is validated before proceeding:

- **Automated gates:** Score-based (does the review pass 0.8 threshold?), format-based (is the output valid JSON?), length-based (is the script within word count?).
- **Human-in-the-loop gates:** Core presents the output to the user for approval. Used for marketing content, financial decisions, or anything with reputation risk. The Marketing Engine's weekly batch approval is a human-in-the-loop gate.
- **On failure:** Revise (loop back to a previous agent with feedback), skip (proceed without this step), halt (stop the workflow and notify).

### 4.4 Checkpoint/Resume

All Forge workflow state persists to SQLite. If a pipeline fails at step 4 of 6, it restarts at step 4 with all prior outputs intact. No wasted compute. No re-running expensive steps.

This also enables pause/resume — you can pause a workflow before bed and resume it in the morning.

### 4.5 Provider Agnosticism

LLM calls go through a shared interface. The provider is configured per-agent or per-workflow:

```yaml
# Use Claude for creative work, Ollama for cheap extraction
agents:
  - name: scriptwriter
    provider: claude
    model: claude-sonnet-4-5
  - name: metadata_extractor
    provider: ollama
    model: llama3.1:8b
```

Swap providers without touching agent logic.

---

## 5. Animus Swarm: Stigmergic Coordination

Swarm is the coordination layer. It prevents parallel agents from colliding without requiring a supervisor.

### 5.1 The Problem with Supervisors

The standard approach to multi-agent coordination is a supervisor agent that receives all outputs, decides what runs next, and resolves conflicts. This has three problems:

1. **Token cost:** The supervisor must read every agent's output to make decisions. For n agents, the supervisor processes O(n) outputs per decision cycle. Over a full workflow, supervision cost scales linearly with workflow complexity.

2. **Single point of failure:** If the supervisor halts, stalls, or makes a poor decision, the entire workflow is affected. There's no redundancy.

3. **Bottleneck:** All decisions flow through one entity. Agents that could proceed independently must wait for the supervisor to process earlier agents' outputs first.

### 5.2 Stigmergy: Coordination Through Environment

Swarm replaces the supervisor with stigmergy — a coordination mechanism observed in ant colonies, termite mounds, and bird flocks. Instead of communicating with each other, agents communicate through a shared environment.

In ant colonies, ants deposit pheromones on the ground. Other ants read the pheromone trails and adjust their behavior. No ant gives orders to another ant. The colony's intelligence emerges from simple environmental signals.

In Animus Swarm, the "pheromone" is a shared intent graph — a data structure where agents declare what they intend to do, what they've decided, and how confident they are.

### 5.3 Intent Graph

Each agent, before taking an action, reads the intent graph and writes its own intentions:

```json
{
  "agent": "scriptwriter",
  "action": "write_opening_scene",
  "category": "narrative",
  "provides": ["opening_scene_draft"],
  "requires": ["content_brief"],
  "constraints": ["tone:mythological", "length:500-800_words"],
  "stability": 0.85,
  "evidence": ["content_brief analyzed", "3 reference myths reviewed"],
  "files_affected": ["output/script.md"],
  "timestamp": "2026-02-18T14:30:00Z"
}
```

**Stability** is the key field. It ranges from 0.0 (pure speculation) to 1.0 (committed and verified). An agent with stability 0.3 is still exploring. An agent with stability 0.9 has high confidence in its decision. Other agents treat high-stability declarations as constraints to respect.

### 5.4 Intent Resolver

Before each decision, an agent runs the intent resolver:

```
1. Read all entries in the intent graph
2. Identify high-stability decisions from other agents
3. Check for conflicts with my intended action
4. If conflict exists:
   a. If other agent's stability > my stability → adjust my plan
   b. If my stability > other agent's stability → proceed, they'll adjust
   c. If equal → wait one cycle, re-evaluate
5. Write my intent to the graph
6. Execute my action
```

This creates emergent coordination. No agent tells another what to do. Each agent independently reads the shared state and self-adjusts. The result is the same — agents don't collide — but the mechanism is fundamentally different from supervision.

### 5.5 Complexity Analysis

| Pattern | Messages per cycle | Failure mode |
|---------|-------------------|--------------|
| Sequential | 0 (no parallelism) | Slow |
| Supervisor | O(n) reads + O(n) writes to supervisor | SPOF |
| Full mesh | O(n²) inter-agent messages | Communication explosion |
| **Swarm** | **O(n) reads of shared graph** | **Graceful degradation** |

Swarm's communication cost is the same as a supervisor pattern (O(n) reads), but without the single point of failure. If one agent stalls, others continue — they simply read the graph and adjust around the gap.

---

## 6. Production Deployment: Media Engine

The first production workload exercising all three layers.

### 6.1 Overview

Three autonomous YouTube channels:

| Channel | Domain | Content Style |
|---------|--------|--------------|
| Story Fire | World folklore & mythology | Narrative retellings with AI-generated or archival visuals |
| New Eden Whispers | EVE Online lore | In-universe storytelling for the EVE community |
| Holmes Wisdom | Sherlock Holmes | Deduction lessons and wisdom extraction |

Each channel produces content in 8 languages. Total output: ~480 videos/month.

### 6.2 Pipeline

```
Core: Trigger daily production schedule
  ↓
Forge: Load situation config for channel + language
  ↓
Agent Pipeline:
  Research → Script → Voice (TTS) → Video Assembly → QA → Publish
  ↓ (Swarm coordinates parallel language versions)
Forge: Report results to Core
  ↓
Core: Log to memory, update analytics
```

### 6.3 Marketing Engine

A second Forge workflow that promotes the media content (and other projects) across five platforms:

- **Twitter/X, LinkedIn, Reddit** — text posts via API
- **YouTube** — full video uploads (Media Engine output) + Shorts (repurposed clips)
- **TikTok** — vertical clips via Content Posting API or browser fallback

Content sources: original drafts, Internet Archive public domain films (repurposed with narration), AI-generated video (Kling/Sora).

Weekly batch approval: Forge generates the full week's content on Monday, Core presents it for review, approved posts publish on schedule.

ROI tracking: UTM attribution on every link, cost tracking per post, funnel conversion measurement (Shorts → BenchGoblins signups, GitHub stars, channel subscribers).

---

## 7. Developer Tools Ecosystem

Animus's architecture produces standalone tools that are useful independently and serve as entry points to the full system:

| Tool | What It Does | Animus Connection |
|------|-------------|-------------------|
| claudemd-forge | Generates CLAUDE.md for AI coding agents | Generates context for Core |
| mcp-manager | Manages MCP servers across IDEs | Manages servers for Forge agents |
| agent-audit | Estimates workflow costs, lints configs | Validates Forge workflow configs |
| memboot | Instant project memory, serves as MCP server | IS the Core memory layer, extracted |
| ai-spend | Aggregates AI API costs across providers | Tracks Forge + Marketing Engine spend |

Each tool is free on PyPI with optional paid tiers. Users who adopt the standalone tools are pre-qualified for the full Animus system — they've already bought into the architectural thinking.

---

## 8. Design Principles

### 8.1 Make Cost Visible

Every token has a price. Every workflow has a budget. Every agent reports its consumption. Inspired by the Toyota Production System's principle that waste must be visible before it can be eliminated. Forge dashboards show real-time cost alongside progress — not after the fact.

### 8.2 No Supervisor

Swarm eliminates the supervisor pattern. Agents self-coordinate through environmental awareness. This is both cheaper (no supervision tokens) and more resilient (no single point of failure). The inspiration is biological: ant colonies, bird flocks, and fish schools coordinate complex behavior without centralized control.

### 8.3 Declarative Over Imperative

Workflows are YAML configs, not code. Agent archetypes are templates, not implementations. Adding a new capability means writing a config file, not modifying a codebase. This keeps Forge extensible without accumulating technical debt.

### 8.4 Checkpoint Everything

All state persists. Workflows resume from failure points. Nothing is lost to crashes, timeouts, or rate limits. This is table stakes for production systems but surprisingly rare in AI orchestration frameworks.

### 8.5 Local-First Sovereignty

Your memory, your identity, your hardware. Animus is designed to run on your machine, with cloud services (LLM APIs, CAPTCHA solvers) as external tools, not hosts. You own your data. You choose your providers.

### 8.6 Gateway Architecture

Standalone tools lead to the full system. Each free tool addresses one pain point. Users who outgrow the standalone tool find that Animus already solves their next problem. This is both a product strategy and an architectural principle — the tools are real extractions from Animus layers, not marketing gimmicks bolted on.

---

## 9. Technical Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python (primary), TypeScript (dashboard) | Python for ML/AI ecosystem. TypeScript for React dashboard. |
| State | SQLite | Zero-infrastructure persistence. Ships with Python. |
| Vector search | ChromaDB or SQLite + numpy | ChromaDB for full deployments. SQLite + numpy for memboot (no external deps). |
| Workflows | Declarative YAML | Human-readable, version-controllable, diff-friendly. |
| LLM interface | Provider-agnostic wrapper | Claude API, OpenAI, Ollama. Swap without touching agents. |
| Web framework | FastAPI | Forge ops dashboard, Core API. |
| Dashboard UI | React | Forge monitoring. Admin only, not user-facing. |
| Browser | Playwright | Async-native, headless, multi-browser. |
| Deployment | Docker + Docker Compose | Reproducible, portable. |
| Packaging | PyPI (tools), Docker (services) | Standard distribution channels. |

---

## 10. Current Status and Roadmap

### Built
- Architecture documentation (this whitepaper)
- CLAUDE.md files for all monorepos (Claude Code context)
- Media Engine situation configs
- Marketing Engine design (5 platforms, ROI tracking, AI video pipeline)
- Developer tools build specs with Claude Code prompts
- Ollama Modelfile for local LLM parity
- Browser automation architecture spec
- Swarm protocol specification with worked examples

### In Progress
- claudemd-forge v0.1.0 (built, needs Pro tier)
- BenchGoblins (RevenueCat configured, deployment in progress)

### Next
1. Forge Phase 1: FastAPI skeleton, agent runner, YAML loader
2. mcp-manager v0.1.0 (PyPI)
3. agent-audit v0.1.0 (PyPI)
4. Core memory layer (which becomes memboot)
5. Marketing Engine Phase 1 (content generation, manual posting)
6. Swarm implementation (intent graph, resolver, stability scoring)

---

## 11. Related Work

**LangChain / LangGraph:** Graph-based orchestration with broad tool integration. Animus Forge differs in its budget-first design and YAML-declarative workflows. LangGraph requires code to define graphs; Forge uses config files.

**CrewAI:** Role-based multi-agent framework. Similar agent archetype concept but without budget management, checkpoint/resume, or stigmergic coordination.

**AutoGen (Microsoft):** Conversational multi-agent framework. Strong on agent-to-agent dialogue but heavy on inter-agent messaging (O(n²) in complex workflows).

**Mem0 / Zep:** Memory layers for LLMs. Animus Core's memory system serves the same function but is integrated with the orchestration and coordination layers rather than being a standalone service.

**Conductor / Claude Squad:** Parallel agent execution tools. Address the parallelism problem but without Swarm's stigmergic coordination — they use git worktrees and manual conflict resolution.

Animus's contribution is not any single layer but the integration: persistent memory informing orchestration informed by supervisor-free coordination, with cost visibility throughout.

---

## 12. Conclusion

Animus is an architecture, not a product. Its value is in the separation of concerns: Core handles identity and memory, Forge handles execution and cost, Swarm handles coordination without centralization. Each layer can be adopted independently (memboot for memory, agent-audit for cost visibility) or as a unified system.

The initial deployment — autonomous media production at scale — demonstrates that the architecture handles real workloads. The developer tools ecosystem demonstrates that the architecture produces useful standalone artifacts. The lean manufacturing principles embedded in Forge demonstrate that AI workflows can be as disciplined as physical production lines.

The name Animus comes from the Latin for "mind, spirit, courage." The project's ambition is to be exactly that: an extension of the user's mind, with the spirit to act autonomously and the courage to make cost and complexity visible rather than hiding them behind abstractions.

---

*For implementation details, see the CLAUDE.md files in each monorepo. For build instructions, see the Developer Tools Build Specs. For the Marketing Engine design, see Marketing-Engine-Design.md.*

*Contact: github.com/AreteDriver*
