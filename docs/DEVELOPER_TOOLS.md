# Developer Tools Build Specs

> Five tools to build after CLAUDE.md Generator. Each follows the same pattern: underserved niche, CLI-first, PyPI published, free tier + paid upgrade path.

---

## Tool 1: MCP Server Manager

**Codename:** `mcp-manager`
**Tagline:** Manage your MCP servers like infrastructure, not post-it notes.
**Priority:** HIGH — Ship first. MCP adoption is exploding, management tooling is nonexistent.

### The Problem

Developers using Claude Code, Cursor, and other agentic IDEs accumulate MCP servers with no central view. No health checks, no version tracking, no way to know if a server is down until an agent fails mid-task. Configuration is scattered across JSON files, environment variables, and IDE settings.

### What It Does

```bash
# List all configured MCP servers across tools
mcp-manager list

# Health check all servers
mcp-manager health
# ✅ github-mcp       healthy   v1.2.3   latency: 45ms
# ✅ notion-mcp       healthy   v0.9.1   latency: 120ms
# ❌ slack-mcp        timeout   v1.0.0   last seen: 2h ago
# ⚠️ postgres-mcp     degraded  v0.8.0   latency: 890ms

# Test a specific server's capabilities
mcp-manager test github-mcp

# Check for updates across all servers
mcp-manager update --check

# Show which tools/IDEs use which servers
mcp-manager map
# Claude Code  → github-mcp, notion-mcp, slack-mcp
# Cursor       → github-mcp, postgres-mcp

# Add a new server to your config
mcp-manager add https://mcp.linear.app/sse --name linear-mcp

# Remove a server
mcp-manager remove slack-mcp

# Export config (portable across machines)
mcp-manager export > mcp-config.yaml

# Import config on new machine
mcp-manager import mcp-config.yaml
```

### Architecture

```
mcp-manager/
├── src/
│   ├── cli.py              ← Click-based CLI entry point
│   ├── discovery.py        ← Find MCP configs across Claude Code, Cursor, Windsurf, etc.
│   ├── health.py           ← Health check / latency probe per server
│   ├── registry.py         ← Central registry of configured servers
│   ├── updater.py          ← Version check against known MCP server repos
│   ├── mapper.py           ← Map servers → IDE/tool usage
│   └── exporters.py        ← YAML/JSON import/export
├── tests/
├── pyproject.toml
└── README.md
```

### Config Discovery Locations

| Tool | Config Path |
|------|------------|
| Claude Code | `~/.claude/mcp_servers.json` or project `.mcp.json` |
| Cursor | `~/.cursor/mcp.json` |
| Windsurf | `~/.windsurf/mcp_config.json` |
| Custom | `~/.mcp-manager/config.yaml` |

### Key Technical Decisions

- **Zero dependencies beyond stdlib + click + httpx** — keep it light like SteamProtonHelper
- **Config discovery is read-only** — never modify IDE configs without explicit `--write` flag
- **Health checks use MCP protocol's native ping** — not just HTTP, but actual MCP capability probing
- **Offline-safe** — if a server is unreachable, report it, don't crash

### Monetization

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | List, health, test, map, export/import |
| Pro | $5/mo | Auto-health-check daemon, Slack/Discord alerts on failures, usage analytics (which servers get called most), team config sync |

### Claude Code Build Prompts

**Prompt 1 — Foundation**
```
Create a Python CLI tool called mcp-manager using Click.

Project structure:
- src/mcp_manager/ package
- cli.py with Click group and subcommands: list, health, test, map, add, remove, export, import
- discovery.py that finds MCP server configs from:
  - ~/.claude/mcp_servers.json
  - Project-level .mcp.json files
  - ~/.cursor/mcp.json (if exists)
  - ~/.mcp-manager/config.yaml (our own registry)
- models.py with Pydantic models: MCPServer, HealthResult, ServerMap
- pyproject.toml with CLI entry point: mcp-manager → src.mcp_manager.cli:main

Use type hints everywhere. Tests in tests/ mirroring src structure.
Start with discovery.py and the list subcommand only.
```

**Prompt 2 — Health Checks**
```
Add health check functionality to mcp-manager:

- health.py: For each discovered MCP server, attempt connection
  - For URL-based servers (SSE): HTTP GET with timeout
  - For stdio-based servers: attempt to spawn process and read initial handshake
  - Record latency, status (healthy/degraded/timeout/error), last seen timestamp
- Update cli.py: `mcp-manager health` runs checks on all servers
  - Color-coded output: green ✅, yellow ⚠️, red ❌
  - Table format with columns: name, status, version, latency
- Add --json flag for machine-readable output
- Tests with mocked HTTP responses
```

**Prompt 3 — Map & Export**
```
Add server mapping and config portability:

- mapper.py: Cross-reference discovered configs to show which IDE uses which server
  - Output as a tree: IDE → [server1, server2, ...]
- exporters.py:
  - export: Dump all discovered servers to a unified YAML format
  - import: Read YAML and write to ~/.mcp-manager/config.yaml
  - Never write to IDE-specific configs without --write-to-ide flag
- Update cli.py with map, export, import subcommands
- Tests for round-trip export → import
```

**Prompt 4 — Add/Remove & Polish**
```
Add server management and final polish:

- cli.py add subcommand: add a new MCP server by URL or command
  - Auto-detect server type (SSE vs stdio)
  - Probe for capabilities on add
  - Write to ~/.mcp-manager/config.yaml
- cli.py remove subcommand: remove from our registry
- README.md: problem statement, install instructions, usage examples, GIF placeholder
- Publish to PyPI: verify pyproject.toml, build, test install
- Tag v0.1.0
```

### Estimated Build Time: 1-2 weekends

---

## Tool 2: Workflow Auditor / Agent Cost Calculator

**Codename:** `agent-audit`
**Tagline:** Know what your agents cost before you run them.
**Priority:** HIGH — Maps directly to FDE/AI Enablement roles. Portfolio differentiator.

### The Problem

Multi-agent workflows (LangChain, CrewAI, Forge, AutoGen) execute without cost visibility. Developers discover they burned $50 in tokens only after the workflow completes. No tool analyzes a workflow definition and estimates cost, execution time, and failure risk upfront.

### What It Does

```bash
# Analyze a Forge YAML workflow
agent-audit estimate configs/media_engine/story_fire.yaml
#
# WORKFLOW: Story Fire Episode Production
# ┌─────────────────────────────────────────────┐
# │ Agent           │ Est. Tokens │ Est. Cost   │
# ├─────────────────┼─────────────┼─────────────┤
# │ researcher      │ 4,200       │ $0.063      │
# │ scriptwriter    │ 8,500       │ $0.128      │
# │ reviewer        │ 3,100       │ $0.047      │
# │ publisher       │ 1,200       │ $0.018      │
# ├─────────────────┼─────────────┼─────────────┤
# │ TOTAL           │ 17,000      │ $0.256      │
# │ Est. Duration   │ ~45 seconds │             │
# └─────────────────────────────────────────────┘
#
# ⚠️ WARNINGS:
# - researcher has no token budget cap (unbounded cost risk)
# - No failure/retry path defined for publisher
# - scriptwriter + reviewer could run parallel (currently sequential)

# Audit an existing workflow for anti-patterns
agent-audit lint configs/media_engine/story_fire.yaml
#
# [WARN] researcher: no max_tokens specified
# [WARN] publisher: no on_failure handler
# [INFO] scriptwriter → reviewer: could be parallelized
# [PASS] all agents have defined outputs
# [PASS] budget total under $1.00 per run
#
# Score: 72/100

# Compare costs across providers
agent-audit compare configs/media_engine/story_fire.yaml --providers claude,openai,ollama

# Watch a running workflow's actual costs vs estimates
agent-audit watch --pid 12345
```

### Architecture

```
agent-audit/
├── src/
│   ├── cli.py              ← Click CLI
│   ├── parsers/
│   │   ├── forge.py        ← Parse Animus Forge YAML
│   │   ├── langchain.py    ← Parse LangChain/LangGraph configs
│   │   ├── crewai.py       ← Parse CrewAI configs
│   │   └── generic.py      ← Fallback for custom YAML
│   ├── estimator.py        ← Token estimation engine
│   ├── pricing.py          ← Provider pricing database (Claude, OpenAI, Ollama, etc.)
│   ├── linter.py           ← Anti-pattern detection rules
│   ├── reporter.py         ← Output formatting (table, JSON, markdown)
│   └── rules/
│       ├── budget.py        ← Budget-related rules
│       ├── resilience.py    ← Failure handling rules
│       ├── efficiency.py    ← Parallelization, deduplication rules
│       └── security.py      ← Secret exposure, scope creep rules
├── data/
│   └── pricing.yaml         ← Current provider pricing (auto-updatable)
├── tests/
├── pyproject.toml
└── README.md
```

### Lint Rules (Your TPS Brain, Encoded)

| Rule | Category | Severity |
|------|----------|----------|
| Agent has no token budget | Budget | WARN |
| Workflow has no total cost ceiling | Budget | ERROR |
| No on_failure handler for agent | Resilience | WARN |
| No checkpoint/resume configured | Resilience | WARN |
| Sequential agents with no data dependency could be parallel | Efficiency | INFO |
| Agent output not consumed by any downstream agent | Efficiency | WARN |
| API keys in workflow config (not in secrets) | Security | ERROR |
| Agent has write access to filesystem with no scope limit | Security | WARN |

### Monetization

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | Estimate, lint (basic rules), single provider pricing |
| Pro | $8/mo | Multi-provider comparison, custom rule definitions, CI integration (GitHub Action that audits on PR), team pricing overrides, historical cost tracking |

### Claude Code Build Prompts

**Prompt 1 — Foundation + Forge Parser**
```
Create a Python CLI tool called agent-audit using Click.

Structure:
- src/agent_audit/ package
- cli.py with subcommands: estimate, lint, compare
- parsers/forge.py: Parse Animus Forge YAML workflow files
  - Extract: agent names, archetypes, budget_tokens, inputs/outputs, gates
  - Return a WorkflowDefinition Pydantic model
- estimator.py: Given a WorkflowDefinition, estimate total tokens and cost
  - Use pricing from data/pricing.yaml (Claude, OpenAI, Ollama rates)
  - If agent has budget_tokens, use that as estimate
  - If not, estimate based on archetype defaults (researcher: 5000, writer: 8000, reviewer: 3000, publisher: 1000)
- reporter.py: Format output as ASCII table
- pyproject.toml with entry point: agent-audit

Include data/pricing.yaml with current per-token costs for:
- Claude Sonnet 4.5, Opus 4.5, Haiku 4.5
- GPT-4o, GPT-4o-mini
- Ollama (local, $0)

Tests for parser and estimator.
```

**Prompt 2 — Linter**
```
Add workflow linting to agent-audit:

- linter.py: Analyze WorkflowDefinition against rules
- rules/ directory with separate rule files:
  - budget.py: check for missing token limits, missing cost ceiling
  - resilience.py: check for missing on_failure, missing checkpoint config
  - efficiency.py: detect parallelizable sequential agents (no data dependency between them)
  - security.py: detect API keys in config, overly broad file access
- Each rule returns: severity (ERROR/WARN/INFO), message, suggestion
- cli.py lint subcommand: run all rules, output results, calculate score (0-100)
- Add --json flag for CI integration
- Tests for each rule category with passing and failing examples
```

**Prompt 3 — Multi-Provider Comparison + Polish**
```
Add provider comparison and final polish:

- cli.py compare subcommand: estimate same workflow across multiple providers
  - Table showing: provider, model, est. tokens, est. cost, est. speed
  - Highlight cheapest and fastest options
- Add --format flag: table, json, markdown
- README.md: problem statement, install, usage, example output, rule list
- Publish to PyPI
- Tag v0.1.0
```

### Estimated Build Time: 1-2 weekends

---

## Tool 3: Agent Memory Bootstrap

**Codename:** `memboot`
**Tagline:** Instant memory for any LLM. No infrastructure required.
**Priority:** MEDIUM — Powerful concept, takes longer to build, but feeds directly into Animus.

### The Problem

AI agents forget everything between sessions. Solutions like Mem0 and Zep require infrastructure setup (Redis, Postgres, hosted services). Most developers just want to say "remember my project context" without deploying a database.

### What It Does

```bash
# Bootstrap memory from your project
memboot init
# Scanning project...
# Found: README.md, CLAUDE.md, 12 Python files, 3 config files
# Indexed 847 chunks into ~/.memboot/myproject.db
# Memory ready. 

# Query memory from CLI (for testing)
memboot query "What's the database schema?"
# Found 3 relevant chunks:
# 1. [models.py:15-40] SQLAlchemy models for User, Post, Comment...
# 2. [CLAUDE.md:80-95] Database uses PostgreSQL with Alembic migrations...
# 3. [README.md:45-52] Schema diagram showing entity relationships...

# Add a conversation/decision to memory
memboot remember "We decided to use Redis for caching instead of memcached"

# Export memory as context for any LLM
memboot context --max-tokens 4000
# Outputs a formatted context block ready to paste into any LLM prompt

# Serve memory as an MCP server (!!!)
memboot serve
# MCP server running on stdio
# Compatible with Claude Code, Cursor, Windsurf
# Tools exposed: query_memory, remember, get_context

# Ingest external docs
memboot ingest ~/Documents/api-docs.pdf
memboot ingest https://docs.example.com/api

# Reset
memboot reset --confirm
```

### Architecture

```
memboot/
├── src/
│   ├── cli.py              ← Click CLI
│   ├── indexer.py          ← Scan project files, chunk, embed
│   ├── chunker.py          ← Smart chunking (respects function/class boundaries)
│   ├── embedder.py         ← Embedding generation (local or API)
│   ├── store.py            ← SQLite + numpy vector search (no external DB needed)
│   ├── query.py            ← Similarity search + reranking
│   ├── memory.py           ← Episodic memory (conversations, decisions)
│   ├── context.py          ← Export formatted context blocks
│   ├── mcp_server.py       ← Serve memory as MCP server
│   └── ingest/
│       ├── files.py        ← .py, .md, .yaml, .json, .txt
│       ├── pdf.py          ← PDF extraction
│       └── web.py          ← URL fetching + extraction
├── tests/
├── pyproject.toml
└── README.md
```

### Key Technical Decisions

- **SQLite + numpy for vector search** — No ChromaDB, no Pinecone, no external deps. Just sqlite3 (stdlib) and numpy for cosine similarity. This is the "zero infrastructure" promise.
- **Embedding options:** 
  - Local: sentence-transformers (if installed) — fully offline
  - API: OpenAI embeddings or Anthropic (if key available)
  - Fallback: TF-IDF with sklearn — works with zero API keys
- **MCP server mode** is the killer feature. Your memory becomes a tool any agent can use. This is the bridge between memboot (standalone tool) and Animus Core (full exocortex).
- **Chunking respects code structure** — splits on function/class boundaries, not arbitrary token counts.

### Monetization

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | Init, query, remember, context export, file ingestion |
| Pro | $8/mo | MCP server mode, PDF/web ingestion, multi-project memory, team shared memory, auto-reindex on file change (watch mode) |

### Connection to Animus

memboot is **Animus Core's memory layer extracted as a standalone tool.** Users who adopt memboot are pre-qualified for Animus. The upgrade path: "You're using memboot for project memory. Animus gives you persistent identity, multi-device access, and Forge orchestration on top of it."

### Claude Code Build Prompts

**Prompt 1 — Foundation + Indexer**
```
Create a Python CLI tool called memboot using Click.

Structure:
- src/memboot/ package
- cli.py with subcommands: init, query, remember, context, reset
- indexer.py: Scan current directory for supported files (.py, .md, .yaml, .json, .txt, .toml)
  - Skip: .git, __pycache__, node_modules, .env, binary files
  - Return list of FileContent(path, content, file_type)
- chunker.py: Split file contents into chunks
  - For Python: split on function/class boundaries (use ast module)
  - For Markdown: split on headers
  - For everything else: split on ~500 token windows with 50 token overlap
  - Return list of Chunk(content, source_file, start_line, end_line, chunk_type)
- store.py: SQLite-backed vector store
  - Table: chunks (id, content, source_file, start_line, end_line, embedding BLOB)
  - Table: memory (id, content, timestamp, type: 'episodic'|'decision')
  - Store in ~/.memboot/{project_hash}.db
  - Use numpy for cosine similarity search (no external vector DB)
- embedder.py: Generate embeddings
  - Try sentence-transformers first (if installed)
  - Fall back to TF-IDF with sklearn (always available after pip install)
  - Make embedding backend configurable

Start with init (scan + chunk + embed + store) and query (similarity search).
Include tests with a sample project fixture.
```

**Prompt 2 — Memory + Context Export**
```
Add episodic memory and context export:

- memory.py: 
  - remember(text): Store a decision/conversation in the memory table with timestamp
  - recall(query, n=5): Search episodic memories by similarity
- context.py:
  - get_context(query=None, max_tokens=4000): Build a formatted context block
  - If query provided: return most relevant chunks + memories
  - If no query: return project summary (README chunks + recent memories)
  - Format as markdown with source attribution
  - Respect max_tokens by truncating least relevant results
- Update cli.py: remember and context subcommands
- Tests for memory storage/retrieval and context formatting
```

**Prompt 3 — MCP Server**
```
Add MCP server mode to memboot:

- mcp_server.py: Implement MCP stdio server
  - Expose tools: query_memory, remember, get_context
  - query_memory(query: str, n: int = 5) → list of relevant chunks
  - remember(text: str) → confirmation
  - get_context(query: str = None, max_tokens: int = 4000) → formatted context
- cli.py serve subcommand: start the MCP server on stdio
- Add instructions to README for connecting to Claude Code and Cursor
- Test MCP protocol compliance
```

**Prompt 4 — Ingest + Polish**
```
Add external document ingestion and final polish:

- ingest/pdf.py: Extract text from PDFs (use pymupdf or pdfplumber)
- ingest/web.py: Fetch URL, extract main content (use trafilatura or readability)
- cli.py ingest subcommand: accept file path or URL
- README.md: problem statement, install, usage, MCP setup guide, comparison to Mem0/Zep
- Publish to PyPI
- Tag v0.1.0
```

### Estimated Build Time: 2-3 weekends

---

## Tool 4: YAML Workflow Linter

**Codename:** `workflow-lint`
**Tagline:** eslint for your agent pipelines.
**Priority:** MEDIUM-LOW — Overlaps heavily with agent-audit's lint functionality. Consider shipping as part of agent-audit instead of standalone.

### The Problem

Agent workflow YAML configs (Forge, LangChain, CrewAI) have no validation beyond "is this valid YAML?" Nobody catches structural issues like missing failure handlers, unbounded agents, circular dependencies, or security anti-patterns until runtime.

### What It Does

```bash
# Lint a workflow file
workflow-lint check pipeline.yaml
#
# pipeline.yaml
#   2:5  error    agent "researcher" has no token budget         budget/no-limit
#   4:3  warning  no on_failure handler for "writer"             resilience/no-fallback
#   8:1  info     agents "writer" and "reviewer" have no data    efficiency/parallelizable
#                 dependency — consider parallel execution
#  12:5  error    API key found in config (line 12)              security/exposed-secret
#
# ✖ 2 errors, 1 warning, 1 info

# Lint with auto-fix suggestions
workflow-lint check pipeline.yaml --fix
# Writes pipeline.yaml.fixed with suggested corrections

# Lint in CI (exit code 1 on errors)
workflow-lint check pipeline.yaml --strict

# List available rules
workflow-lint rules

# Create custom ruleset
workflow-lint init-rules > .workflow-lint.yaml
```

### Decision: Standalone vs. Part of agent-audit

**Recommendation: Ship the linter as part of agent-audit** (`agent-audit lint`), not as a separate tool. The user base is the same, the rules overlap, and two separate PyPI packages for the same audience fragments your distribution.

If the linting rules grow significantly (50+ rules, custom rule plugins, team config), then spin it out as a standalone tool later.

### Rules to Implement

See agent-audit's rules section above. Additional rules for standalone version:

| Rule | Category | Severity |
|------|----------|----------|
| Circular dependency between agents | Structure | ERROR |
| Agent output type doesn't match downstream input type | Structure | ERROR |
| Duplicate agent names | Structure | ERROR |
| Workflow has no entry point | Structure | ERROR |
| Gate has no pass/fail paths | Resilience | ERROR |
| More than 10 sequential agents (pipeline too long) | Efficiency | WARN |
| Agent uses deprecated model identifier | Compatibility | WARN |

### Estimated Build Time: 1 weekend (if standalone), 0 (if part of agent-audit)

---

## Tool 5: AI Spend Dashboard

**Codename:** `ai-spend`
**Tagline:** One dashboard for all your AI API costs.
**Priority:** LOW — Useful but competitive. OpenAI and Anthropic both have dashboards. The value-add is aggregation across providers.

### The Problem

Developers use multiple AI providers (Anthropic, OpenAI, Kling, ElevenLabs, etc.) and have no unified view of total spend. Each provider has its own dashboard with its own format. For someone running Animus Forge + Media Engine + Marketing Engine, costs are scattered across 5+ services.

### What It Does

```bash
# Configure providers
ai-spend config add anthropic --api-key sk-ant-...
ai-spend config add openai --api-key sk-...
ai-spend config add kling --api-key ...

# Current month summary
ai-spend summary
#
# AI SPEND — February 2026
# ┌──────────────┬──────────┬──────────┬──────────┐
# │ Provider     │ This Week│ MTD      │ Budget   │
# ├──────────────┼──────────┼──────────┼──────────┤
# │ Anthropic    │ $12.40   │ $34.80   │ $50.00   │
# │ OpenAI       │ $3.20    │ $8.90    │ $20.00   │
# │ Kling        │ $5.60    │ $18.40   │ $30.00   │
# │ ElevenLabs   │ $2.10    │ $6.30    │ $15.00   │
# ├──────────────┼──────────┼──────────┼──────────┤
# │ TOTAL        │ $23.30   │ $68.40   │ $115.00  │
# └──────────────┴──────────┴──────────┴──────────┘
# ⚠️ Anthropic at 70% of monthly budget

# Set budget alerts
ai-spend budget set anthropic 50.00
ai-spend budget set --total 115.00

# Daily cost breakdown
ai-spend daily --last 7

# Export for the ROI dashboard
ai-spend export --format json --range 2026-02-01:2026-02-18
```

### Architecture

```
ai-spend/
├── src/
│   ├── cli.py
│   ├── providers/
│   │   ├── anthropic.py    ← Fetch usage from Anthropic API
│   │   ├── openai.py       ← Fetch usage from OpenAI API
│   │   ├── kling.py        ← Fetch usage from Kling API
│   │   ├── elevenlabs.py   ← Fetch usage from ElevenLabs API
│   │   └── manual.py       ← Manual cost entry for providers without APIs
│   ├── store.py            ← SQLite local cost history
│   ├── budget.py           ← Budget tracking and alerts
│   ├── reporter.py         ← Table/chart/JSON output
│   └── alerts.py           ← Notification when approaching budget
├── tests/
├── pyproject.toml
└── README.md
```

### Monetization

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | CLI summary, 3 providers, manual entry |
| Pro | $5/mo | Unlimited providers, budget alerts (email/Slack/Discord), historical charts, team aggregation, export to Marketing Engine ROI dashboard |

### Note

This tool becomes significantly more valuable when integrated with agent-audit (predicted costs) and the Marketing Engine ROI dashboard (actual costs vs. revenue). As a standalone, it's competing with provider-native dashboards. As part of the Animus ecosystem, it's the financial nervous system.

### Estimated Build Time: 2 weekends

---

## Build Order

| Order | Tool | Effort | Ship By |
|-------|------|--------|---------|
| 1 | ✅ CLAUDE.md Generator | Built | Done — add Pro tier |
| 2 | MCP Server Manager | 1-2 weekends | Week of Feb 24 |
| 3 | Agent Audit (includes linting) | 1-2 weekends | Week of Mar 3 |
| 4 | Agent Memory Bootstrap | 2-3 weekends | Week of Mar 17 |
| 5 | AI Spend Dashboard | 2 weekends | When needed |

### Portfolio Narrative

After shipping all five:

"I built a suite of developer tools for the agentic AI ecosystem: context generation (CLAUDE.md Generator), infrastructure management (MCP Manager), cost estimation and workflow linting (Agent Audit), persistent memory without infrastructure (memboot), and spend aggregation (AI Spend). All available on PyPI. The memory tool also serves as an MCP server, and the audit tool runs in CI via GitHub Actions."

That's a complete developer tools story that maps directly to AI Enablement, FDE, and Solutions Engineering roles.

---

## Connection to Animus

These tools aren't random — they're the **developer-facing surface area** of the Animus ecosystem:

```
STANDALONE TOOLS (free, PyPI, portfolio pieces)
├── claudemd-forge     → generates context for → Animus Core
├── mcp-manager        → manages servers for  → Animus Forge agents
├── agent-audit        → validates configs for → Animus Forge workflows
├── memboot            → IS the memory layer of → Animus Core
└── ai-spend           → tracks costs for      → Animus Forge + Marketing Engine

ANIMUS (the full system)
├── Core (memboot + identity + interface)
├── Forge (workflows + agents + budgets)
└── Swarm (coordination)
```

Each standalone tool is a **gateway** to Animus. Users adopt the free tool, hit its ceiling, and the upgrade path is Animus.
