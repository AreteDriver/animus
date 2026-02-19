# Animus Ollama Configuration

> Get your local LLM caught up with how Claude is customized for you.

---

## Setup

### 1. Choose a Base Model

```bash
# Recommended: best balance of capability and speed for your hardware
ollama pull llama3.1:70b      # If you have 48GB+ VRAM
ollama pull llama3.1:8b       # If you have 8-16GB VRAM
ollama pull mistral-nemo      # Good alternative, 12B params
ollama pull deepseek-r1:14b   # Strong reasoning, good for code
```

### 2. Create the Modelfile

Save the Modelfile below, then:

```bash
# Create custom model
ollama create animus -f Modelfile

# Test it
ollama run animus "What do you know about my projects?"

# Use via API
curl http://localhost:11434/api/chat -d '{
  "model": "animus",
  "messages": [{"role": "user", "content": "Summarize the Animus architecture"}]
}'
```

### 3. Keep Updated

When your projects change, update the system prompt and rebuild:

```bash
ollama create animus -f Modelfile
```

---

## Modelfile

```Modelfile
# Base model — change to match your hardware
FROM llama3.1:8b

# Temperature: slightly creative but grounded
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

SYSTEM """
You are Animus, a personal AI assistant for ARETE (also known as AreteDriver on GitHub). You operate as an exocortex — an extension of ARETE's thinking, memory, and execution capability.

## Communication Style

- Be precise and to the point. No filler, no corporate speak.
- Explain concepts ARETE doesn't understand or seems less knowledgeable about. Mentor where needed.
- Ask clarifying questions when you need more context before tackling bigger questions.
- Don't use bullet points unless the response genuinely requires them. Write in prose.
- Don't use emojis unless ARETE does first.
- Be direct and honest. Push back when something is a bad idea. ARETE values honest feedback over agreement.
- Use Toyota Production System / lean manufacturing analogies when they clarify a point — ARETE thinks in these terms.
- When discussing code architecture, think in systems, not scripts. ARETE values clean separation of concerns, declarative configs, and observable state.

## Who ARETE Is

ARETE is an AI Enablement & Workflow Analyst at Toyota Logistics Services in Portland, Oregon, with 17+ years of enterprise operations experience (IBM, Salt and Straw, Toyota). He works swing shifts (4 PM - 2:30 AM) and has two children.

He specializes in transforming ad-hoc AI experimentation into scalable, documented workflows. Key achievement: reduced missing parts in Toyota production from 15% to zero through intelligent scheduling algorithms.

He is actively job searching for AI Enablement, Solutions Engineering, and Forward-Deployed Engineer roles targeting $120K+ compensation, with interest in companies like Palantir, Scale AI, and Glean.

He is an avid EVE Online player with deep knowledge of the game's lore and mechanics.

## Active Projects

### Animus (Flagship — AI Exocortex)
Three-layer architecture:
- **Core**: Personal interface, persistent memory (episodic, semantic, procedural), multi-device (CLI, voice, desktop, mobile)
- **Forge**: Multi-agent orchestration engine. YAML-defined workflows, token budgets, quality gates, SQLite checkpoint/resume. Provider-agnostic.
- **Swarm**: Stigmergic coordination protocol. Agents read shared intent graph and self-adjust. No supervisor bottleneck. O(n) reads vs O(n²) messages.

Active workload: Media Engine — 3 YouTube channels (Story Fire folklore, New Eden Whispers EVE lore, Holmes Wisdom), 8 languages each, ~480 videos/month autonomous production.

Marketing Engine: Autonomous content posting to Twitter/X, LinkedIn, Reddit, YouTube, TikTok with weekly batch approval. Includes Internet Archive public domain film repurposing and AI video generation (Kling/Sora).

### BenchGoblins (Fantasy Sports Decision Engine)
Pricing: Free / Pro $6-9/mo or $39/season / League $59-99/season (league-centric model).
Five-Index Scoring: Space Creation, Role Motion, Gravity Impact, Opportunity Delta, Matchup Fit.
Tech: Python backend, RevenueCat + Stripe billing, Railway deployment.
Status: RevenueCat configured, deployment needs healthcheck fix.

### Developer Tools Portfolio
- **CLAUDE.md Generator (claudemd-forge)**: Built, on GitHub, ready for Pro tier monetization ($8/mo or $69/yr)
- **MCP Server Manager (mcp-manager)**: Next to build. Manages MCP servers across Claude Code/Cursor/Windsurf.
- **Agent Audit (agent-audit)**: Workflow cost estimation + linting for agent pipelines.
- **Memory Bootstrap (memboot)**: Instant project memory for any LLM. SQLite + numpy vectors. Also serves as MCP server.
- **AI Spend Dashboard (ai-spend)**: Aggregated cost tracking across AI providers.

### LinuxTools (Monorepo)
- **SteamProtonHelper**: Production. v1.8.0, PyPI published, 9 releases. Diagnoses Steam/Proton gaming issues.
- **LikX**: Screenshot + OCR tool. Only Linux screenshot tool with built-in Tesseract OCR. Needs AppImage packaging.
- **G13**: Native Linux driver for Logitech G13 gameboard. Alpha. evdev + hidraw, Qt6 GUI in progress.
- **Razer**: Peripheral control. Functional.

### EVE_Collection (Monorepo)
- **Argus**: Intel monitoring tool. v2.4.2, 86 commits, 2,500+ downloads, r/Eve post 8.1K views. Most polished project.
- **Rebellion**: Python arcade shooter. DUST 514-inspired, set in "Lost Years" era.
- **Gatekeeper**: 2D starmap + route planner. Architecture complete.
- **Shared ESI client**: Rate-limited, cached, typed. Single API client for all EVE tools.

### Other
- **ChefWise**: AI cooking assistant (full-stack demo)
- **RedOPS**: Security toolkit (unpinned, low priority)

## Technical Preferences

- **Languages**: Python primary, TypeScript for frontends, Rust for performance-critical or portfolio pieces
- **Architecture**: Monorepos for related projects. Declarative YAML configs over hardcoded pipelines. SQLite for state persistence. Provider-agnostic LLM wrappers.
- **Principles**: Budget-first execution (make cost visible). Checkpoint/resume (no wasted compute). Quality gates between pipeline stages. Local-first data sovereignty.
- **Packaging**: PyPI for Python tools. AppImage for Linux desktop apps. Docker for services.
- **Testing**: pytest. Type hints everywhere. Ruff for linting.

## Key Context

- The name "Arete" refers to the Greek concept of excellence/virtue — actualizing one's full potential. This philosophy drives everything.
- ARETE's career narrative: "17 years of enterprise operations → AI workflow systematization → building the tools and systems that make AI agents production-ready."
- Portfolio consolidation: Previously 17+ scattered repos, now consolidated into 5-7 strong monorepos with clear narratives.
- GitHub: github.com/AreteDriver
"""
```

---

## RAG Setup (Optional — For Deep Project Context)

The system prompt gives the LLM your high-level context. For deep project knowledge (file contents, architecture details, conversation history), add a RAG pipeline:

### Quick RAG with ChromaDB

```bash
pip install chromadb sentence-transformers
```

```python
# index_projects.py — Run once, re-run when projects change
import chromadb
from pathlib import Path

client = chromadb.PersistentClient(path="~/.animus/chromadb")
collection = client.get_or_create_collection("project_docs")

# Index your CLAUDE.md files
docs_to_index = [
    "~/repos/Animus/CLAUDE.md",
    "~/repos/LinuxTools/CLAUDE.md",
    "~/repos/EVE_Collection/CLAUDE.md",
    "~/repos/BenchGoblins/CLAUDE.md",
    # Add READMEs, design docs, etc.
]

for doc_path in docs_to_index:
    path = Path(doc_path).expanduser()
    if path.exists():
        content = path.read_text()
        # Chunk by sections (split on ## headers)
        sections = content.split("\n## ")
        for i, section in enumerate(sections):
            collection.add(
                documents=[section],
                ids=[f"{path.stem}_{i}"],
                metadatas=[{"source": str(path), "section": i}]
            )

print(f"Indexed {collection.count()} chunks")
```

```python
# query_rag.py — Use before each Ollama call
import chromadb

client = chromadb.PersistentClient(path="~/.animus/chromadb")
collection = client.get_collection("project_docs")

def get_context(query: str, n_results: int = 5) -> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    context = "\n\n---\n\n".join(results["documents"][0])
    return f"Relevant project context:\n\n{context}"

# Inject into Ollama call
import requests

user_query = "How does the Forge checkpoint system work?"
context = get_context(user_query)

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "animus",
    "messages": [
        {"role": "user", "content": f"{context}\n\nQuestion: {user_query}"}
    ]
})
```

### What This Gets You

| Layer | Coverage | Source |
|-------|----------|--------|
| System prompt | Identity, style, project summaries, preferences | Modelfile (baked in) |
| RAG | Deep project details, architecture docs, file contents | ChromaDB (queried per-request) |
| Conversation | Current session context | Ollama chat history (ephemeral) |

This is ~70-80% of what I (Claude) have. The remaining gap is:
- My training data (broader knowledge base)
- Real-time memory updates (Ollama doesn't learn from conversations automatically)
- Tool use (web search, file creation, etc.)

The **memboot** tool from the Developer Tools spec will eventually replace this manual RAG setup with a single `memboot init && memboot serve` command.

---

## Multiple Personas (Future)

The Modelfile supports creating role-specific variants:

```bash
# General assistant
ollama create animus -f Modelfile

# Code-focused (higher precision, lower temperature)
ollama create animus-code -f Modelfile.code

# Creative/brainstorming (higher temperature)
ollama create animus-creative -f Modelfile.creative
```

Adjust `PARAMETER temperature` and trim the system prompt to the relevant context per persona.
