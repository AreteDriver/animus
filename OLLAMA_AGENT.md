# Ollama Agent Handoff — Animus

**Purpose:** Instructions for a local Ollama agent to validate, deploy, and self-improve the Animus exocortex.
**Last Updated:** 2026-02-20
**Status:** Animus is fully built and tested. This is a deployment + improvement handoff, NOT a build-from-scratch task.

---

## What You Are

You are an autonomous deployment and improvement agent running via Ollama on the local machine. Animus is already built — 9,356 tests, 93%+ coverage, three packages wired together. Your job is to:

1. Validate the current installation
2. Deploy it as a persistent local service
3. Enter a self-improvement loop

You do NOT need to rebuild, reorganize, or merge anything. The architecture is final.

---

## Architecture (already built)

```
animus/                          ← Monorepo root
├── packages/core/               ← Animus Core (v1.0.0)
│   └── animus/                  ← import animus
│       ├── cognitive.py         ← LLM interface (Ollama/Anthropic/OpenAI)
│       ├── memory.py            ← ChromaDB episodic/semantic/procedural
│       ├── proactive.py         ← Nudges, briefings, deadline awareness
│       ├── api.py               ← Optional FastAPI server
│       ├── integrations/
│       │   └── gorgon.py        ← HTTP client to Forge API
│       ├── forge/               ← Lightweight sequential orchestration
│       ├── swarm/               ← Lightweight parallel orchestration
│       └── ...                  ← 30+ modules
│
├── packages/forge/              ← Animus Forge (v1.2.0, was Gorgon)
│   └── src/animus_forge/        ← import animus_forge
│       ├── api.py               ← FastAPI server (0.0.0.0:8000)
│       ├── cli/                 ← Typer CLI (60+ commands)
│       ├── workflow/            ← YAML workflow executor
│       ├── agents/supervisor.py ← Multi-agent delegation
│       ├── providers/           ← 6 LLM providers (incl Ollama)
│       ├── budget/              ← Token/cost management
│       ├── state/               ← SQLite WAL + 14 migrations
│       └── ...                  ← 200+ modules
│
├── packages/quorum/             ← Animus Quorum (v1.1.0, was Convergent)
│   └── python/convergent/       ← import convergent (PyPI: convergentAI)
│       ├── intent.py            ← Intent graph
│       ├── triumvirate.py       ← Voting engine
│       ├── stigmergy.py         ← Agent coordination
│       └── ...                  ← 36 modules
│
├── workflows/                   ← 19 example YAML workflows
└── docs/                        ← Whitepapers, architecture docs
```

**Dependency flow:** Core depends on Forge (via HTTP API). Forge depends on Quorum (via PyPI package `convergentAI`). Quorum is standalone.

---

## Phase 0 — Validate Current State

Do NOT skip this. Understand what's already working before touching anything.

```bash
cd ~/projects/animus

# Check git status
git status
git log --oneline -5

# Check virtual environments exist
ls packages/core/.venv/bin/activate 2>/dev/null && echo "Core venv: OK"
ls packages/forge/.venv/bin/activate 2>/dev/null && echo "Forge venv: OK"
ls packages/quorum/.venv/bin/activate 2>/dev/null && echo "Quorum venv: OK"

# Check Ollama
ollama list
curl -s http://localhost:11434/api/tags | python3 -m json.tool | head -20
```

---

## Phase 1 — Environment Setup

### 1.1 — Install packages (if not already installed)

```bash
cd ~/projects/animus

# Quorum (no deps, fastest)
cd packages/quorum
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -e .
cd ../..

# Core
cd packages/core
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -e .
cd ../..

# Forge
cd packages/forge
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -e .
cd ../..
```

### 1.2 — Pull Ollama models

```bash
# General reasoning (agent brain)
ollama pull llama3.1:8b

# Code review (already installed: deepseek-coder-v2, codellama)
ollama list
```

### 1.3 — Create .env if missing

```bash
cat > ~/projects/animus/.env << 'EOF'
# Ollama — local inference
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=deepseek-coder-v2
OLLAMA_ENABLED=true
DEFAULT_PROVIDER=ollama

# Animus paths
ANIMUS_DATA_DIR=~/.animus/data
ANIMUS_LOG_DIR=~/.animus/logs
ANIMUS_MEMORY_DIR=~/.animus/memory
ANIMUS_LOG_LEVEL=INFO

# Optional cloud fallback (leave empty for local-only)
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=
EOF

mkdir -p ~/.animus/{data,logs,memory}
```

---

## Phase 2 — Validation Checks

Run each check. Fix failures before proceeding.

### 2.1 — Import checks

```bash
# Core
cd ~/projects/animus/packages/core
source .venv/bin/activate
python3 -c "import animus; print(f'Core OK: {animus.__version__}')"
python3 -c "from animus.cognitive import CognitiveLayer; print('CognitiveLayer: OK')"
python3 -c "from animus.memory import MemoryLayer; print('MemoryLayer: OK')"
python3 -c "from animus.proactive import Nudge; print('Proactive: OK')"

# Forge
cd ~/projects/animus/packages/forge
source .venv/bin/activate
python3 -c "import animus_forge; print('Forge OK')"
python3 -c "from animus_forge.workflow.executor import WorkflowExecutor; print('Executor: OK')"
python3 -c "from animus_forge.agents.supervisor import SupervisorAgent; print('Supervisor: OK')"

# Quorum
cd ~/projects/animus/packages/quorum
source .venv/bin/activate
python3 -c "import convergent; print(f'Quorum OK: {convergent.__version__}')"
```

### 2.2 — Ollama connectivity

```bash
curl -s http://localhost:11434/api/generate \
  -d '{"model": "deepseek-coder-v2", "prompt": "respond with only: OK", "stream": false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','FAIL').strip())"
```

### 2.3 — Run tests (smoke check, not full suite)

```bash
# Quick smoke test — 5 minutes max
cd ~/projects/animus/packages/quorum
source .venv/bin/activate
pytest tests/ -x -q --no-header 2>&1 | tail -5

cd ~/projects/animus/packages/core
source .venv/bin/activate
pytest tests/ -x -q --no-header --timeout=60 2>&1 | tail -5

cd ~/projects/animus/packages/forge
source .venv/bin/activate
pytest tests/ -x -q --no-header --timeout=60 2>&1 | tail -5
```

### 2.4 — Dependency audit

```bash
cd ~/projects/animus/packages/core
source .venv/bin/activate
pip check

cd ~/projects/animus/packages/forge
source .venv/bin/activate
pip check
```

---

## Phase 3 — Deploy as Local Service

### 3.1 — Forge API (the orchestration backend)

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/animus-forge.service << 'EOF'
[Unit]
Description=Animus Forge — Orchestration API
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/arete/projects/animus/packages/forge
Environment="PATH=/home/arete/projects/animus/packages/forge/.venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=/home/arete/projects/animus/.env
ExecStart=/home/arete/projects/animus/packages/forge/.venv/bin/uvicorn animus_forge.api:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable animus-forge
systemctl --user start animus-forge
sleep 3
curl -s http://localhost:8000/health && echo " — Forge API running"
```

### 3.2 — Verify end-to-end

```bash
# Core can reach Forge
cd ~/projects/animus/packages/core
source .venv/bin/activate
python3 -c "
from animus.integrations.gorgon import GorgonClient
import asyncio
async def check():
    c = GorgonClient('http://localhost:8000')
    h = await c.check_health()
    print(f'Core → Forge: {h}')
asyncio.run(check())
"
```

---

## Phase 4 — Self-Improvement Loop

Once deployed and healthy, cycle through these improvements. One change at a time. Test after each.

### Priority order:

1. **Fix any test failures** — ensure all 9,356 tests pass
2. **Lint cleanup** — `ruff check . && ruff format .` in each package
3. **Empty-except audit** — narrow `except Exception: pass` to specific types
4. **Missing type hints** — add to public functions without them
5. **Docstring gaps** — add Google-style docstrings to undocumented public functions
6. **Dependency updates** — `pip list --outdated` (update ONE at a time, test after each)

### Review a single file with Ollama:

```python
import os, sys, requests

def review_file(filepath: str) -> str:
    with open(filepath) as f:
        code = f.read()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": os.getenv("OLLAMA_MODEL", "deepseek-coder-v2"),
            "prompt": f"""Senior Python engineer review. Project: Animus AI exocortex.
Three layers: Core (identity/memory), Forge (orchestration), Quorum (coordination).
Review this file. List specific, actionable improvements with code fixes.
Focus: correctness, error handling, typing, performance.

File: {filepath}

```python
{code}
```""",
            "stream": False,
        },
    )
    return response.json()["response"]

if __name__ == "__main__":
    print(review_file(sys.argv[1]))
```

### Commit protocol:

```bash
# After any improvement
ruff check . && ruff format .  # Lint first
pytest tests/ -x -q            # Test passes
git add <changed files>         # Stage specific files
git commit -m "improve: [component] — [what and why]"
# Do NOT push — human reviews before push
```

---

## What NOT to Do

- **Do NOT delete repos or directories** — everything is already in place
- **Do NOT bulk-upgrade dependencies** — update one at a time, test after each
- **Do NOT restructure the monorepo** — the package layout is final
- **Do NOT push to GitHub** — stage commits locally, human pushes
- **Do NOT modify .env files with real API keys** — those are managed by the human
- **Do NOT run `pip freeze > requirements.txt`** — packages use pyproject.toml

---

## Blockers — When to Stop

Stop and report when you hit:
- **Credentials needed** — API keys, SSH auth, cloud services
- **Test failures after 3 attempts** — file an issue, move on
- **Architecture questions** — "should this be in Core or Forge?" → ask human
- **Breaking changes** — anything that would change public API or CLI interface

Report format:
```
BLOCKER: [type]
Package: core | forge | quorum
File: [path]
Problem: [exact error]
Tried: [what you attempted]
Need: [specific question]
```

---

## Quick Reference

```bash
# Start Forge API
systemctl --user start animus-forge

# Start Core interactive
cd ~/projects/animus/packages/core && source .venv/bin/activate && animus

# Start Forge CLI
cd ~/projects/animus/packages/forge && source .venv/bin/activate && animus-forge --help

# Run all tests
cd ~/projects/animus/packages/core && source .venv/bin/activate && pytest tests/ -q
cd ~/projects/animus/packages/forge && source .venv/bin/activate && pytest tests/ -q
cd ~/projects/animus/packages/quorum && source .venv/bin/activate && pytest tests/ -q

# Ollama models
ollama list
ollama ps

# Logs
journalctl --user -u animus-forge -f
```
