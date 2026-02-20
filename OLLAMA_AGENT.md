# Ollama Agent Handoff — Animus

**Purpose:** Complete instructions and prompts for a local Ollama agent to finish, test, harden, and deploy the Animus exocortex as a persistent local service.
**Last Updated:** 2026-02-20
**Status:** Built and tested. Needs production hardening, CodeQL cleanup, and deployment.

---

## What You Are

You are an autonomous agent running via Ollama on this machine. Animus is a personal AI exocortex — three packages, 9,373 tests, deployed locally. Your job:

1. Validate the installation (Phase 1)
2. Fix remaining code quality issues (Phase 2)
3. Run the full test suite and fix failures (Phase 3)
4. Deploy as persistent systemd services (Phase 4)
5. Production hardening (Phase 5)
6. Enter self-improvement loop (Phase 6)

**You do NOT need to rebuild or restructure anything. The architecture is final.**

---

## Architecture

```
~/projects/animus/                    ← Monorepo root
├── packages/core/                    ← Animus Core v1.0.0
│   └── animus/                       ← import animus
│       ├── cognitive.py              ← LLM interface (Ollama/Anthropic/OpenAI/Mock)
│       ├── memory.py                 ← ChromaDB episodic/semantic/procedural
│       ├── proactive.py              ← Nudges, briefings, deadline awareness
│       ├── autonomous.py             ← Autonomous executor
│       ├── api.py                    ← Optional FastAPI server
│       ├── integrations/             ← Google, Todoist, filesystem, webhooks
│       │   └── gorgon.py            ← HTTP client to Forge API
│       ├── forge/                    ← Lightweight sequential orchestration
│       ├── swarm/                    ← Lightweight parallel DAG orchestration
│       ├── protocols/                ← Abstract interfaces (memory, intelligence, sync, safety)
│       └── sync/                     ← Peer discovery + sync
│
├── packages/forge/                   ← Animus Forge v1.2.0 (was: Gorgon)
│   └── src/animus_forge/             ← import animus_forge
│       ├── api.py                    ← FastAPI server (127.0.0.1:8000)
│       ├── cli/                      ← Typer CLI (60+ commands)
│       ├── workflow/                 ← YAML workflow executor (sequential + parallel)
│       ├── agents/supervisor.py      ← Multi-agent delegation + consensus
│       ├── providers/                ← 6 LLM providers (Ollama, Anthropic, OpenAI, etc)
│       ├── budget/                   ← Token/cost management (persistent SQLite)
│       ├── state/                    ← SQLite WAL + 14 migrations
│       ├── skills/                   ← Skill definitions (YAML + docs)
│       ├── mcp/                      ← MCP tool execution
│       ├── dashboard/                ← Streamlit dashboard
│       └── webhooks/                 ← Webhook delivery + circuit breaker
│
├── packages/quorum/                  ← Animus Quorum v1.1.0 (was: Convergent)
│   └── python/convergent/            ← import convergent (PyPI: convergentAI)
│       ├── intent.py                 ← Intent graph
│       ├── triumvirate.py            ← Voting engine
│       ├── stigmergy.py              ← Agent coordination signals
│       ├── gorgon_bridge.py          ← Integration bridge to Forge
│       └── ...                       ← 36 modules total
│
├── .env                              ← Ollama config (gitignored)
├── .github/workflows/                ← CI: lint + test + security
└── docs/                             ← Whitepapers
```

**Dependency flow:** Quorum is standalone (zero deps) → Forge depends on Quorum (`convergentai` PyPI) → Core depends on Forge (via HTTP API)

**Entry points:**
- `animus` — Interactive CLI (prompt-toolkit)
- `animus-forge --help` — Orchestration CLI (Typer, 60+ commands)
- `uvicorn animus_forge.api:app` — Forge REST API

---

## Current State (verified 2026-02-20)

| Package | Tests | Coverage | Status |
|---------|-------|----------|--------|
| Quorum  | 906   | 97%      | All passing |
| Core    | 1,736 | 95%      | 1,630 pass, 106 skip (optional deps) |
| Forge   | 6,731 | 86%      | Running validation |

- Virtual environments: all 3 created and verified
- Ollama models: `deepseek-coder-v2` (8.9GB), `codellama` (3.8GB), `llama3.1:8b` (4.9GB)
- `.env` created with Ollama defaults
- `~/.animus/{data,logs,memory}` directories exist
- CI: all green on GitHub

**Known issues to fix:**
- 85 CodeQL alerts (mostly false positives, ~15 need code fixes)
- `pytest-timeout` not installed in any venv (use plain `pytest`)
- Forge tests MUST run from `packages/forge/` directory (relative path deps)

---

## Phase 1 — Validate Installation

**Do NOT skip this.** Run every check. If anything fails, fix it before proceeding.

```bash
cd ~/projects/animus

# 1. Git status
git status && git log --oneline -3

# 2. Virtual environments
for pkg in core forge quorum; do
  ls "packages/$pkg/.venv/bin/activate" 2>/dev/null && echo "$pkg venv: OK" || echo "$pkg venv: MISSING"
done

# 3. Import checks
cd packages/quorum && source .venv/bin/activate
python3 -c "import convergent; print(f'Quorum: {convergent.__version__}')"

cd ~/projects/animus/packages/core && source .venv/bin/activate
python3 -c "import animus; print(f'Core: {animus.__version__}')"
python3 -c "from animus.cognitive import CognitiveLayer; print('CognitiveLayer: OK')"
python3 -c "from animus.memory import MemoryLayer; print('MemoryLayer: OK')"

cd ~/projects/animus/packages/forge && source .venv/bin/activate
python3 -c "import animus_forge; print('Forge: OK')"
python3 -c "from animus_forge.agents.supervisor import SupervisorAgent; print('Supervisor: OK')"

# 4. Ollama connectivity
ollama list
curl -s http://localhost:11434/api/generate \
  -d '{"model": "deepseek-coder-v2", "prompt": "respond with only: OK", "stream": false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','FAIL').strip())"

# 5. Dependency health
cd ~/projects/animus/packages/core && source .venv/bin/activate && pip check
cd ~/projects/animus/packages/forge && source .venv/bin/activate && pip check
```

**Expected output:** All "OK", no broken deps. If venvs are missing, create them:
```bash
cd packages/<pkg> && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

---

## Phase 2 — Code Quality Fixes

### 2.1 — Lint all packages

```bash
cd ~/projects/animus
ruff check packages/ && ruff format --check packages/
```

If issues found, fix them:
```bash
ruff check packages/ --fix
ruff format packages/
```

### 2.2 — Fix empty-except blocks

Use this Ollama prompt to find and fix empty except blocks:

> **PROMPT — empty-except audit:**
> ```
> You are reviewing Python code for the Animus project. Find all `except` blocks
> that contain only `pass` without an explanatory comment. For each one:
> 1. Read the surrounding code to understand WHY the exception is caught and ignored
> 2. Add a comment on the `pass` line explaining the reason
> 3. If the except clause is too broad (catches Exception or BaseException), narrow
>    it to the specific exception types that can actually occur
>
> Format: `pass  # Reason why this exception is safely ignored`
>
> Example fixes:
>   except (ValueError, TypeError):
>       pass  # Invalid date format — treat as no timeout
>   except json.JSONDecodeError:
>       pass  # Keep raw string if not valid JSON
>   except ImportError:
>       pass  # Optional dependency not installed — feature disabled
> ```

Files to check:
- `packages/core/animus/api.py:1444`
- `packages/core/animus/forge/gates.py:141`
- `packages/core/animus/sync/discovery.py:170,175`
- `packages/core/animus/sync/client.py:135`
- `packages/core/animus/integrations/google/calendar.py:33`
- `packages/core/animus/integrations/google/gmail.py:34`
- `packages/core/tests/test_coverage_push3.py:662`
- `packages/forge/tests/test_circuit_breaker.py:505`

### 2.3 — Fix side-effect-in-assert

> **PROMPT — assert side effects:**
> ```
> Find assert statements that call functions with side effects. Extract the call
> into a variable, then assert the variable. Example:
>   # Bad:  assert cache.pop("key") == value
>   # Good: result = cache.pop("key"); assert result == value
> ```

File: `packages/forge/tests/test_cache_coverage.py:60-61`

### 2.4 — Fix redundant self-assignment

File: `packages/forge/tests/test_budget_passthrough.py:224`
Look for `x = x` and either remove or replace with the intended assignment.

### 2.5 — Fix catch-base-exception

File: `packages/core/animus/integrations/manager.py:259`
Narrow `except BaseException` to `except Exception` (unless it intentionally catches KeyboardInterrupt/SystemExit).

### 2.6 — Add `__all__` for re-export modules

> **PROMPT — unused import fix:**
> ```
> This file re-exports symbols for convenience. Add an `__all__` list at the top
> of the module listing all intentionally re-exported names. This tells linters
> and CodeQL the imports are intentional.
>
> Example:
>   __all__ = ["Layout", "Live", "Style", "Text"]
> ```

Files:
- `packages/forge/src/animus_forge/cli/rich_output.py` (Layout, Live, Style, Text)
- `packages/forge/src/animus_forge/cli/main.py` (workflow commands, codebase commands, dev commands)
- `packages/forge/src/animus_forge/dashboard/app.py` (render functions)

---

## Phase 3 — Full Test Suite

Run each package separately. Forge MUST run from its own directory.

```bash
# Quorum (fastest — ~11 seconds)
cd ~/projects/animus/packages/quorum
source .venv/bin/activate
pytest tests/ -q --no-header
# Expected: 906 passed

# Core (~7 seconds)
cd ~/projects/animus/packages/core
source .venv/bin/activate
pytest tests/ -q --no-header
# Expected: ~1630 passed, ~106 skipped

# Forge (largest — may take several minutes)
cd ~/projects/animus/packages/forge
source .venv/bin/activate
pytest tests/ -q --no-header
# Expected: ~6731 passed
```

**If tests fail:**
1. Read the error message carefully
2. Check if it's a missing dependency (install it)
3. Check if it's a flaky test (run again with `-x` to isolate)
4. Fix the issue, run again
5. If stuck after 3 attempts, log a BLOCKER and move on

**Coverage check:**
```bash
cd ~/projects/animus/packages/core && source .venv/bin/activate
pytest tests/ --cov=animus --cov-report=term-missing -q 2>&1 | tail -5
# Must be >= 95%

cd ~/projects/animus/packages/forge && source .venv/bin/activate
pytest tests/ --cov=animus_forge --cov-report=term-missing -q 2>&1 | tail -5
# Must be >= 85%
```

---

## Phase 4 — Deploy as Local Services

### 4.1 — Forge API (orchestration backend)

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

### 4.2 — Verify Core → Forge connectivity

```bash
cd ~/projects/animus/packages/core && source .venv/bin/activate
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

### 4.3 — Test a workflow execution

```bash
cd ~/projects/animus/packages/forge && source .venv/bin/activate
animus-forge workflow list
animus-forge workflow validate workflows/code-review.yaml
```

### 4.4 — Verify Ollama is the active provider

```bash
cd ~/projects/animus/packages/core && source .venv/bin/activate
python3 -c "
from animus.cognitive import CognitiveLayer, ModelProvider
layer = CognitiveLayer(provider=ModelProvider.OLLAMA)
import asyncio
result = asyncio.run(layer.think('What is 2+2? Reply with just the number.'))
print(f'Ollama response: {result}')
"
```

---

## Phase 5 — Production Hardening

### 5.1 — Narrow exception types

> **PROMPT — exception narrowing:**
> ```
> You are hardening Python code for production. Find `except Exception` blocks
> and narrow them to specific exception types. Read the try block to determine
> what exceptions can actually be raised.
>
> Common mappings:
> - File I/O: OSError, FileNotFoundError, PermissionError
> - JSON: json.JSONDecodeError
> - HTTP: httpx.HTTPError, requests.RequestException
> - Database: sqlite3.Error, sqlite3.OperationalError
> - Type conversion: ValueError, TypeError, KeyError
> - Import: ImportError, ModuleNotFoundError
>
> Do NOT narrow exceptions in test files — those are intentionally broad.
> Do NOT change `except Exception` in top-level error handlers (API routes, CLI commands).
> ```

### 5.2 — Add missing type hints

> **PROMPT — type hints:**
> ```
> Add type hints to all public functions (not starting with _) that are missing them.
> Use Python 3.10+ syntax: list[str] not List[str], str | None not Optional[str].
>
> Rules:
> - If a class has a method named `list`, `set`, or `dict`, add
>   `from __future__ import annotations` at the top of the file
> - Return types are required
> - Use `Any` sparingly — prefer specific types
> - For callbacks, use `Callable[[arg_types], return_type]`
> ```

### 5.3 — Dependency audit

```bash
# Check for outdated packages
cd ~/projects/animus/packages/core && source .venv/bin/activate
pip list --outdated

cd ~/projects/animus/packages/forge && source .venv/bin/activate
pip list --outdated
```

**Update ONE at a time. Test after each.**
```bash
pip install --upgrade <package>
pytest tests/ -x -q
```

**Do NOT update:**
- `pandas` — major version (2→3) blocked by Streamlit
- Any package that requires Python version changes

---

## Phase 6 — Self-Improvement Loop

Once deployed and healthy, cycle through improvements. **One change at a time. Test after each.**

### 6.1 — Review files with Ollama

```python
#!/usr/bin/env python3
"""Review a single file using local Ollama model."""
import os, sys, requests

def review_file(filepath: str) -> str:
    with open(filepath) as f:
        code = f.read()

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": os.getenv("OLLAMA_MODEL", "deepseek-coder-v2"),
            "prompt": f"""Senior Python engineer code review.
Project: Animus — personal AI exocortex.
Three layers: Core (identity/memory), Forge (orchestration), Quorum (coordination).

Review this file. For each issue found, provide:
1. Line number
2. Issue category (correctness/security/performance/typing/style)
3. Current code
4. Suggested fix
5. Why it matters

Focus on: correctness bugs, error handling gaps, missing type hints, performance issues.
Skip: style preferences, docstring formatting, import ordering.

File: {filepath}

```python
{code}
```""",
            "stream": False,
        },
    )
    return response.json()["response"]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python review.py <filepath>")
        sys.exit(1)
    print(review_file(sys.argv[1]))
```

Save as `~/projects/animus/scripts/review.py` and run:
```bash
python3 scripts/review.py packages/core/animus/cognitive.py
```

### 6.2 — Batch review priority files

Review these files first (highest impact):

```bash
# Core — brain
python3 scripts/review.py packages/core/animus/cognitive.py
python3 scripts/review.py packages/core/animus/memory.py
python3 scripts/review.py packages/core/animus/proactive.py
python3 scripts/review.py packages/core/animus/autonomous.py

# Forge — orchestration
python3 scripts/review.py packages/forge/src/animus_forge/workflow/executor_core.py
python3 scripts/review.py packages/forge/src/animus_forge/agents/supervisor.py
python3 scripts/review.py packages/forge/src/animus_forge/api.py
python3 scripts/review.py packages/forge/src/animus_forge/budget/persistent.py

# Quorum — coordination
python3 scripts/review.py packages/quorum/python/convergent/intent.py
python3 scripts/review.py packages/quorum/python/convergent/triumvirate.py
```

### 6.3 — Test generation

> **PROMPT — write tests:**
> ```
> Write pytest tests for this module. Requirements:
> - Test all public functions
> - Include happy path, edge cases, and error cases
> - Mock external dependencies (LLM calls, HTTP, filesystem)
> - Use pytest fixtures for setup/teardown
> - Use parametrize for similar test cases
> - No pytest-asyncio — use asyncio.run() wrapper for async tests
> - Target 95%+ coverage for the module
>
> Project conventions:
> - Mocks use unittest.mock (patch, MagicMock)
> - Test files go in the package's tests/ directory
> - Name: test_<module_name>.py
> ```

### 6.4 — Commit protocol

```bash
# After EVERY improvement:
cd ~/projects/animus

# 1. Lint
ruff check packages/ --fix
ruff format packages/

# 2. Test the affected package
cd packages/<affected_package>
source .venv/bin/activate
pytest tests/ -x -q

# 3. Commit (specific files only)
cd ~/projects/animus
git add <changed files>
git commit -m "improve(<package>): <what and why>"

# Do NOT push — human reviews before push
```

---

## Ollama Prompts Library

### General reasoning (use llama3.1:8b)
```
You are an AI systems architect working on Animus, a personal AI exocortex.
The system has three layers:
- Core: Identity, memory (ChromaDB), proactive engine, CLI
- Forge: Multi-agent orchestration, workflow execution, budget management
- Quorum: Coordination protocol, intent graphs, voting, stigmergy

Answer the following question about the system: <question>
```

### Code review (use deepseek-coder-v2)
```
Senior Python engineer review. Project: Animus AI exocortex (3 packages, 9300+ tests).
Review this code for: correctness, error handling, typing, performance, security.
List specific issues with line numbers and fixes. No style opinions.

<code>
```

### Test writing (use deepseek-coder-v2)
```
Write pytest unit tests for this Python module from the Animus project.
Requirements: mock all external calls, test edge cases, use fixtures.
No pytest-asyncio (use asyncio.run() wrapper). Target 95% coverage.

Module path: <path>
<code>
```

### Bug investigation (use deepseek-coder-v2)
```
A test is failing in the Animus project. Analyze the error and suggest a fix.
Show the exact code change needed. Do not change test expectations unless the
test itself is wrong.

Test file: <path>
Error output:
<error>
```

### Docstring generation (use llama3.1:8b)
```
Add Google-style docstrings to all public functions in this module.
Include: one-line summary, Args with types, Returns, Raises.
Do not modify any code logic.

<code>
```

---

## What NOT to Do

- **Do NOT delete any files or directories**
- **Do NOT bulk-upgrade dependencies** — one at a time, test after each
- **Do NOT restructure the monorepo** — the package layout is final
- **Do NOT push to GitHub** — stage commits locally, human pushes
- **Do NOT modify .env with real API keys** — managed by human
- **Do NOT run `pip freeze > requirements.txt`** — packages use pyproject.toml
- **Do NOT install new packages** without checking pyproject.toml first
- **Do NOT change public API signatures** — existing code depends on them

---

## Blockers — When to Stop

Stop and write a report when you hit:

- **Credentials needed** — API keys, SSH auth, cloud services
- **Test failures after 3 attempts** — log it, move on
- **Architecture questions** — "should this be in Core or Forge?" → ask human
- **Breaking changes** — anything that would change public API or CLI

**Report format:**
```
BLOCKER: [type]
Package: core | forge | quorum
File: [path]
Problem: [exact error]
Tried: [what you attempted]
Need: [what the human should do]
```

Write blocker reports to `~/projects/animus/BLOCKERS.md` (append, don't overwrite).

---

## Quick Reference

```bash
# Services
systemctl --user start animus-forge     # Start Forge API
systemctl --user stop animus-forge      # Stop Forge API
systemctl --user status animus-forge    # Check status
journalctl --user -u animus-forge -f    # Tail logs

# CLI
cd ~/projects/animus/packages/core && source .venv/bin/activate && animus
cd ~/projects/animus/packages/forge && source .venv/bin/activate && animus-forge --help

# Tests
cd ~/projects/animus/packages/quorum && source .venv/bin/activate && pytest tests/ -q
cd ~/projects/animus/packages/core && source .venv/bin/activate && pytest tests/ -q
cd ~/projects/animus/packages/forge && source .venv/bin/activate && pytest tests/ -q

# Lint
cd ~/projects/animus && ruff check packages/ && ruff format --check packages/

# Ollama
ollama list                             # List models
ollama ps                               # Running models
curl -s http://localhost:11434/api/tags  # API check

# Review a file
python3 ~/projects/animus/scripts/review.py <filepath>
```

---

## Success Criteria

You're done when:
- [ ] All 3 packages import cleanly
- [ ] Ollama responds to prompts
- [ ] Quorum: 906 tests pass
- [ ] Core: 1630+ tests pass (skips OK for optional deps)
- [ ] Forge: 6700+ tests pass
- [ ] `ruff check packages/` returns 0 errors
- [ ] `ruff format --check packages/` returns 0 changes needed
- [ ] Forge API runs as systemd service (`curl localhost:8000/health`)
- [ ] Core can reach Forge via HTTP
- [ ] No `except Exception: pass` without comments
- [ ] All public functions have type hints
- [ ] Coverage: Core >= 95%, Forge >= 85%, Quorum >= 97%
