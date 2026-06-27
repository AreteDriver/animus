# Debugging Guide

> Common issues and how to fix them.

---

## Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'animus'`

**Fix**: Install shared types first, then the package:
```bash
pip install -e packages/types/
pip install -e packages/core/
```

## Virtual Environment Isolation

**Symptom**: Packages from different venvs conflict.

**Fix**: Each package has its own `.venv/` in CI, but for local dev use one venv at the repo root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e packages/types/
pip install -e "packages/core/[dev,api]"
```

## Missing API Keys

**Symptom**: Forge tests fail with "OPENAI_API_KEY not set"

**Fix**: Set dummy values for CI-style tests:
```bash
export OPENAI_API_KEY="sk-dummy-ci-no-network-calls"
export ANTHROPIC_API_KEY="sk-dummy-ci-no-network-calls"
```

## Ollama Not Running

**Symptom**: `Connection refused` on port 11434

**Fix**: Start Ollama:
```bash
ollama serve
```

## Memory Backend Errors

**Symptom**: ChromaDB connection fails

**Fix**: Default to SQLite:
```bash
animus-bootstrap config set memory.backend sqlite
```

## Forge Budget Exceeded

**Symptom**: `BudgetExceededError` during workflow

**Fix**: Increase budget or check `packages/forge/src/animus_forge/budget/manager.py`

## Import Errors After Update

**Symptom**: `ModuleNotFoundError` after pulling latest code

**Fix**: Reinstall in editable mode:
```bash
pip install -e packages/types/
pip install -e packages/core/
```

---

## See Also

- [Operators → Troubleshooting](../operators/troubleshooting.md) — Production issues
- [Operators → Known Issues](../operators/known-issues.md) — Documented bugs
