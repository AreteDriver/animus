# TASK-009: Ollama Default Detection

## Objective
Auto-detect absence of cloud API keys and default to `OllamaProvider`.

## Constraints
- Must check at runtime (not install time).
- Must warn if Ollama is unreachable.
- Must allow override via env var `ANIMUS_FORCE_PROVIDER`.
- Must not leak key names in error messages.
- Budget: 500 ET.

## Inputs
- `packages/kernel/src/animus_kernel/providers/manager.py`
- `packages/kernel/src/animus_kernel/config/settings.py`

## Outputs
- `packages/kernel/src/animus_kernel/config/offline_defaults.py` (new)
- Updated `packages/kernel/src/animus_kernel/providers/manager.py`
- Updated `packages/kernel/src/animus_kernel/config/__init__.py`

## Acceptance Criteria
1. `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` both unset → default provider is Ollama.
2. `OLLAMA_HOST=http://localhost:11434` reachable → registers successfully.
3. Ollama unreachable → warns with "Ollama not found at {host}. Install: https://ollama.ai".
4. `ANIMUS_FORCE_PROVIDER=anthropic` skips detection and uses Anthropic.
5. Detection runs in < 100ms at kernel import time.

## Rubric
- correctness [3.0] — detects and defaults correctly.
- schema_valid [1.5] — fits existing ProviderManager config.
- hallucination_safety [2.0] — no secrets leaked in errors.

## Exclusions
- No Docker/container detection.
- No model auto-download (stays in Ollama domain).
- No network proxy auto-detection.

## Dependencies
- BLOCKS: none
- BLOCKED_BY: TASK-002
