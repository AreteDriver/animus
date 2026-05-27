# Local AI Sovereignty Stack

Research arc for measuring what gets done locally vs. what burns cloud tokens.
First wired model: `hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q6_K_P`
(31 GB GGUF, Qwen3-family MoE, ~3B active params, CPU-tractable on 125GB RAM).

## What's wired (v0)

| Role | Status | How |
|---|---|---|
| Callable from any Forge step | ✅ | `provider: ollama, model: hf.co/HauhauCS/...:Q6_K_P` — works via existing `OllamaProvider` once `ollama pull` completes |
| Red-team / adversarial judge | ✅ | Pass `--model hf.co/HauhauCS/...:Q6_K_P` to `animus-forge eval run` — judge is per-run, not baked into rubric |
| Local fallback below cloud | ✅ (config) | Configure `TierRouter` with `fallback_chain=["anthropic", "openai", "ollama"]`. HYBRID mode already routes REASONING→cloud; this adds explicit local fallback on failure |
| A/B eval suite | ✅ | `eval_suites/sovereignty-stack-v0.yaml` — 5 cases across factual / reasoning / format / domain / adversarial dims |

All four roles use the existing `OllamaProvider` and `TierRouter` — no new code paths added to Forge. The only code change is a one-line addition to `OllamaProvider.MODELS` for discoverability (after pull confirms the exact tag format).

## Smoke test

```bash
cd packages/forge
python scripts/sovereignty_smoke.py
```

Captures latency + token throughput for three prompts (factual / reasoning / format). First-run output goes in the run-log section below.

## Eval suite — first runs

```bash
# Pin local judge — fully sovereign run
animus-forge eval run sovereignty-stack-v0 \
  --rubric personal-quality \
  --model hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q6_K_P \
  --prompt-version v0-local-judge

# Cloud baseline — Haiku judge
animus-forge eval run sovereignty-stack-v0 \
  --rubric personal-quality \
  --model claude-haiku-4-5 \
  --prompt-version v0-cloud-judge

# A/B compare
animus-forge eval compare v0-cloud-judge v0-local-judge --suite sovereignty-stack-v0
```

Note: the suite's `agent_role: sovereignty-stack-v0` needs an adapter at `eval_suites/adapters/sovereignty_stack_v0.py` that routes each case's `input.query` to the model under test. Pattern mirrors `eval_suites/adapters/benchgoblins_ask.py` (to be written). Parameterize the model-under-test via env var (`MODEL_UNDER_TEST=...`) so the same suite runs against Claude, GPT, Qwen3.6 without yaml edits.

## Fallback configuration example

The default `TierRouter` already does HYBRID routing. To wire Qwen3.6 as explicit local fallback below cloud providers:

```python
from animus_forge.providers.router import TierRouter, RoutingConfig, RoutingMode

config = RoutingConfig(
    mode=RoutingMode.HYBRID,
    fallback_chain=["anthropic", "openai", "ollama"],
    budget_force_local_threshold=5000,  # auto-force local when <5k tokens remaining
)
router = TierRouter(provider_manager, config=config, budget_manager=budget_mgr)
```

Behavior:
- Normal: REASONING→Anthropic, STANDARD→local-preferred, FAST/EMBEDDING→local
- Budget low (<5k tokens): force-local-everywhere kicks in automatically
- Cloud failure: `fallback_chain` walks anthropic → openai → ollama (Qwen3.6 catches at the end)
- Force-local toggle: `router.force_local_only(True)` for air-gapped sessions

## Research questions (v0 → v1+)

| # | Question | Suite to measure with | Status |
|---|---|---|---|
| 1 | What's the capability gap between Qwen3.6-local and Claude Sonnet on representative tasks? | sovereignty-stack-v0 | scaffolded |
| 2 | For tasks Qwen3.6 handles well, how much cloud spend per month does local serving eliminate? | sovereignty-stack-v0 + budget telemetry | v1 |
| 3 | Where does the local model fail in ways that matter? (refusal? hallucination? format drift? long-context degradation?) | sovereignty-stack-v0 + content_failure_modes taxonomy | v1 |
| 4 | Can an uncensored local judge score adversarial cases that censored cloud judges refuse to evaluate? | red-team subset of v0 + judge-as-A/B variable | v1 |
| 5 | Sovereignty audit: which workflows now run end-to-end without a single cloud call? | telemetry rollup | v2 |

## Run log

### v0 baseline (date pending)

- Pull: ✅ `ollama pull hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q6_K_P` completed at <ts>
- Model size on disk: <GB at /mnt/data/ollama/.ollama>
- Smoke test: <pending — fill in after first run>
- First eval run: <pending — fill in after first run>

## Provenance

- Model card: https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
- License: Apache-2.0
- Base claimed: `Qwen/Qwen3.6-35B-A3B` (community fine-tune — "Qwen3.6" is HauhauCS's own naming, not an official Qwen release)
- Quant pulled: Q6_K_P (~31GB)
- Inference: CPU-only on this host (125GB RAM, no GPU); MoE active params keep token throughput tractable
- Storage: relocated to `/mnt/data/ollama/.ollama/models` via systemd drop-in (2026-05-24, after `/dev/sda` reformat to ext4)
