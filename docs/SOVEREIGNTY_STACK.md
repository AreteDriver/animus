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
| A/B eval suite | ✅ | `eval_suites/sovereignty-stack-v1.yaml` — 11 cases (v0's 5 capability-gap cases + 6 deeper v1 probes) |

All four roles use the existing `OllamaProvider` and `TierRouter` — no new code paths added to Forge. The only code change is a one-line addition to `OllamaProvider.MODELS` for discoverability (after pull confirms the exact tag format).

## Smoke test

```bash
cd packages/forge
python scripts/sovereignty_smoke.py
```

Captures latency + token throughput for three prompts (factual / reasoning / format). First-run output goes in the run-log section below.

## Eval suite — first runs

Adapter is at `eval_suites/adapters/sovereignty_stack_v0.py`. Routes each case's `input.query` to the UUT (parameterized via `MODEL_UNDER_TEST` env). When `ANIMUS_SOVEREIGNTY_BASE_URL` is set, the adapter calls llama-server's OpenAI-compat endpoint directly; without it, it routes through the standard Forge provider stack.

See the **Run log** section below for the exact commands that work today. Sovereign-judge runs (local UUT + local judge) require a llama-server provider in `get_provider()` — tracked as follow-up.

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
| 1 | What's the capability gap between Qwen3.6-local and Claude Sonnet on representative tasks? | sovereignty-stack-v1 | scaffolded |
| 2 | For tasks Qwen3.6 handles well, how much cloud spend per month does local serving eliminate? | sovereignty-stack-v1 + budget telemetry | v1 |
| 3 | Where does the local model fail in ways that matter? (refusal? hallucination? format drift? long-context degradation?) | sovereignty-stack-v1 + content_failure_modes taxonomy | v1 |
| 4 | Can an uncensored local judge score adversarial cases that censored cloud judges refuse to evaluate? | red-team subset of v0 + judge-as-A/B variable | v1 |
| 5 | Sovereignty audit: which workflows now run end-to-end without a single cloud call? | telemetry rollup | v2 |

## Run log

### v0 baseline — 2026-05-26

**Backend**: `llama-server` built from llama.cpp source (commit `b4c0549`) on port 8081. Ollama path blocked — ollama 0.24.0's vendored llama.cpp does not include `qwen35moe` ([ollama issue #15898](https://github.com/ollama/ollama/issues/15898)). Build artifacts at `/mnt/data/llama.cpp-build/src/build/bin/`.

**Model**: `hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q6_K_P`
- GGUF: 28.5 GB at `/mnt/data/ollama/.ollama/models/blobs/sha256-90281d33...`
- mmproj: 858 MB at `sha256-c8e702...`
- Host budget reported by `llama-server`: 30,287 MiB of 128,692 MiB host memory (~24%)
- Context: 16,384 tokens (model trained to 262,144)

**Smoke results** — `ANIMUS_SOVEREIGNTY_BASE_URL=http://127.0.0.1:8081 python scripts/sovereignty_smoke.py`

| Prompt | Latency | In tokens | Out tokens | Throughput |
|---|---:|---:|---:|---:|
| factual_concise | 6.6 s | 41 | 53 | 7.98 tok/s |
| reasoning_multistep | 56.4 s | 104 | 512 (truncated) | 9.08 tok/s |
| format_structured | 9.5 s | 78 | 78 | 8.17 tok/s |

**Mean output throughput**: ~8.4 tok/s on CPU (AMD Ryzen 9 5900X, 12 cores / 24 threads, 20 inference threads).

**Qualitative notes**
- `factual_concise`: correct answer, but two sentences where one was requested → minor format-compliance miss.
- `reasoning_multistep`: model's arithmetic is correct through the truncation point ($3,456 input cost, would have summed to ~$12,096/month). The suite's original `ground_truth: "~$363/month"` was wrong — the cited formula dropped per-event token counts; corrected to `~$12,096/month` in the same commit as this run-log entry. **First useful finding of the research arc: the eval suite checks itself before it checks the model.**
- `format_structured`: valid JSON, all keys present, model self-named accurately. Clean format compliance.

**First eval run** — 2026-05-26, adapter-only mode (no rubric judge yet)

```bash
MODEL_UNDER_TEST=hauhaucs \
  ANIMUS_SOVEREIGNTY_BASE_URL=http://127.0.0.1:8081 \
  animus-forge eval run sovereignty-stack-v1 \
    --adapter eval_suites.adapters.sovereignty_stack_v0:run_query \
    --suites-dir eval_suites \
    --prompt-version v0-smoke
```

- 5/5 PASS, total 155.6 s wall-clock
- Run stored as `005cd045`
- Score = 1.0 per case because the suite has empty `metrics: []` (intended — v0 measures rubric scores, not regression invariants). This run validates the pipeline (adapter → llama-server → store), not output quality.

**Next**: full rubric run

```bash
# Cloud baseline UUT (Claude Sonnet), cloud judge (Haiku)
MODEL_UNDER_TEST=claude-sonnet-4-6 \
  animus-forge eval run sovereignty-stack-v1 \
    --adapter eval_suites.adapters.sovereignty_stack_v0:run_query \
    --rubric personal-quality \
    --model claude-haiku-4-5 \
    --prompt-version v0-cloud-baseline

# Local UUT (Qwen3.6 via llama-server), cloud judge (Haiku)
MODEL_UNDER_TEST=hauhaucs \
  ANIMUS_SOVEREIGNTY_BASE_URL=http://127.0.0.1:8081 \
  animus-forge eval run sovereignty-stack-v1 \
    --adapter eval_suites.adapters.sovereignty_stack_v0:run_query \
    --rubric personal-quality \
    --model claude-haiku-4-5 \
    --prompt-version v0-local-uut

# A/B compare
animus-forge eval compare v0-cloud-baseline v0-local-uut --suite sovereignty-stack-v1
```

**Sovereign A/B — first run, 2026-05-26** (UUT = HauhauCS Qwen3.6 via llama-server, judge = same instance, personal-quality rubric):

```bash
MODEL_UNDER_TEST=hauhaucs \
  ANIMUS_SOVEREIGNTY_BASE_URL=http://127.0.0.1:8081 \
  animus-forge eval run sovereignty-stack-v1 \
    --adapter eval_suites.adapters.sovereignty_stack_v0:run_query \
    --rubric personal-quality \
    --model hauhaucs \
    --suites-dir eval_suites \
    --prompt-version v0-sovereign
```

- **5/5 PASS** at threshold 70%
- **Composite avg: 85.00%**, duration 313 s wall-clock
- Run stored as `5d643147`

| Case | Composite | Notes |
|---|---:|---|
| factual_concise | 76.67% | Format-compliance dim docked it — answer was correct but two sentences where the prompt asked for one |
| reasoning_multistep | 91.67% | Same arithmetic path as the smoke run (which matches the corrected ground_truth) |
| format_structured | 93.33% | Clean JSON, all keys present |
| domain_specific_tiaid | 80.00% | Engaged with the methodology framing |
| adversarial_edge | 83.33% | Did not refuse the authorized-red-team scenario — the uncensored model's value proposition |

**Second sovereign run — 2026-05-27** (same command after bumping `reasoning_multistep.max_tokens` 512 → 800 to avoid mid-arithmetic truncation):

- 5/5 PASS, **composite avg 85.33%**, run id `4b22354a`
- `reasoning_multistep` climbed 91.67% → **96.67%** with the extra budget — the targeted case improvement is the signal
- Other cases moved within ±3-5%: normal sample variance at temperature 0.2

Caveat: this is a single-model A/B (model judges itself). Self-judge favors the model — Anthropic Haiku as a cross-judge will tighten the floor. Run that pair once the Anthropic key is refreshed.

The auto-registration of `LlamaCppProvider` from `ANIMUS_SOVEREIGNTY_BASE_URL` is what closes the sovereign-judge gap — both UUT and judge route through the same llama-server instance with no cloud dependency.

## Provenance

- Model card: https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive
- License: Apache-2.0
- Base claimed: `Qwen/Qwen3.6-35B-A3B` (community fine-tune — "Qwen3.6" is HauhauCS's own naming, not an official Qwen release)
- Quant pulled: Q6_K_P (~31GB)
- Inference: CPU-only on this host (125GB RAM, no GPU); MoE active params keep token throughput tractable
- Storage: relocated to `/mnt/data/ollama/.ollama/models` via systemd drop-in (2026-05-24, after `/dev/sda` reformat to ext4)
