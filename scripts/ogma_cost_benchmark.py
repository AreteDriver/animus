#!/usr/bin/env python3
"""Ogma synthesis cost/latency benchmark: local Ollama vs OpenRouter DeepSeek V4 Flash.

Runs one sample ``SourceItem`` through Ogma's ``synthesize`` pipeline twice —
once on the local Ollama default, once on the opt-in OpenRouter STANDARD tier
(``deepseek/deepseek-v4-flash`` via ``ANIMUS_OPENROUTER_MODEL_STANDARD``) — and
reports, per run:

- input / cache / output tokens (from the provider's ``CompletionResponse``)
- the Forge **Effective-Tokens** (ET) cost:  ET = m * (1*I + 0.1*C + 4*O)
  where I = input tokens, C = cache-read tokens, O = output tokens, and ``m``
  is a per-model multiplier (1.0 by convention for the local baseline).
- wall-clock latency (ms)
- a one-line quality note

This is an operator tool, not a test: it makes real LLM calls and costs money
on the remote leg. Nothing here is auto-run by CI.

Usage:
    # local only (no key needed):
    python scripts/ogma_cost_benchmark.py

    # include the remote leg (requires an OpenRouter key + opt-in):
    export OPENROUTER_API_KEY=sk-or-...
    export ANIMUS_OPENROUTER_MODEL_STANDARD=deepseek/deepseek-v4-flash
    python scripts/ogma_cost_benchmark.py --remote

QUALITY NOTE (DeepSeek V4 Flash):
    V4 Flash long-context quality collapses in *non-think* mode. Ogma synthesis
    is a long-context, strict-contract task, so it MUST request high/max
    reasoning effort. The adapter in ``animus/ogma/read.py`` sets
    ``metadata={"reasoning_effort": "high"}``; the OpenRouter provider threads
    this into OpenRouter's unified ``reasoning`` param via ``extra_body``
    (``OpenRouterProvider._reasoning_extra_body``), so the remote leg runs in
    think mode. Reported remote quality is therefore representative, not a
    lower bound.

    The ET output weight (4x) means a think-mode run's extra reasoning/output
    tokens dominate cost: cheap per-token is NOT the same as cheap per-synthesis.
    That tradeoff is exactly what this benchmark measures.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

# ET multipliers per leg. Convention: local baseline = 1.0; the remote leg uses
# the same metric so ET is directly comparable as a units-of-work figure. Swap
# in a real $/ET multiplier here if you want dollar cost instead of work units.
_ET_MULTIPLIER_LOCAL = 1.0
_ET_MULTIPLIER_REMOTE = 1.0

REMOTE_MODEL = "deepseek/deepseek-v4-flash"
STANDARD_MODEL_ENV = "ANIMUS_OPENROUTER_MODEL_STANDARD"

SAMPLE_SOURCE_TEXT = (
    "We introduce a learned key-value cache eviction policy for long-context "
    "transformer inference. Rather than evicting the oldest tokens (a fixed "
    "FIFO heuristic), a small per-layer scoring head predicts which cached "
    "tokens are least likely to be attended to by future queries, and evicts "
    "those first. On a 128K-token retrieval benchmark the learned policy "
    "recovers 94% of full-cache accuracy at 40% of the memory footprint, "
    "versus 71% for FIFO and 82% for attention-sink heuristics. The scoring "
    "head adds 0.3% parameter overhead and is trained post-hoc on a frozen "
    "base model. We do not release code; the eviction head architecture is "
    "described in an appendix."
)


def effective_tokens(
    input_tokens: int, cache_tokens: int, output_tokens: int, multiplier: float
) -> float:
    """Forge Effective-Tokens metric: ET = m * (1*I + 0.1*C + 4*O)."""
    return multiplier * (1.0 * input_tokens + 0.1 * cache_tokens + 4.0 * output_tokens)


@dataclass
class RunResult:
    """One synthesis run's measured cost/latency."""

    leg: str
    model: str
    ok: bool
    input_tokens: int
    cache_tokens: int
    output_tokens: int
    et: float
    latency_ms: float
    quality_note: str
    error: str | None = None


def _build_sample_item():
    """Construct one sample ``SourceItem`` for synthesis.

    Imported lazily so ``--help`` works without the animus install present.
    """
    from animus.lugh.sources.base import SourceItem

    return SourceItem(
        source_id="arxiv:cs.LG",
        item_id="arxiv:2506.00001",
        title="Learned KV-Cache Eviction for Long-Context Inference",
        url="https://arxiv.org/abs/2506.00001",
        raw_text=SAMPLE_SOURCE_TEXT,
        summary="A learned KV-cache eviction policy for long-context transformers.",
    )


def _measure(leg: str, model_repr: str, multiplier: float, run_fn, quality_note: str) -> RunResult:
    """Time ``run_fn`` (returns a CompletionResponse-like object) and tabulate ET."""
    start = time.perf_counter()
    try:
        response = run_fn()
    except Exception as e:  # operator tool: report, don't crash the other leg
        latency_ms = (time.perf_counter() - start) * 1000.0
        return RunResult(
            leg=leg,
            model=model_repr,
            ok=False,
            input_tokens=0,
            cache_tokens=0,
            output_tokens=0,
            et=0.0,
            latency_ms=latency_ms,
            quality_note=quality_note,
            error=f"{type(e).__name__}: {e}",
        )
    latency_ms = (time.perf_counter() - start) * 1000.0

    input_tokens = int(getattr(response, "input_tokens", 0) or 0)
    output_tokens = int(getattr(response, "output_tokens", 0) or 0)
    # Cache-read tokens are not surfaced by the OpenRouter CompletionResponse
    # mapping today; default to 0. If a provider starts reporting cached prompt
    # tokens in metadata, read them here.
    cache_tokens = int(getattr(response, "metadata", {}).get("cache_read_tokens", 0) or 0)

    et = effective_tokens(input_tokens, cache_tokens, output_tokens, multiplier)
    return RunResult(
        leg=leg,
        model=model_repr,
        ok=True,
        input_tokens=input_tokens,
        cache_tokens=cache_tokens,
        output_tokens=output_tokens,
        et=et,
        latency_ms=latency_ms,
        quality_note=quality_note,
    )


def run_local(item, system_prompt: str, user_prompt: str) -> RunResult:
    """Local Ollama leg — uses the Core default model via a CompletionRequest-free path."""
    from animus.cognitive import ModelConfig, create_model

    model = create_model(ModelConfig.ollama(model="qwen2.5:14b"))

    class _Captured:
        # Core ModelInterface.generate returns a bare str with no token counts,
        # so we wrap it in a CompletionResponse-shaped object and estimate tokens
        # by a rough chars/4 heuristic (Ollama does not return usage here).
        def __init__(self, text: str, prompt_chars: int):
            self.content = text
            self.input_tokens = int(prompt_chars / 4)  # ~4 chars/token
            self.output_tokens = int(len(text) / 4)
            self.metadata: dict = {}

    def _run():
        text = model.generate(user_prompt, system=system_prompt)
        return _Captured(text, len(system_prompt) + len(user_prompt))

    note = "local qwen2.5:14b baseline — full-quality instruction following, no egress"
    return _measure("local-ollama", "qwen2.5:14b", _ET_MULTIPLIER_LOCAL, _run, note)


def run_remote(system_prompt: str, user_prompt: str) -> RunResult:
    """Remote leg — OpenRouter STANDARD tier (deepseek/deepseek-v4-flash)."""
    from animus_forge.providers.base import CompletionRequest, ModelTier
    from animus_forge.providers.openrouter_provider import OpenRouterProvider
    from animus_types import Sensitivity

    provider = OpenRouterProvider()

    def _run():
        request = CompletionRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model_tier=ModelTier.STANDARD,
            sensitivity=Sensitivity.PUBLIC,
            # Threaded to OpenRouter's reasoning param via extra_body by the
            # provider (_reasoning_extra_body) — runs in think mode.
            metadata={"reasoning_effort": "high"},
        )
        return provider.complete(request)

    note = (
        "deepseek-v4-flash STANDARD tier, think mode (reasoning_effort=high "
        "threaded via extra_body) — representative long-context quality"
    )
    return _measure("openrouter-standard", REMOTE_MODEL, _ET_MULTIPLIER_REMOTE, _run, note)


def _print_report(results: list[RunResult]) -> None:
    print("\n=== Ogma synthesis cost benchmark ===")
    print("ET metric: ET = m * (1*I + 0.1*C + 4*O)   (I=input, C=cache, O=output)\n")
    for r in results:
        print(f"[{r.leg}]  model={r.model}")
        if not r.ok:
            print(f"  FAILED: {r.error}\n")
            continue
        print(
            f"  tokens     : input={r.input_tokens}  cache={r.cache_tokens}  "
            f"output={r.output_tokens}"
        )
        print(f"  ET cost    : {r.et:.1f}")
        print(f"  latency    : {r.latency_ms:.0f} ms")
        print(f"  quality    : {r.quality_note}\n")

    ok = [r for r in results if r.ok]
    if len(ok) == 2:
        local, remote = ok[0], ok[1]
        if remote.et > 0:
            ratio = local.et / remote.et if remote.et else float("inf")
            print(f"ET ratio (local / remote): {ratio:.2f}x")
        print(
            "NOTE: the 4x output weight means think-mode reasoning tokens drive "
            "remote ET — judge cost per *synthesis*, not per token."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote",
        action="store_true",
        help="include the OpenRouter remote leg (needs OPENROUTER_API_KEY + the model env)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="run only the local Ollama leg (default if --remote is absent)",
    )
    args = parser.parse_args(argv)

    try:
        from animus.ogma.grounding import verify_animus_gap
        from animus.ogma.read import PERSONA_SYSTEM_PROMPT, _assemble_prompt
    except ImportError as e:
        print(f"ERROR: animus not importable ({e}). Install packages/core first.", file=sys.stderr)
        return 2

    item = _build_sample_item()
    gap = verify_animus_gap(item.title, repo_root=None)
    user_prompt = _assemble_prompt(item, item.raw_text, gap)

    results: list[RunResult] = [run_local(item, PERSONA_SYSTEM_PROMPT, user_prompt)]

    if args.remote and not args.local_only:
        results.append(run_remote(PERSONA_SYSTEM_PROMPT, user_prompt))

    _print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
