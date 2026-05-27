"""Smoke test for Qwen3.6 sovereignty-stack v0.

First runnable artifact of the local-AI-sovereignty research arc. Calls the
locally-hosted Qwen3.6-35B-A3B-Uncensored model through the standard Forge
OllamaProvider path and captures latency + token throughput for three
representative prompts.

Usage:
    cd packages/forge
    python scripts/sovereignty_smoke.py

The script is intentionally minimal:
- No BudgetManager (smoke only; BM is enforced at WorkflowExecutor level)
- No fallback wiring (this run is to characterize the local model in isolation)
- No judge — raw outputs are written to stdout for manual inspection

Output:
    Prompt N: ...
    Latency: NNN ms
    Tokens (in/out/total): a/b/c
    Throughput: N.NN tok/s
    Response:
        <content>

Captured metrics feed docs/SOVEREIGNTY_STACK.md for the v0 baseline.
"""

from __future__ import annotations

import sys
from time import perf_counter

from animus_forge.providers.base import CompletionRequest
from animus_forge.providers.ollama_provider import OllamaProvider

MODEL_TAG = "hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q6_K_P"

PROMPTS = [
    {
        "name": "factual_concise",
        "system": "Answer in one sentence. No preamble.",
        "user": "What is the difference between TRIM and a secure-erase on an SSD?",
    },
    {
        "name": "reasoning_multistep",
        "system": "Think step by step, then give a final answer in one line prefixed with 'ANSWER:'",
        "user": (
            "A workflow runs every 6 hours and processes ~12k events per run. "
            "Each event triggers exactly one LLM call at ~800 input + ~400 output tokens. "
            "At Anthropic Sonnet pricing ($3/MT in, $15/MT out), what is the monthly "
            "cost? Show the arithmetic."
        ),
    },
    {
        "name": "format_structured",
        "system": "Respond ONLY with the requested JSON. No prose.",
        "user": (
            "Return a JSON object with keys: model_name (string), "
            "tradeoff_summary (string, <=20 words), best_for (array of 3 strings). "
            "The model is yourself: local Qwen3.6-35B-A3B."
        ),
    },
]


def run() -> int:
    provider = OllamaProvider(model=MODEL_TAG)
    if not provider.is_configured():
        print(
            "ERROR: Ollama not reachable at http://localhost:11434",
            file=sys.stderr,
        )
        return 1

    available = provider.list_models()
    if not any(MODEL_TAG.split(":")[0] in m for m in available):
        print(
            f"ERROR: model not pulled. Run: ollama pull {MODEL_TAG}",
            file=sys.stderr,
        )
        print(f"Available models: {available}", file=sys.stderr)
        return 1

    provider.initialize()
    print("=== Sovereignty smoke v0 ===")
    print(f"Model: {MODEL_TAG}")
    print(f"Host: {provider.base_url}")
    print()

    for i, p in enumerate(PROMPTS, 1):
        print(f"--- Prompt {i}: {p['name']} ---")
        req = CompletionRequest(
            prompt=p["user"],
            system_prompt=p["system"],
            model=MODEL_TAG,
            temperature=0.2,
            max_tokens=512,
        )
        start = perf_counter()
        try:
            resp = provider.complete(req)
        except Exception as e:
            print(f"  FAILED: {e}")
            print()
            continue
        elapsed_ms = (perf_counter() - start) * 1000

        out_tokens = resp.output_tokens or 0
        in_tokens = resp.input_tokens or 0
        throughput = (out_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0

        print(f"  Latency: {elapsed_ms:.0f} ms")
        print(f"  Tokens (in/out/total): {in_tokens}/{out_tokens}/{in_tokens + out_tokens}")
        print(f"  Throughput: {throughput:.2f} tok/s")
        print("  Response:")
        for line in (resp.content or "").splitlines():
            print(f"    {line}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(run())
