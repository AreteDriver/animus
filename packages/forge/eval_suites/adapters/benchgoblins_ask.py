"""Agent adapter for the benchgoblins-ask eval suite.

Wires the eval harness to BenchGoblins' `ClaudeService.make_decision`.
Imports BG via sys.path insertion rather than HTTP so the suite can run
without a live BG server — matches the "live LLM calls against local
code" regression-test pattern.

Environment:
    BENCHGOBLINS_PATH: Override for BG `src/api` location. Defaults to
        `/home/arete/projects/BenchGoblins/src/api`.
    ANTHROPIC_API_KEY: Required. `make_decision` raises RuntimeError
        without it.
    EVAL_USE_CACHE: "1" to let BG's TTL cache serve responses (speeds up
        iterative dev). Default "0" — cache disabled so each run hits
        the API fresh, which is what CI wants.

Usage from the runner:
    from eval_suites.adapters.benchgoblins_ask import run_ask
    from animus_forge.evaluation.base import AgentEvaluator
    evaluator = AgentEvaluator(run_ask, threshold=0.75)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

_DEFAULT_BG_PATH = Path(os.getenv("BENCHGOBLINS_PATH", "/home/arete/projects/BenchGoblins/src/api"))


def _ensure_bg_importable() -> None:
    """Insert BG's src/api on sys.path once per process."""
    path_str = str(_DEFAULT_BG_PATH)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _get_service():
    """Lazy ClaudeService singleton. Deferred so import-time env checks
    don't crash when the adapter module is merely loaded (e.g. during
    test collection without ANTHROPIC_API_KEY).
    """
    _ensure_bg_importable()
    from services.claude import ClaudeService  # type: ignore[import-not-found]

    if not hasattr(_get_service, "_instance"):
        _get_service._instance = ClaudeService()
    return _get_service._instance


def _extract_shape_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Project the parsed response onto only the fields the suite's
    metrics inspect. Keeps the JSON string the metrics regex against
    stable as BG internals change (e.g. token tracking fields).
    """
    return {
        "decision": result.get("decision"),
        "confidence": result.get("confidence"),
        "rationale": result.get("rationale"),
        "details": result.get("details"),
        "source": result.get("source"),
    }


def run_ask(case_input: str | dict) -> str:
    """Eval agent adapter.

    Args:
        case_input: Eval case input. Either a dict matching the
            ``ClaudeService.make_decision`` kwargs (query, sport,
            risk_mode, decision_type, league_type, player_a, player_b),
            or a plain string treated as `query` with `sport=nfl`,
            `risk_mode=median`, `decision_type=start_sit` defaults.

    Returns:
        JSON string of the parsed response, projected onto the fields
        the suite's structural metrics inspect. The suite's regex
        patterns are anchored to this shape.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not set. Propagated from
            `ClaudeService.make_decision`.
        ImportError: If BG's src/api is not at BENCHGOBLINS_PATH.
    """
    if isinstance(case_input, str):
        kwargs = {
            "query": case_input,
            "sport": "nfl",
            "risk_mode": "median",
            "decision_type": "start_sit",
        }
    else:
        kwargs = dict(case_input)

    # Force cache off by default for deterministic eval runs.
    use_cache = os.getenv("EVAL_USE_CACHE", "0") == "1"
    kwargs.setdefault("use_cache", use_cache)

    service = _get_service()
    result = asyncio.run(service.make_decision(**kwargs))
    projected = _extract_shape_fields(result)
    return json.dumps(projected)


__all__ = ["run_ask"]
