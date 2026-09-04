"""API routes for evaluation baselines and regression checks."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header

from animus_forge.api_errors import AUTH_RESPONSES, bad_request, not_found
from animus_forge.api_routes.auth import verify_auth
from animus_forge.evaluation.store import get_eval_store

router = APIRouter()


@router.get("/evals/baselines/{suite_name}", responses=AUTH_RESPONSES)
def get_baseline(
    suite_name: str,
    authorization: str | None = Header(None),
):
    """Get the current baseline for an eval suite."""
    verify_auth(authorization)

    store = get_eval_store()
    baseline = store.get_baseline(suite_name)
    if baseline is None:
        raise not_found("Baseline", suite_name)
    return {
        "suite_name": suite_name,
        "baseline": baseline,
    }


@router.post("/evals/baselines/{suite_name}", responses=AUTH_RESPONSES)
def set_baseline(
    suite_name: str,
    request: dict[str, Any],
    authorization: str | None = Header(None),
):
    """Set the gold baseline run for an eval suite.

    Body must include ``run_id``.
    """
    verify_auth(authorization)

    run_id = request.get("run_id")
    if not run_id:
        raise bad_request("run_id is required")

    store = get_eval_store()
    run = store.get_run(run_id)
    if run is None:
        raise not_found("Eval run", run_id)

    store.set_baseline(suite_name, run_id)
    return {"status": "baseline_set", "suite_name": suite_name, "run_id": run_id}


@router.get("/evals/baselines/{suite_name}/regression", responses=AUTH_RESPONSES)
def check_regression(
    suite_name: str,
    pass_rate: float,
    delta_threshold: float = 0.2,
    authorization: str | None = Header(None),
):
    """Check whether a pass_rate regresses against the stored baseline.

    Query params:
        pass_rate: Current pass rate to compare.
        delta_threshold: Delta that triggers regression (default 0.2).
    """
    verify_auth(authorization)

    store = get_eval_store()
    result = store.check_regression(
        suite_name,
        pass_rate,
        delta_threshold=delta_threshold,
    )
    return {
        "suite_name": suite_name,
        **result,
    }
