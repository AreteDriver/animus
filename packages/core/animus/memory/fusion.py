"""Rank fusion helpers for hybrid memory retrieval.

Currently exposes `_rrf_fuse` (Reciprocal Rank Fusion). Future variants
(confidence-weighted fusion, per-source calibration) land here alongside.
"""

from __future__ import annotations


def _rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion — merge multiple ranked ID lists.

    Args:
        ranked_lists: Each inner list is memory IDs in rank order.
        k: RRF constant (default 60, standard value).

    Returns:
        Fused list of memory IDs sorted by combined RRF score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, mem_id in enumerate(ranked):
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
