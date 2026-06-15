"""Auto-promotion: close the Kaizen loop (roadmap B5).

When ``eval compare`` shows a statistically-significant *improvement* for a new
prompt version, stage a human-gated WorkflowEvolution pending patch pinning that
version. This turns the eval suite into the experiment runner the evolution loop
needs, while keeping a human in the approval gate — promotion only *proposes*, it
never applies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from animus_kernel.coordination.workflow_evolution import (
    WorkflowEvolution,
    WorkflowPatch,
    WorkflowPatchResult,
)

if TYPE_CHECKING:
    from animus_kernel.evaluation.compare import ComparisonReport


def auto_promote_on_improvement(
    comparison: ComparisonReport,
    *,
    workflow_id: str,
    prompt_version: str,
    new_yaml_content: str,
    evolution: WorkflowEvolution,
) -> WorkflowPatchResult | None:
    """Stage a pending patch iff the comparison is a significant improvement.

    Returns the ``WorkflowPatchResult`` from staging, or ``None`` when no
    promotion happens — i.e. the comparison is not significant (including the
    underpowered case), a regression, or no-change. Never auto-applies; the
    patch is staged for human approval.
    """
    if not comparison.significant or comparison.direction != "improvement":
        return None

    lo, hi = comparison.score_ci_95
    patch = WorkflowPatch(
        workflow_id=workflow_id,
        patch_type="update_metadata",
        change_description=(
            f"Auto-promote prompt version {prompt_version} (score Δ {comparison.score_delta:+.3f})"
        ),
        new_yaml_content=new_yaml_content,
        reasoning=(
            f"eval compare shows a statistically-significant improvement: "
            f"score Δ {comparison.score_delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] "
            f"(excludes 0). Staged for human review — not applied."
        ),
        proposed_by="auto-promotion",
        performance_evidence=(
            f"prompt_version={prompt_version}; score_delta={comparison.score_delta:+.3f}; "
            f"ci_95=[{lo:+.3f}, {hi:+.3f}]"
        ),
    )
    return evolution.propose_patch(patch)
