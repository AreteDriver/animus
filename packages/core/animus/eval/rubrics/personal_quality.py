"""Personal Quality rubric: 6 dimensions from Animus eval standard.

Maps the existing personal-quality scoring dimensions to formal
rubric criteria with 5-level scales. Used for evaluating agent
outputs against the established Animus quality bar.
"""

from animus.eval.rubric import Dimension, Rubric, ScoreLevel


def create_personal_quality_rubric() -> Rubric:
    """Create the standard personal-quality evaluation rubric.

    Dimensions:
    1. Relevance — how well the output addresses the specific request
    2. Precision — accuracy and correctness of claims/facts
    3. Actionability — how easy it is to act on the output
    4. Evidence Quality — strength of supporting reasoning/data
    5. Format Compliance — adherence to requested structure/style
    6. Hallucination Safety — absence of unsupported/fabricated claims
    """
    return (
        Rubric(
            name="personal-quality",
            description="Six-dimension quality assessment for Animus outputs",
            task_type="general",
            version="1.0",
        )
        .add_dimension(
            Dimension(
                name="relevance",
                description="How well the output addresses the specific request",
                weight=1.2,
                criteria={
                    ScoreLevel.CRITICAL: "Completely misses the point; addresses a different question",
                    ScoreLevel.POOR: "Partially relevant but misses major aspects of the request",
                    ScoreLevel.ACCEPTABLE: "Addresses the request but includes tangential content",
                    ScoreLevel.GOOD: "Directly addresses all aspects of the request",
                    ScoreLevel.EXCELLENT: "Directly addresses the request with insightful, targeted content",
                },
            )
        )
        .add_dimension(
            Dimension(
                name="precision",
                description="Accuracy and correctness of claims, facts, and reasoning",
                weight=1.5,
                required=True,
                criteria={
                    ScoreLevel.CRITICAL: "Contains factual errors or incorrect reasoning",
                    ScoreLevel.POOR: "Mostly correct but contains significant inaccuracies",
                    ScoreLevel.ACCEPTABLE: "Generally correct with minor errors",
                    ScoreLevel.GOOD: "Accurate with well-reasoned claims",
                    ScoreLevel.EXCELLENT: "Demonstrably correct with rigorous reasoning and citations",
                },
            )
        )
        .add_dimension(
            Dimension(
                name="actionability",
                description="How easy it is to act on the output",
                weight=1.0,
                criteria={
                    ScoreLevel.CRITICAL: "No actionable content; purely abstract",
                    ScoreLevel.POOR: "Vague suggestions without concrete steps",
                    ScoreLevel.ACCEPTABLE: "Contains some actionable steps but lacks detail",
                    ScoreLevel.GOOD: "Clear, actionable steps with sufficient detail",
                    ScoreLevel.EXCELLENT: "Immediately actionable with explicit next steps and dependencies",
                },
            )
        )
        .add_dimension(
            Dimension(
                name="evidence_quality",
                description="Strength of supporting reasoning, data, and citations",
                weight=1.1,
                criteria={
                    ScoreLevel.CRITICAL: "No evidence or reasoning provided",
                    ScoreLevel.POOR: "Weak reasoning with unsupported assertions",
                    ScoreLevel.ACCEPTABLE: "Some evidence but gaps in reasoning",
                    ScoreLevel.GOOD: "Well-supported with relevant evidence and reasoning",
                    ScoreLevel.EXCELLENT: "Compelling evidence with citations, data, and clear reasoning chains",
                },
            )
        )
        .add_dimension(
            Dimension(
                name="format_compliance",
                description="Adherence to requested structure, format, and style",
                weight=0.8,
                criteria={
                    ScoreLevel.CRITICAL: "Completely ignores format requirements",
                    ScoreLevel.POOR: "Major format violations",
                    ScoreLevel.ACCEPTABLE: "Mostly compliant with minor deviations",
                    ScoreLevel.GOOD: "Fully compliant with requested format",
                    ScoreLevel.EXCELLENT: "Exceeds format requirements with professional presentation",
                },
            )
        )
        .add_dimension(
            Dimension(
                name="hallucination_safety",
                description="Absence of unsupported, fabricated, or confidently wrong claims",
                weight=1.5,
                required=True,
                criteria={
                    ScoreLevel.CRITICAL: "Contains fabricated facts presented as true",
                    ScoreLevel.POOR: "Several unsupported claims stated with confidence",
                    ScoreLevel.ACCEPTABLE: "Minor unsupported claims, mostly hedged appropriately",
                    ScoreLevel.GOOD: "All claims are supported or appropriately qualified",
                    ScoreLevel.EXCELLENT: "Rigorous epistemic calibration: uncertainty clearly signaled",
                },
            )
        )
    )


PERSONAL_QUALITY_RUBRIC = create_personal_quality_rubric()