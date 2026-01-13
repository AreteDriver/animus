"""
Animus Decision Support Framework

Structured decision analysis with pros/cons and criteria-based evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.cognitive import CognitiveLayer

logger = get_logger("decision")


@dataclass
class Decision:
    """A structured decision with options and analysis."""

    question: str
    options: list[str]
    criteria: list[str]
    analysis: dict[str, dict[str, str]]  # option -> criterion -> assessment
    recommendation: str | None = None
    reasoning: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "options": self.options,
            "criteria": self.criteria,
            "analysis": self.analysis,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
        }

    def format_analysis(self) -> str:
        """Format analysis as readable text."""
        lines = [f"# Decision: {self.question}\n"]

        # Options summary
        lines.append("## Options")
        for i, opt in enumerate(self.options, 1):
            lines.append(f"{i}. {opt}")
        lines.append("")

        # Criteria
        if self.criteria:
            lines.append("## Criteria")
            for crit in self.criteria:
                lines.append(f"- {crit}")
            lines.append("")

        # Analysis matrix
        if self.analysis:
            lines.append("## Analysis")
            for option, assessments in self.analysis.items():
                lines.append(f"\n### {option}")
                for criterion, assessment in assessments.items():
                    lines.append(f"- **{criterion}**: {assessment}")

        # Recommendation
        if self.recommendation:
            lines.append(f"\n## Recommendation\n{self.recommendation}")

        if self.reasoning:
            lines.append(f"\n## Reasoning\n{self.reasoning}")

        return "\n".join(lines)


class DecisionFramework:
    """
    Framework for structured decision analysis.

    Uses CognitiveLayer to analyze options against criteria.
    """

    def __init__(self, cognitive: "CognitiveLayer"):
        self.cognitive = cognitive
        logger.debug("DecisionFramework initialized")

    def analyze(
        self,
        question: str,
        options: list[str] | None = None,
        criteria: list[str] | None = None,
    ) -> Decision:
        """
        Perform structured decision analysis.

        If options/criteria not provided, asks the model to identify them.

        Args:
            question: The decision question
            options: List of options to consider (auto-identified if None)
            criteria: Evaluation criteria (auto-identified if None)

        Returns:
            Decision object with full analysis
        """
        # If no options provided, ask model to identify them
        if not options:
            options = self._identify_options(question)

        # If no criteria provided, suggest relevant ones
        if not criteria:
            criteria = self._identify_criteria(question, options)

        # Analyze each option against each criterion
        analysis = self._analyze_options(question, options, criteria)

        # Generate recommendation
        recommendation, reasoning = self._make_recommendation(question, options, criteria, analysis)

        decision = Decision(
            question=question,
            options=options,
            criteria=criteria,
            analysis=analysis,
            recommendation=recommendation,
            reasoning=reasoning,
        )

        logger.info(f"Decision analysis complete for: {question[:50]}...")
        return decision

    def _identify_options(self, question: str) -> list[str]:
        """Use model to identify options from question."""
        prompt = f"""Given this decision question, identify 2-4 distinct options to consider.
Question: {question}

List only the options, one per line, no numbering or explanation."""

        response = self.cognitive.think(prompt)
        options = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return options[:4]  # Limit to 4 options

    def _identify_criteria(self, question: str, options: list[str]) -> list[str]:
        """Use model to suggest evaluation criteria."""
        options_text = ", ".join(options)
        prompt = f"""For this decision, suggest 3-5 important evaluation criteria.
Question: {question}
Options: {options_text}

List only the criteria, one per line, no numbering or explanation.
Focus on practical, measurable factors."""

        response = self.cognitive.think(prompt)
        criteria = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return criteria[:5]  # Limit to 5 criteria

    def _analyze_options(
        self,
        question: str,
        options: list[str],
        criteria: list[str],
    ) -> dict[str, dict[str, str]]:
        """Analyze each option against each criterion."""
        analysis = {}

        for option in options:
            option_analysis = {}
            for criterion in criteria:
                assessment = self._assess_option_criterion(question, option, criterion)
                option_analysis[criterion] = assessment
            analysis[option] = option_analysis

        return analysis

    def _assess_option_criterion(self, question: str, option: str, criterion: str) -> str:
        """Assess one option against one criterion."""
        prompt = f"""Briefly assess this option against this criterion (1-2 sentences).
Decision: {question}
Option: {option}
Criterion: {criterion}

Assessment:"""

        response = self.cognitive.think(prompt)
        # Clean up response
        return response.strip().replace("Assessment:", "").strip()

    def _make_recommendation(
        self,
        question: str,
        options: list[str],
        criteria: list[str],
        analysis: dict[str, dict[str, str]],
    ) -> tuple[str, str]:
        """Generate a recommendation based on the analysis."""
        # Format analysis for context
        analysis_text = []
        for option, assessments in analysis.items():
            analysis_text.append(f"\n{option}:")
            for crit, assess in assessments.items():
                analysis_text.append(f"  - {crit}: {assess}")

        prompt = f"""Based on this analysis, make a recommendation.
Question: {question}
Options: {", ".join(options)}
Criteria: {", ".join(criteria)}

Analysis:
{"".join(analysis_text)}

Provide:
1. Your recommendation (which option)
2. Brief reasoning (2-3 sentences)

Format:
RECOMMENDATION: [option]
REASONING: [explanation]"""

        response = self.cognitive.think(prompt)

        # Parse response
        recommendation = ""
        reasoning = ""

        for line in response.split("\n"):
            if line.upper().startswith("RECOMMENDATION:"):
                recommendation = line.split(":", 1)[1].strip()
            elif line.upper().startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # If parsing failed, use whole response as reasoning
        if not recommendation:
            recommendation = options[0] if options else "Unable to determine"
            reasoning = response.strip()

        return recommendation, reasoning

    def quick_decide(self, question: str) -> str:
        """
        Quick decision without full analysis.

        Args:
            question: Decision question

        Returns:
            Quick recommendation
        """
        prompt = f"""Provide a quick, practical recommendation for this decision.
Be direct and concise.

Decision: {question}

Recommendation:"""

        return self.cognitive.think(prompt)
