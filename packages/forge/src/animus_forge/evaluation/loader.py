"""Suite loader with metric factory.

Extends EvalSuite.from_yaml() with automatic metric instantiation
from YAML config and suite discovery from the eval_suites/ directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import EvalMetric, EvalSuite
from .metrics import (
    ContainsMetric,
    ExactMatchMetric,
    LengthMetric,
    LLMJudgeMetric,
    RegexMatchMetric,
    SimilarityMetric,
)

logger = logging.getLogger(__name__)

# Default suites directory: <project_root>/eval_suites/
_DEFAULT_SUITES_DIR = Path(__file__).parent.parent.parent.parent / "eval_suites"

# Maps YAML type strings to metric constructors
METRIC_MAP: dict[str, type[EvalMetric]] = {
    "contains": ContainsMetric,
    "similarity": SimilarityMetric,
    "regex": RegexMatchMetric,
    "length": LengthMetric,
    "exact_match": ExactMatchMetric,
    "llm_judge": LLMJudgeMetric,
}


def _build_metric(spec: dict[str, Any]) -> EvalMetric | None:
    """Instantiate a metric from a YAML spec dict.

    Expected format: {"type": "contains", "case_sensitive": false, ...}
    The 'type' key selects the class; remaining keys are passed as kwargs.
    """
    metric_type = spec.get("type", "")
    cls = METRIC_MAP.get(metric_type)
    if cls is None:
        logger.warning("Unknown metric type '%s', skipping", metric_type)
        return None

    kwargs = {k: v for k, v in spec.items() if k != "type"}
    try:
        return cls(**kwargs)
    except TypeError as e:
        logger.warning("Failed to create metric '%s': %s", metric_type, e)
        return None


class SuiteLoader:
    """Discovers and loads evaluation suites from YAML files.

    Args:
        suites_dir: Directory containing suite YAML files.
            Defaults to ``eval_suites/`` in the project root.
    """

    def __init__(self, suites_dir: Path | None = None):
        self.suites_dir = suites_dir or _DEFAULT_SUITES_DIR

    def load_suite(self, name: str) -> EvalSuite:
        """Load a suite by name and instantiate its metrics.

        Args:
            name: Suite name (without .yaml extension).

        Returns:
            Fully configured EvalSuite with metrics attached.

        Raises:
            FileNotFoundError: If the suite YAML file does not exist.
        """
        path = self.suites_dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Suite not found: {path}")

        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        suite = EvalSuite(
            name=data.get("name", name),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            threshold=data.get("threshold", 0.7),
        )

        # Add cases
        for case_data in data.get("cases", []):
            suite.add_case(
                input=case_data["input"],
                expected=case_data.get("expected"),
                name=case_data.get("name", ""),
                **case_data.get("metadata", {}),
            )

        # Instantiate metrics from YAML spec
        for metric_spec in data.get("metrics", []):
            if isinstance(metric_spec, str):
                metric_spec = {"type": metric_spec}
            metric = _build_metric(metric_spec)
            if metric:
                suite.add_metric(metric)

        # Stash agent_role from YAML for CLI use
        suite.tags = suite.tags or []
        agent_role = data.get("agent_role")
        if agent_role:
            suite.tags.insert(0, f"role:{agent_role}")

        logger.debug(
            "Loaded suite '%s': %d cases, %d metrics",
            suite.name,
            len(suite.cases),
            len(suite.metrics),
        )
        return suite

    def list_suites(self) -> list[dict[str, Any]]:
        """List all available suites with summary info.

        Returns:
            List of dicts with name, description, agent_role, cases_count, threshold.
        """
        if not self.suites_dir.exists():
            return []

        import yaml

        suites = []
        for path in sorted(self.suites_dir.glob("*.yaml")):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                suites.append(
                    {
                        "name": path.stem,
                        "description": data.get("description", ""),
                        "agent_role": data.get("agent_role", ""),
                        "cases_count": len(data.get("cases", [])),
                        "threshold": data.get("threshold", 0.7),
                    }
                )
            except Exception as e:
                logger.warning("Failed to read suite %s: %s", path.name, e)

        return suites
