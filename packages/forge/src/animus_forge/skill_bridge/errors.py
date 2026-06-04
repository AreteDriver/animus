"""Exceptions raised by SkillBridge."""

from __future__ import annotations


class SkillBridgeError(Exception):
    """Base class for SkillBridge errors."""


class SkillResolutionError(SkillBridgeError):
    """Skill could not be resolved from registry or filesystem.

    Raised on: unknown skill name, tier mismatch, out-of-repo path,
    missing SKILL.md.
    """


class SkillConfigError(SkillBridgeError):
    """Workflow YAML step config invalid (missing skill/tier, wrong type)."""


class SkillInputError(SkillBridgeError):
    """Inputs for an agent-tier skill violate schema.yaml.inputs."""


class SkillOutputDriftError(SkillBridgeError):
    """Provider output failed validation against output.schema.yaml."""


class SkillNestingDepthError(SkillBridgeError):
    """Workflow-tier nesting exceeded SKILLBRIDGE_MAX_NESTING_DEPTH."""


class SkillBudgetError(SkillBridgeError):
    """Budget allocation failed; step did not run."""


class SkillApprovalDeniedError(SkillBridgeError):
    """Destructive-risk skill was denied via approval pipeline."""


class SkillApprovalTimeoutError(SkillBridgeError):
    """Destructive-risk skill approval did not arrive within the window."""
