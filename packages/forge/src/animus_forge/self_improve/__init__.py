"""Self-Improvement system for Gorgon.

This module allows Gorgon to analyze and improve its own codebase
under strict safety constraints.

Safety Features:
- Protected files that cannot be modified
- Limits on changes per PR
- Mandatory test passing
- Human approval gates at plan, apply, and merge stages
- Automatic rollback on failures
"""

from .analyzer import CodebaseAnalyzer, ImprovementSuggestion
from .approval import ApprovalGate, ApprovalStatus
from .orchestrator import SelfImproveOrchestrator
from .pr_manager import PRManager, PRStatus
from .rollback import RollbackManager, Snapshot
from .safety import SafetyChecker, SafetyConfig
from .sandbox import Sandbox, SandboxResult

__all__ = [
    # Safety
    "SafetyConfig",
    "SafetyChecker",
    # Analysis
    "CodebaseAnalyzer",
    "ImprovementSuggestion",
    # Orchestration
    "SelfImproveOrchestrator",
    # Sandbox
    "Sandbox",
    "SandboxResult",
    # Approval
    "ApprovalGate",
    "ApprovalStatus",
    # Rollback
    "RollbackManager",
    "Snapshot",
    # PR Management
    "PRManager",
    "PRStatus",
]
