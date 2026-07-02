"""Animus Head — local-first agentic REPL loop.

A conversational interface that sits atop the Animus kernel and uses
local Ollama models for reasoning, tool selection, and execution.
Zero cloud API calls in default mode.
"""

from .checkpoint import HeadCheckpoint, HeadCheckpointStore
from .context_manager import ContextStats, HeadContextManager
from .fallback_controller import FallbackStatus, HeadFallbackController
from .quality_gate import HeadQualityGate, QualityScore
from .repl import HeadREPL
from .session_bootstrap import SessionBootstrap
from .tool_orchestrator import HeadToolOrchestrator

__all__ = [
    "HeadREPL",
    "SessionBootstrap",
    "HeadToolOrchestrator",
    "HeadContextManager",
    "ContextStats",
    "HeadQualityGate",
    "QualityScore",
    "HeadFallbackController",
    "FallbackStatus",
    "HeadCheckpoint",
    "HeadCheckpointStore",
]
