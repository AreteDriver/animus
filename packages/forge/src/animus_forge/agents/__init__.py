"""AI Agents for autonomous task orchestration.

This module provides intelligent agents that can analyze user requests,
delegate to specialized sub-agents, and synthesize results.
"""

from .agent_config import AgentConfig, get_agent_config
from .convergence import HAS_CONVERGENT, ConvergenceResult, DelegationConvergenceChecker
from .provider_wrapper import AgentProvider, create_agent_provider
from .subagent_manager import AgentRun, RunStatus, SubAgentManager
from .supervisor import AgentDelegation, SupervisorAgent

__all__ = [
    "AgentConfig",
    "AgentDelegation",
    "AgentProvider",
    "AgentRun",
    "ConvergenceResult",
    "DelegationConvergenceChecker",
    "HAS_CONVERGENT",
    "RunStatus",
    "SubAgentManager",
    "SupervisorAgent",
    "create_agent_provider",
    "get_agent_config",
]
