"""P1-P9 validator stub — spec R13 + RD2 + A18.

v0 stub: returns `Allow` unconditionally and emits the documented log
line. Validator module path, call signature, and hook registration are
in place so the follow-on hardening spec only swaps the implementation.
"""

from __future__ import annotations

import logging

from ...types import AgentContext, Allow, HookDecision

logger = logging.getLogger(__name__)

STUB_LOG_MESSAGE = "P1-P9 validator stubbed; enforcement deferred to follow-on hardening spec"


def pre_tool_use(context: AgentContext, tool: str, args: dict) -> HookDecision:
    """Stub validator — always allows; emits stub-notice log once per process."""
    # Per A18 the test captures the log output. We log every invocation so
    # the receipt / observer can confirm the hook ran for every tool use.
    logger.info(STUB_LOG_MESSAGE)
    return Allow()
