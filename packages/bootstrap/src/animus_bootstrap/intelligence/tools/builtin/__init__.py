"""Built-in tool definitions for the Animus Bootstrap intelligence layer."""

from __future__ import annotations

from animus_bootstrap.intelligence.tools.builtin.code_edit import get_code_edit_tools
from animus_bootstrap.intelligence.tools.builtin.filesystem import get_filesystem_tools
from animus_bootstrap.intelligence.tools.builtin.forge_ctl import get_forge_tools
from animus_bootstrap.intelligence.tools.builtin.gateway_tools import get_gateway_tools
from animus_bootstrap.intelligence.tools.builtin.identity_tools import get_identity_tools
from animus_bootstrap.intelligence.tools.builtin.memory_tools import get_memory_tools
from animus_bootstrap.intelligence.tools.builtin.self_improve import get_self_improve_tools
from animus_bootstrap.intelligence.tools.builtin.system import get_system_tools
from animus_bootstrap.intelligence.tools.builtin.timer_ctl import get_timer_tools
from animus_bootstrap.intelligence.tools.builtin.web import get_web_tools
from animus_bootstrap.intelligence.tools.executor import ToolDefinition


def get_all_builtin_tools() -> list[ToolDefinition]:
    """Collect and return all built-in tool definitions."""
    tools: list[ToolDefinition] = []
    tools.extend(get_web_tools())
    tools.extend(get_filesystem_tools())
    tools.extend(get_system_tools())
    tools.extend(get_gateway_tools())
    tools.extend(get_memory_tools())
    tools.extend(get_code_edit_tools())
    tools.extend(get_forge_tools())
    tools.extend(get_timer_tools())
    tools.extend(get_self_improve_tools())
    tools.extend(get_identity_tools())
    return tools


__all__ = ["get_all_builtin_tools"]
