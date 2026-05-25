"""Discover and import user hook modules from a directory.

Per spec §4.8 / §4.1: hooks live in `~/.config/animus/agent/hooks.d/*.py`.
Each module may define any subset of `agent_start`, `pre_tool_use`,
`post_tool_use`, `agent_end`. Syntax errors raise during load (the CLI
maps that to exit 6 with the offending filename per §4.1 error cases).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

DEFAULT_HOOK_DIR = Path.home() / ".config" / "animus" / "agent" / "hooks.d"


def load_user_hooks(hook_dir: Path | None = None) -> list[ModuleType]:
    """Load `*.py` files from `hook_dir` in alphabetical order.

    Args:
        hook_dir: Directory to scan. Defaults to the spec path. A missing
            directory is treated as "no user hooks" — returns empty list,
            no error.

    Returns:
        List of imported modules, alphabetically by filename.

    Raises:
        SyntaxError | ImportError: surfaced from a broken hook file with
            the file path in the message (per §4.1: "exit 6, error names
            the file"). Caller (CLI) is responsible for the exit-code mapping.
    """
    root = hook_dir if hook_dir is not None else DEFAULT_HOOK_DIR
    if not root.exists() or not root.is_dir():
        return []

    modules: list[ModuleType] = []
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue  # skip __init__ etc.
        module_name = f"animus_forge_user_hook_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Annotate the exception with the file path so the CLI can
            # surface it cleanly in the §4.1 "exit 6, error names the file"
            # path. Re-raise so the caller knows the load failed.
            raise type(e)(f"{path}: {e}") from e
        modules.append(module)
    return modules
