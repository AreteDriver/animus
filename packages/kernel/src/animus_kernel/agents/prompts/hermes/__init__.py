"""Hermes-format prompt loader for Animus kernel agents."""

from pathlib import Path
from typing import Final

_PROMPTS_DIR: Final = Path(__file__).resolve().parent


def get_role_prompt(role: str) -> str:
    """Load the Hermes system prompt for a given agent role.

    Args:
        role: Agent role name (planner, builder, tester, reviewer, architect, documenter).

    Returns:
        The XML system prompt string.

    Raises:
        ValueError: If the role is not supported.
    """
    path = _PROMPTS_DIR / f"{role.lower()}.xml"
    if not path.exists():
        raise ValueError(f"No Hermes prompt for role: {role}")
    return path.read_text(encoding="utf-8")


__all__ = ["get_role_prompt"]
