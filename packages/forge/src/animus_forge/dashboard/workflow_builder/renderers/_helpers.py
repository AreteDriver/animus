"""Shared helper functions for renderers."""

from __future__ import annotations

import streamlit as st


def _get_node_execution_status(node_id: str) -> tuple[str, str]:
    """Get execution status and color for a node."""
    status_info = st.session_state.builder_execution_step_status.get(node_id, {})
    status = status_info.get("status", "")

    status_colors = {
        "pending": ("#f59e0b", "\u23f3"),
        "running": ("#3b82f6", "\U0001f504"),
        "completed": ("#10b981", "\u2705"),
        "failed": ("#ef4444", "\u274c"),
        "skipped": ("#6b7280", "\u23ed\ufe0f"),
    }
    return status_colors.get(status, ("", ""))
