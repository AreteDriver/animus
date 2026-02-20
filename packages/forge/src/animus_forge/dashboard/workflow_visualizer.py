"""Workflow visualizer component for Streamlit dashboard."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import streamlit as st


class StepStatus(Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VisualStep:
    """A step in the visual workflow."""

    id: str
    name: str
    type: str
    status: StepStatus = StepStatus.PENDING
    duration_ms: int | None = None
    error: str | None = None
    output_preview: str | None = None


# Status colors and icons
STATUS_COLORS = {
    StepStatus.PENDING: "#6c757d",
    StepStatus.RUNNING: "#007bff",
    StepStatus.COMPLETED: "#28a745",
    StepStatus.FAILED: "#dc3545",
    StepStatus.SKIPPED: "#ffc107",
}

STATUS_ICONS = {
    StepStatus.PENDING: "⏳",
    StepStatus.RUNNING: "🔄",
    StepStatus.COMPLETED: "✅",
    StepStatus.FAILED: "❌",
    StepStatus.SKIPPED: "⏭️",
}

STEP_TYPE_ICONS = {
    "claude_code": "🤖",
    "openai": "🧠",
    "shell": "💻",
    "github": "🐙",
    "notion": "📝",
    "slack": "💬",
    "checkpoint": "🏁",
    "condition": "🔀",
}


def render_workflow_visualizer(
    steps: list[dict[str, Any]],
    current_step: str | None = None,
    step_results: dict[str, Any] | None = None,
    compact: bool = False,
) -> None:
    """Render a visual workflow diagram.

    Args:
        steps: List of workflow step definitions
        current_step: ID of the currently executing step
        step_results: Results for completed steps
        compact: Use compact display mode
    """
    step_results = step_results or {}

    st.markdown("### 🔄 Workflow Progress")

    # Calculate progress
    completed = sum(
        1 for s in steps if step_results.get(s.get("id", ""), {}).get("status") == "completed"
    )
    total = len(steps)
    progress = completed / total if total > 0 else 0

    # Progress bar
    st.progress(progress, text=f"{completed}/{total} steps completed")

    if compact:
        _render_compact_view(steps, current_step, step_results)
    else:
        _render_detailed_view(steps, current_step, step_results)


def _render_compact_view(
    steps: list[dict[str, Any]],
    current_step: str | None,
    step_results: dict[str, Any],
) -> None:
    """Render compact horizontal workflow view."""
    cols = st.columns(min(len(steps), 6))

    for i, step in enumerate(steps):
        step_id = step.get("id", f"step_{i}")
        step_type = step.get("type", "unknown")
        result = step_results.get(step_id, {})

        # Determine status
        if step_id == current_step:
            status = StepStatus.RUNNING
        elif result.get("status") == "completed":
            status = StepStatus.COMPLETED
        elif result.get("status") == "failed":
            status = StepStatus.FAILED
        elif result.get("status") == "skipped":
            status = StepStatus.SKIPPED
        else:
            status = StepStatus.PENDING

        col_idx = i % 6
        with cols[col_idx]:
            icon = STEP_TYPE_ICONS.get(step_type, "📦")
            status_icon = STATUS_ICONS[status]
            color = STATUS_COLORS[status]

            st.markdown(
                f"""
                <div style="
                    text-align: center;
                    padding: 10px;
                    border: 2px solid {color};
                    border-radius: 8px;
                    margin: 5px 0;
                    background: {color}20;
                ">
                    <div style="font-size: 24px;">{icon}</div>
                    <div style="font-size: 12px; font-weight: bold;">{step_id}</div>
                    <div style="font-size: 16px;">{status_icon}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_detailed_view(
    steps: list[dict[str, Any]],
    current_step: str | None,
    step_results: dict[str, Any],
) -> None:
    """Render detailed vertical workflow view."""
    for i, step in enumerate(steps):
        step_id = step.get("id", f"step_{i}")
        step_type = step.get("type", "unknown")
        step_params = step.get("params", {})
        result = step_results.get(step_id, {})

        # Determine status
        if step_id == current_step:
            status = StepStatus.RUNNING
        elif result.get("status") == "completed":
            status = StepStatus.COMPLETED
        elif result.get("status") == "failed":
            status = StepStatus.FAILED
        elif result.get("status") == "skipped":
            status = StepStatus.SKIPPED
        else:
            status = StepStatus.PENDING

        icon = STEP_TYPE_ICONS.get(step_type, "📦")
        status_icon = STATUS_ICONS[status]
        color = STATUS_COLORS[status]

        # Step card
        with st.container():
            st.markdown(
                f"""
                <div style="
                    border-left: 4px solid {color};
                    padding: 10px 15px;
                    margin: 10px 0;
                    background: {color}10;
                    border-radius: 0 8px 8px 0;
                ">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 24px;">{icon}</span>
                        <span style="font-size: 18px; font-weight: bold;">{step_id}</span>
                        <span style="font-size: 12px; color: #666;">({step_type})</span>
                        <span style="margin-left: auto; font-size: 20px;">{status_icon}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show details in expander
            with st.expander("Details", expanded=(status == StepStatus.RUNNING)):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Parameters:**")
                    if step_params:
                        role = step_params.get("role", "N/A")
                        st.write(f"- Role: `{role}`")
                        if step_params.get("prompt"):
                            prompt_preview = (
                                step_params["prompt"][:100] + "..."
                                if len(step_params.get("prompt", "")) > 100
                                else step_params.get("prompt", "")
                            )
                            st.write(f"- Prompt: {prompt_preview}")

                with col2:
                    st.markdown("**Result:**")
                    if result:
                        if result.get("duration_ms"):
                            st.write(f"- Duration: {result['duration_ms']}ms")
                        if result.get("error"):
                            st.error(result["error"])
                        if result.get("output"):
                            output_preview = str(result["output"])[:200]
                            st.code(output_preview, language="text")
                    else:
                        st.write("- Waiting...")

        # Arrow between steps (except last)
        if i < len(steps) - 1:
            st.markdown(
                """
                <div style="text-align: center; color: #666; font-size: 20px;">
                    ↓
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_workflow_summary(
    workflow_name: str,
    total_steps: int,
    completed_steps: int,
    failed_steps: int,
    total_duration_ms: int | None = None,
    total_cost_usd: float | None = None,
) -> None:
    """Render workflow execution summary.

    Args:
        workflow_name: Name of the workflow
        total_steps: Total number of steps
        completed_steps: Number of completed steps
        failed_steps: Number of failed steps
        total_duration_ms: Total execution time in milliseconds
        total_cost_usd: Total cost in USD
    """
    st.markdown(f"### 📊 Summary: {workflow_name}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        success_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

    with col2:
        st.metric("Completed", f"{completed_steps}/{total_steps}")

    with col3:
        if total_duration_ms:
            duration_sec = total_duration_ms / 1000
            st.metric("Duration", f"{duration_sec:.1f}s")
        else:
            st.metric("Duration", "N/A")

    with col4:
        if total_cost_usd is not None:
            st.metric("Cost", f"${total_cost_usd:.4f}")
        else:
            st.metric("Cost", "N/A")

    # Status indicator
    if failed_steps > 0:
        st.error(f"❌ Workflow completed with {failed_steps} failed step(s)")
    elif completed_steps == total_steps:
        st.success("✅ Workflow completed successfully!")
    else:
        st.info(f"🔄 Workflow in progress ({completed_steps}/{total_steps} steps)")


def render_step_timeline(
    steps: list[dict[str, Any]],
    step_results: dict[str, Any],
) -> None:
    """Render a timeline view of step execution.

    Args:
        steps: List of workflow steps
        step_results: Results for each step
    """
    st.markdown("### 📅 Execution Timeline")

    timeline_data = []
    cumulative_time = 0

    for step in steps:
        step_id = step.get("id", "unknown")
        result = step_results.get(step_id, {})
        duration = result.get("duration_ms", 0)

        timeline_data.append(
            {
                "step": step_id,
                "start": cumulative_time,
                "duration": duration,
                "status": result.get("status", "pending"),
            }
        )
        cumulative_time += duration

    # Render as horizontal bars
    if timeline_data:
        max_time = max(1, cumulative_time)

        for item in timeline_data:
            width_pct = (item["duration"] / max_time * 100) if max_time > 0 else 0
            left_pct = (item["start"] / max_time * 100) if max_time > 0 else 0

            status = (
                StepStatus(item["status"])
                if item["status"] in [s.value for s in StepStatus]
                else StepStatus.PENDING
            )
            color = STATUS_COLORS[status]

            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 100px; font-size: 12px;">{item["step"]}</div>
                    <div style="flex: 1; height: 20px; background: #f0f0f0; border-radius: 4px; position: relative;">
                        <div style="
                            position: absolute;
                            left: {left_pct}%;
                            width: {max(width_pct, 1)}%;
                            height: 100%;
                            background: {color};
                            border-radius: 4px;
                        "></div>
                    </div>
                    <div style="width: 60px; text-align: right; font-size: 12px;">{item["duration"]}ms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_agent_activity(
    agents: list[dict[str, Any]],
    show_live: bool = True,
) -> None:
    """Render live agent activity feed.

    Args:
        agents: List of agent activities
        show_live: Show live update indicator
    """
    st.markdown("### 🤖 Agent Activity")

    if show_live:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 5px; margin-bottom: 10px;">
                <div style="width: 10px; height: 10px; background: #28a745; border-radius: 50%; animation: pulse 1s infinite;"></div>
                <span style="font-size: 12px; color: #666;">Live</span>
            </div>
            <style>
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if not agents:
        st.info("No agent activity yet")
        return

    for agent in agents[-10:]:  # Show last 10
        role = agent.get("role", "unknown")
        action = agent.get("action", "processing")
        timestamp = agent.get("timestamp", "")
        status = agent.get("status", "running")

        icon = (
            "🤖"
            if role == "builder"
            else "📋"
            if role == "planner"
            else "🧪"
            if role == "tester"
            else "👁️"
        )
        status_icon = "🔄" if status == "running" else "✅" if status == "completed" else "❌"

        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 8px;
                border-bottom: 1px solid #eee;
                font-size: 14px;
            ">
                <span>{icon}</span>
                <span style="font-weight: bold;">{role}</span>
                <span style="color: #666;">{action}</span>
                <span style="margin-left: auto;">{status_icon}</span>
                <span style="color: #999; font-size: 12px;">{timestamp}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
