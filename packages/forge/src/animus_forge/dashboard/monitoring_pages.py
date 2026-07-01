"""Monitoring and Metrics Dashboard Pages for Gorgon.

Provides real-time monitoring UI for orchestrator status,
agent coordination visualization, and metrics dashboards.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from animus_forge.analytics.pipeline import PipelineResult
    from animus_forge.monitoring.parallel_tracker import ParallelExecutionTracker
    from animus_forge.monitoring.tracker import AgentTracker, ExecutionTracker


# Import monitoring components - lazy load to avoid dep issues
def get_tracker() -> ExecutionTracker:
    from animus_forge.monitoring import get_tracker as _get_tracker

    return _get_tracker()


def get_agent_tracker() -> AgentTracker:
    from animus_forge.monitoring.tracker import AgentTracker

    if "agent_tracker" not in st.session_state:
        st.session_state.agent_tracker = AgentTracker()
    return st.session_state.agent_tracker


def get_parallel_tracker() -> ParallelExecutionTracker:
    """Get the parallel execution tracker."""
    from animus_forge.monitoring.parallel_tracker import (
        get_parallel_tracker as _get_parallel_tracker,
    )

    return _get_parallel_tracker()


def render_monitoring_page() -> None:
    """Render real-time monitoring page."""
    st.title("Real-Time Monitoring")

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        auto_refresh = st.toggle("Auto-refresh", value=True)

    if auto_refresh:
        time.sleep(0.1)  # Small delay for smoother updates
        st.rerun()

    try:
        tracker = get_tracker()
        data = tracker.get_dashboard_data()
    except Exception as e:
        st.warning(f"Monitoring data not available: {e}")
        data = {
            "summary": {
                "active_workflows": 0,
                "total_executions": 0,
                "success_rate": 0,
                "avg_duration_ms": 0,
            },
            "active_workflows": [],
            "recent_executions": [],
        }

    # Summary metrics
    summary = data["summary"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Active Workflows",
            summary.get("active_workflows", 0),
            delta=None,
        )
    with col2:
        st.metric(
            "Total Executions",
            summary.get("total_executions", 0),
        )
    with col3:
        success_rate = summary.get("success_rate", 0)
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta="Good"
            if success_rate >= 90
            else ("Warning" if success_rate >= 70 else "Critical"),
            delta_color="normal"
            if success_rate >= 90
            else ("off" if success_rate >= 70 else "inverse"),
        )
    with col4:
        avg_duration = summary.get("avg_duration_ms", 0)
        st.metric(
            "Avg Duration",
            f"{avg_duration:.0f}ms" if avg_duration < 1000 else f"{avg_duration / 1000:.1f}s",
        )

    st.divider()

    # Active workflows
    st.subheader("Active Workflows")
    active = data.get("active_workflows", [])

    if active:
        for wf in active:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{wf['workflow_name']}**")
                    st.caption(f"ID: {wf['execution_id']}")
                with col2:
                    progress = (
                        wf["completed_steps"] / wf["total_steps"] if wf["total_steps"] > 0 else 0
                    )
                    st.progress(progress, f"{wf['completed_steps']}/{wf['total_steps']} steps")
                with col3:
                    st.markdown(f"Status: **{wf['status']}**")
                st.divider()
    else:
        st.info("No active workflows")

    # Recent executions
    st.subheader("Recent Executions")
    recent = data.get("recent_executions", [])

    if recent:
        for execution in recent[:10]:
            with st.expander(
                f"{'✅' if execution['status'] == 'completed' else '❌'} "
                f"{execution['workflow_name']} - {execution['execution_id'][:16]}..."
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Status:** {execution['status']}")
                    st.write(f"**Duration:** {execution['duration_ms']:.0f}ms")
                    st.write(
                        f"**Steps:** {execution['completed_steps']}/{execution['total_steps']}"
                    )
                with col2:
                    st.write(f"**Started:** {execution['started_at']}")
                    st.write(f"**Completed:** {execution.get('completed_at', 'N/A')}")
                    st.write(f"**Tokens:** {execution.get('total_tokens', 0)}")

                if execution.get("error"):
                    st.error(f"Error: {execution['error']}")

                # Show steps
                if execution.get("steps"):
                    st.markdown("**Steps:**")
                    for step in execution["steps"]:
                        step_status = "✅" if step["status"] == "success" else "❌"
                        st.text(
                            f"  {step_status} {step['step_id']} ({step['step_type']}:{step['action']}) "
                            f"- {step['duration_ms']:.0f}ms"
                        )
    else:
        st.info("No recent executions")


def render_agents_page() -> None:
    """Render agent coordination visualization page."""
    st.title("Agent Coordination")

    agent_tracker = get_agent_tracker()

    # Summary
    summary = agent_tracker.get_agent_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Agents", summary["active_count"])
    with col2:
        st.metric("Recent Completions", summary["recent_count"])
    with col3:
        roles_str = ", ".join(f"{k}: {v}" for k, v in summary.get("by_role", {}).items())
        st.metric("By Role", roles_str or "None")

    st.divider()

    # Active agents visualization
    st.subheader("Active Agents")

    active_agents = agent_tracker.get_active_agents()

    if active_agents:
        # Create a visual grid for agents
        cols = st.columns(min(len(active_agents), 4))
        for i, agent in enumerate(active_agents):
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #00d4ff;
                        border-radius: 10px;
                        padding: 15px;
                        margin: 5px;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    ">
                        <h4 style="color: #00d4ff; margin: 0;">🤖 {agent["role"].title()}</h4>
                        <p style="color: #888; font-size: 12px; margin: 5px 0;">
                            ID: {agent["agent_id"][:12]}...
                        </p>
                        <p style="color: #4caf50; font-size: 14px;">
                            ● {agent["status"].upper()}
                        </p>
                        <p style="color: #888; font-size: 11px;">
                            Tasks: {agent.get("tasks_completed", 0)}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No active agents")

        # Demo button to simulate agents
        if st.button("Simulate Agent Activity"):
            import uuid

            for role in ["planner", "builder", "tester"]:
                agent_tracker.register_agent(
                    f"agent_{uuid.uuid4().hex[:8]}",
                    role,
                    "demo_workflow",
                )
            st.rerun()

    st.divider()

    # Agent coordination diagram
    st.subheader("Coordination Flow")

    st.markdown("""
    ```
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   PLANNER   │────▶│   BUILDER   │────▶│   TESTER    │
    │  (analyze)  │     │  (implement)│     │  (validate) │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Workflow   │     │   Context   │     │   Results   │
    │   Context   │     │   Updates   │     │   & Logs    │
    └─────────────┘     └─────────────┘     └─────────────┘
    ```
    """)

    st.divider()

    # Agent history
    st.subheader("Recent Agent Activity")

    history = agent_tracker.get_agent_history(20)

    if history:
        for agent in history:
            status_icon = "✅" if agent["status"] == "completed" else "❌"
            st.text(
                f"{status_icon} {agent['role'].title()} "
                f"({agent['agent_id'][:12]}...) - "
                f"{agent.get('completed_at', 'N/A')}"
            )
    else:
        st.info("No agent history")


def render_metrics_page() -> None:
    """Render metrics and analytics page."""
    st.title("Metrics Dashboard")

    try:
        tracker = get_tracker()
        data = tracker.get_dashboard_data()
    except Exception:
        data = {
            "summary": {},
            "step_performance": {},
            "recent_executions": [],
        }

    # Summary cards
    summary = data.get("summary", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Execution Metrics")

        st.metric("Total Workflows", summary.get("total_executions", 0))
        st.metric("Failed Workflows", summary.get("failed_executions", 0))
        st.metric("Total Steps", summary.get("total_steps_executed", 0))
        st.metric("Total Tokens", summary.get("total_tokens_used", 0))

    with col2:
        st.subheader("Performance")

        success_rate = summary.get("success_rate", 0)
        avg_duration = summary.get("avg_duration_ms", 0)

        # Success rate gauge
        st.markdown(f"### Success Rate: {success_rate:.1f}%")
        st.progress(success_rate / 100 if success_rate <= 100 else 1.0)

        st.markdown(f"### Avg Duration: {avg_duration:.0f}ms")

    st.divider()

    # Step performance breakdown
    st.subheader("Step Performance by Type")

    step_perf = data.get("step_performance", {})

    if step_perf:
        # Create a table
        rows = []
        for step_type, stats in step_perf.items():
            rows.append(
                {
                    "Step Type": step_type,
                    "Count": stats["count"],
                    "Avg Duration (ms)": f"{stats['avg_ms']:.1f}",
                    "Failure Rate": f"{stats['failure_rate']:.1f}%",
                }
            )

        st.table(rows)
    else:
        st.info("No step performance data available")

    st.divider()

    # Token usage over time (simulated for now)
    st.subheader("Token Usage Trend")

    recent = data.get("recent_executions", [])

    if recent:
        # Extract token data
        import pandas as pd

        token_data = [
            {
                "execution": ex["execution_id"][:8],
                "tokens": ex.get("total_tokens", 0),
            }
            for ex in recent[:20]
        ]

        if token_data and any(d["tokens"] > 0 for d in token_data):
            df = pd.DataFrame(token_data)
            st.bar_chart(df.set_index("execution")["tokens"])
        else:
            st.info("No token usage data recorded")
    else:
        st.info("Execute workflows to see token usage trends")

    st.divider()

    # Duration histogram
    st.subheader("Execution Duration Distribution")

    if recent:
        durations = [ex["duration_ms"] for ex in recent if ex.get("duration_ms")]
        if durations:
            import pandas as pd

            df = pd.DataFrame({"duration_ms": durations})
            st.bar_chart(df["duration_ms"])
        else:
            st.info("No duration data available")
    else:
        st.info("Execute workflows to see duration distribution")


def render_system_status() -> None:
    """Render system status component (for sidebar)."""
    st.sidebar.divider()
    st.sidebar.subheader("System Status")

    try:
        tracker = get_tracker()
        summary = tracker.store.get_summary()

        active = summary.get("active_workflows", 0)
        status_color = "🟢" if active == 0 else "🟡"

        st.sidebar.markdown(f"{status_color} **{active}** active workflows")
        st.sidebar.caption(f"Success rate: {summary.get('success_rate', 0):.0f}%")

    except Exception:
        st.sidebar.markdown("⚪ Monitoring unavailable")


def render_analytics_page() -> None:
    """Render analytics pipeline page."""
    st.title("Analytics Pipelines")

    # Pipeline descriptions
    pipelines = {
        "workflow_metrics": {
            "name": "Workflow Metrics",
            "description": "Real-time workflow execution metrics from ExecutionTracker",
            "icon": "⚡",
        },
        "historical_trends": {
            "name": "Historical Trends",
            "description": "Analyze execution trends over time from MetricsStore",
            "icon": "📊",
        },
        "api_health": {
            "name": "API Health",
            "description": "Monitor API client resilience (rate limits, circuit breakers)",
            "icon": "🔌",
        },
        "operations_dashboard": {
            "name": "Operations Dashboard",
            "description": "Comprehensive view combining all metrics sources",
            "icon": "🎛️",
        },
    }

    # Pipeline selector
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_pipeline = st.selectbox(
            "Select Pipeline",
            options=list(pipelines.keys()),
            format_func=lambda x: f"{pipelines[x]['icon']} {pipelines[x]['name']}",
        )
    with col2:
        hours = st.number_input(
            "History (hours)",
            min_value=1,
            max_value=168,
            value=24,
            help="Hours of history for historical_trends pipeline",
        )

    st.info(pipelines[selected_pipeline]["description"])

    # Run pipeline button
    if st.button("Run Pipeline", type="primary", use_container_width=True):
        with st.spinner(f"Running {pipelines[selected_pipeline]['name']} pipeline..."):
            try:
                result = _run_analytics_pipeline(selected_pipeline, hours)
                st.session_state.analytics_result = result
                st.session_state.analytics_pipeline = selected_pipeline
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback

                st.code(traceback.format_exc())

    st.divider()

    # Display results
    if "analytics_result" in st.session_state:
        result = st.session_state.analytics_result
        pipeline_name = st.session_state.get("analytics_pipeline", "unknown")

        _render_pipeline_result(result, pipeline_name, pipelines)


def _run_analytics_pipeline(pipeline_id: str, hours: int = 24) -> PipelineResult:
    """Run selected analytics pipeline."""
    from animus_forge.analytics.pipeline import PipelineBuilder

    if pipeline_id == "workflow_metrics":
        pipeline = PipelineBuilder.workflow_metrics_pipeline()
    elif pipeline_id == "historical_trends":
        pipeline = PipelineBuilder.historical_trends_pipeline(hours=hours)
    elif pipeline_id == "api_health":
        pipeline = PipelineBuilder.api_health_pipeline()
    elif pipeline_id == "operations_dashboard":
        pipeline = PipelineBuilder.operations_dashboard_pipeline()
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_id}")

    return pipeline.execute()


def _render_pipeline_result(result, pipeline_name: str, pipelines: dict) -> None:
    """Render pipeline execution results."""
    # Header with status
    status_icon = "✅" if result.status == "completed" else "❌"
    st.subheader(f"{status_icon} {pipelines.get(pipeline_name, {}).get('name', pipeline_name)}")

    # Pipeline execution summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", result.status.title())
    with col2:
        st.metric("Stages", len(result.stages))
    with col3:
        total_duration = sum(s.duration_ms for s in result.stages)
        st.metric("Duration", f"{total_duration:.1f}ms")
    with col4:
        errors = len(result.errors)
        st.metric(
            "Errors",
            errors,
            delta="OK" if errors == 0 else None,
            delta_color="normal" if errors == 0 else "inverse",
        )

    st.divider()

    # Stage details
    st.subheader("Pipeline Stages")
    for stage in result.stages:
        status_emoji = "✅" if stage.status == "success" else "❌"
        with st.expander(f"{status_emoji} {stage.stage.value.upper()} - {stage.duration_ms:.1f}ms"):
            if stage.error:
                st.error(f"Error: {stage.error}")
            elif stage.output:
                _render_stage_output(stage)

    # Final output visualization
    if result.final_output:
        st.divider()
        st.subheader("Analysis Results")
        _render_final_output(result.final_output)


def _render_collect_output(output) -> None:
    """Render collect stage output."""
    if hasattr(output, "source"):
        st.markdown(f"**Source:** {output.source}")
        st.caption(f"Collected at: {output.collected_at}")
        if hasattr(output, "data") and output.data:
            data = output.data
            counters = data.get("metrics", {}).get("counters", {})
            if counters:
                st.markdown("**Counters:**")
                cols = st.columns(min(4, len(counters)))
                for i, (key, value) in enumerate(list(counters.items())[:8]):
                    cols[i % 4].metric(key.replace("_", " ").title(), value)
            summary = data.get("summary", {})
            if summary:
                st.markdown("**Summary:**")
                st.json(summary)
    else:
        st.json(output if isinstance(output, dict) else str(output))


def _render_analyze_output(output) -> None:
    """Render analyze stage output."""
    if hasattr(output, "severity"):
        severity_colors = {"info": "blue", "warning": "orange", "critical": "red"}
        color = severity_colors.get(output.severity, "gray")
        st.markdown(f"**Severity:** :{color}[{output.severity.upper()}]")
    if hasattr(output, "findings") and output.findings:
        st.markdown("**Findings:**")
        for finding in output.findings:
            sev = finding.get("severity", "info")
            icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(sev, "•")
            st.markdown(f"{icon} {finding.get('message', '')}")
    else:
        st.success("No issues detected")


def _render_visualize_output(output) -> None:
    """Render visualize stage output."""
    if hasattr(output, "charts"):
        st.markdown(f"**Generated {len(output.charts)} charts**")
        for chart in output.charts:
            st.caption(f"• {chart.title} ({chart.chart_type})")
    if hasattr(output, "streamlit_code") and output.streamlit_code:
        with st.expander("View Generated Code"):
            st.code(output.streamlit_code, language="python")


def _render_report_output(output) -> None:
    """Render report stage output."""
    if hasattr(output, "report_type"):
        st.markdown(f"**Report Type:** {output.report_type}")
    if hasattr(output, "summary"):
        st.markdown(output.summary)


def _render_alert_output(output) -> None:
    """Render alert stage output."""
    if hasattr(output, "alerts") and output.alerts:
        st.markdown(f"**{len(output.alerts)} alerts generated**")
        for alert in output.alerts:
            sev = alert.get("severity", "info")
            icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(sev, "•")
            st.markdown(f"{icon} **{alert.get('title', 'Alert')}**: {alert.get('message', '')}")
    else:
        st.success("No alerts triggered")


def _render_fallback_output(output) -> None:
    """Render fallback for unknown stage types."""
    if isinstance(output, dict):
        st.json(output)
    elif hasattr(output, "to_dict"):
        st.json(output.to_dict())
    else:
        st.text(str(output))


_STAGE_RENDERERS = {
    "collect": _render_collect_output,
    "analyze": _render_analyze_output,
    "visualize": _render_visualize_output,
    "report": _render_report_output,
    "alert": _render_alert_output,
}


def _render_stage_output(stage) -> None:
    """Render individual stage output."""
    output = stage.output
    stage_type = stage.stage.value
    renderer = _STAGE_RENDERERS.get(stage_type, _render_fallback_output)
    renderer(output)


def _render_final_output(output) -> None:
    """Render final pipeline output with charts."""
    import pandas as pd

    # Try to extract metrics for visualization
    metrics = {}
    findings = []

    if hasattr(output, "metrics"):
        metrics = output.metrics
    elif hasattr(output, "data") and isinstance(output.data, dict):
        metrics = output.data.get("metrics", {})
    elif isinstance(output, dict):
        metrics = output.get("metrics", output)

    if hasattr(output, "findings"):
        findings = output.findings

    # Counters as bar chart
    counters = metrics.get("counters", {})
    if counters:
        numeric_counters = {
            k: v for k, v in counters.items() if isinstance(v, (int, float)) and v > 0
        }
        if numeric_counters:
            st.markdown("### Key Metrics")
            df = pd.DataFrame(
                {
                    "Metric": list(numeric_counters.keys()),
                    "Value": list(numeric_counters.values()),
                }
            )
            st.bar_chart(df.set_index("Metric"))

    # Findings table
    if findings:
        st.markdown("### Analysis Findings")
        findings_df = pd.DataFrame(
            [
                {"Severity": f.get("severity", "info"), "Finding": f.get("message", "")}
                for f in findings
            ]
        )
        st.dataframe(findings_df, use_container_width=True)

    # Raw output expander
    with st.expander("View Raw Output"):
        if hasattr(output, "to_dict"):
            st.json(output.to_dict())
        elif isinstance(output, dict):
            st.json(output)
        else:
            st.text(str(output))


# =============================================================================
# Parallel Execution Monitoring Page
# =============================================================================


def render_parallel_execution_page() -> None:
    """Render parallel execution monitoring page.

    Shows real-time status of fan-out, map-reduce, and auto-parallel executions,
    including rate limit status and branch progress.
    """
    st.title("Parallel Execution Monitor")

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        auto_refresh = st.toggle("Auto-refresh", value=True, key="parallel_auto_refresh")

    if auto_refresh:
        time.sleep(0.1)
        st.rerun()

    try:
        tracker = get_parallel_tracker()
        data = tracker.get_dashboard_data()
    except Exception as e:
        st.warning(f"Parallel tracking data not available: {e}")
        data = {
            "summary": {
                "active_executions": 0,
                "total_executions": 0,
                "success_rate": 0,
                "counters": {},
            },
            "active_executions": [],
            "recent_executions": [],
            "rate_limits": {},
        }

    summary = data["summary"]

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Active Executions",
            summary.get("active_executions", 0),
            delta=f"{summary.get('active_branches', 0)} branches",
        )
    with col2:
        st.metric(
            "Total Executions",
            summary.get("total_executions", 0),
        )
    with col3:
        success_rate = summary.get("success_rate", 0)
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta="Good"
            if success_rate >= 90
            else ("Warning" if success_rate >= 70 else "Critical"),
            delta_color="normal"
            if success_rate >= 90
            else ("off" if success_rate >= 70 else "inverse"),
        )
    with col4:
        counters = summary.get("counters", {})
        total_branches = counters.get("branches_completed", 0) + counters.get("branches_failed", 0)
        st.metric(
            "Total Branches",
            total_branches,
            delta=f"{counters.get('branches_failed', 0)} failed"
            if counters.get("branches_failed", 0) > 0
            else None,
            delta_color="inverse" if counters.get("branches_failed", 0) > 0 else "normal",
        )

    st.divider()

    # Rate Limit Status Section
    _render_rate_limit_section(data.get("rate_limits", {}), summary)

    st.divider()

    # Active Parallel Executions
    st.subheader("Active Parallel Executions")
    active = data.get("active_executions", [])

    if active:
        for execution in active:
            _render_active_execution(execution)
    else:
        st.info("No active parallel executions")

    st.divider()

    # Recent Executions History
    st.subheader("Recent Executions")
    recent = data.get("recent_executions", [])

    if recent:
        for execution in recent[:10]:
            _render_recent_execution(execution)
    else:
        st.info("No recent parallel executions")

    # Performance Stats
    st.divider()
    st.subheader("Performance Statistics")
    _render_performance_stats(summary)


def _render_rate_limit_section(rate_limits: dict, summary: dict) -> None:
    """Render rate limit status gauges."""
    st.subheader("Rate Limit Status")

    # Check if any rate limiting is happening
    rate_limit_waits = summary.get("rate_limit_waits", {})
    rate_limit_states = summary.get("rate_limit_states", {})

    if not rate_limits and not rate_limit_states:
        st.success("No rate limiting active - all providers operating normally")
        return

    # Combine rate_limits dict and rate_limit_states
    all_providers = set(rate_limits.keys()) | set(rate_limit_states.keys())

    if not all_providers:
        st.success("No rate limiting active")
        return

    cols = st.columns(min(len(all_providers), 3))

    for idx, provider in enumerate(sorted(all_providers)):
        with cols[idx % 3]:
            state = rate_limits.get(provider, rate_limit_states.get(provider, {}))

            base_limit = state.get("base_limit", 10)
            current_limit = state.get("current_limit", base_limit)
            total_429s = state.get("total_429s", 0)
            is_throttled = state.get("is_throttled", current_limit < base_limit)

            # Determine status color
            if is_throttled:
                status_color = "#ff6b6b"  # Red
                status_text = "THROTTLED"
            elif total_429s > 0:
                status_color = "#ffd93d"  # Yellow
                status_text = "WARNING"
            else:
                status_color = "#6bcf6b"  # Green
                status_text = "OK"

            # Calculate utilization
            utilization = (base_limit - current_limit) / base_limit * 100 if base_limit > 0 else 0

            st.markdown(
                f"""
                <div style="
                    border: 2px solid {status_color};
                    border-radius: 10px;
                    padding: 15px;
                    margin: 5px 0;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                ">
                    <h4 style="color: {status_color}; margin: 0;">
                        {provider.upper()}
                    </h4>
                    <p style="color: {status_color}; font-size: 12px; margin: 5px 0;">
                        {status_text}
                    </p>
                    <p style="color: #ccc; font-size: 14px; margin: 5px 0;">
                        Limit: {current_limit}/{base_limit}
                    </p>
                    <p style="color: #888; font-size: 12px; margin: 0;">
                        429 errors: {total_429s}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Progress bar showing how throttled the provider is
            if is_throttled:
                st.progress(utilization / 100, f"Throttled {utilization:.0f}%")

    # Rate limit wait statistics
    if rate_limit_waits.get("count", 0) > 0:
        st.markdown("**Rate Limit Wait Times:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Waits", rate_limit_waits["count"])
        with col2:
            st.metric("Avg Wait", f"{rate_limit_waits['avg']:.1f}ms")
        with col3:
            st.metric("Max Wait", f"{rate_limit_waits['max']:.1f}ms")


def _render_active_execution(execution: dict) -> None:
    """Render a single active parallel execution."""
    pattern_type = execution.get("pattern_type", "unknown")
    step_id = execution.get("step_id", "unknown")
    execution_id = execution.get("execution_id", "")[:16]

    # Pattern type icon
    pattern_icons = {
        "fan_out": "🔀",
        "fan_in": "🔄",
        "map_reduce": "🗺️",
        "auto_parallel": "⚡",
        "parallel_group": "📦",
    }
    icon = pattern_icons.get(pattern_type, "⏳")

    with st.container():
        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            st.markdown(f"### {icon} {pattern_type}")
            st.caption(f"Step: {step_id}")
            st.caption(f"ID: {execution_id}...")

        with col2:
            # Progress calculation
            total = execution.get("total_items", 0)
            completed = execution.get("completed_count", 0)
            failed = execution.get("failed_count", 0)
            active = execution.get("active_branch_count", 0)

            progress = completed / total if total > 0 else 0

            st.progress(progress, f"{completed}/{total} completed")

            # Branch status
            st.markdown(f"**Active:** {active} | **Completed:** {completed} | **Failed:** {failed}")

        with col3:
            # Timing info
            duration = execution.get("duration_ms", 0)
            if duration > 0:
                st.metric(
                    "Duration",
                    f"{duration:.0f}ms" if duration < 1000 else f"{duration / 1000:.1f}s",
                )

            tokens = execution.get("total_tokens", 0)
            if tokens > 0:
                st.metric("Tokens", tokens)

        # Branch details expander
        branches = execution.get("branches", [])
        if branches:
            with st.expander(f"View {len(branches)} branches"):
                for branch in branches[:20]:  # Limit to 20
                    branch_status = branch.get("status", "unknown")
                    status_icon = {
                        "pending": "⏳",
                        "running": "🏃",
                        "completed": "✅",
                        "failed": "❌",
                        "cancelled": "🚫",
                    }.get(branch_status, "❓")

                    st.text(
                        f"  {status_icon} {branch.get('branch_id', 'unknown')[:20]} "
                        f"[{branch.get('item_index', 0)}] - "
                        f"{branch.get('duration_ms', 0):.0f}ms"
                    )

        st.divider()


def _render_recent_execution(execution: dict) -> None:
    """Render a recent parallel execution in history."""
    pattern_type = execution.get("pattern_type", "unknown")
    step_id = execution.get("step_id", "unknown")
    status = execution.get("status", "unknown")
    execution_id = execution.get("execution_id", "")[:16]

    # Status icon
    status_icon = "✅" if status == "completed" else "❌"

    # Pattern type icon
    pattern_icons = {
        "fan_out": "🔀",
        "fan_in": "🔄",
        "map_reduce": "🗺️",
        "auto_parallel": "⚡",
        "parallel_group": "📦",
    }
    pattern_icon = pattern_icons.get(pattern_type, "⏳")

    with st.expander(f"{status_icon} {pattern_icon} {step_id} - {execution_id}..."):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Pattern:** {pattern_type}")
            st.write(f"**Status:** {status}")
            st.write(f"**Total Items:** {execution.get('total_items', 0)}")
            st.write(f"**Completed:** {execution.get('completed_count', 0)}")
            st.write(f"**Failed:** {execution.get('failed_count', 0)}")

        with col2:
            st.write(f"**Duration:** {execution.get('duration_ms', 0):.0f}ms")
            st.write(f"**Tokens:** {execution.get('total_tokens', 0)}")
            st.write(f"**Started:** {execution.get('started_at', 'N/A')}")
            st.write(f"**Completed:** {execution.get('completed_at', 'N/A')}")

        # Show error if failed
        if status == "failed":
            branches = execution.get("branches", [])
            failed_branches = [b for b in branches if b.get("status") == "failed"]
            if failed_branches:
                st.error("Failed Branches:")
                for fb in failed_branches[:5]:
                    st.text(
                        f"  {fb.get('branch_id', 'unknown')}: {fb.get('error', 'Unknown error')}"
                    )


def _render_performance_stats(summary: dict) -> None:
    """Render performance statistics section."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Execution Duration (ms)**")
        duration_stats = summary.get("execution_duration", {})
        if duration_stats.get("count", 0) > 0:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Avg", f"{duration_stats.get('avg', 0):.1f}")
            with metrics_col2:
                st.metric("P50", f"{duration_stats.get('p50', 0):.1f}")
            with metrics_col3:
                st.metric("P95", f"{duration_stats.get('p95', 0):.1f}")
        else:
            st.info("No execution data yet")

    with col2:
        st.markdown("**Branch Duration (ms)**")
        branch_stats = summary.get("branch_duration", {})
        if branch_stats.get("count", 0) > 0:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Avg", f"{branch_stats.get('avg', 0):.1f}")
            with metrics_col2:
                st.metric("P50", f"{branch_stats.get('p50', 0):.1f}")
            with metrics_col3:
                st.metric("P95", f"{branch_stats.get('p95', 0):.1f}")
        else:
            st.info("No branch data yet")

    # Token usage stats
    token_stats = summary.get("execution_tokens", {})
    if token_stats.get("count", 0) > 0:
        st.markdown("**Token Usage per Execution**")
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        with tcol1:
            st.metric("Total", token_stats.get("count", 0))
        with tcol2:
            st.metric("Avg", f"{token_stats.get('avg', 0):.0f}")
        with tcol3:
            st.metric("Min", f"{token_stats.get('min', 0):.0f}")
        with tcol4:
            st.metric("Max", f"{token_stats.get('max', 0):.0f}")

    # Pattern type breakdown
    counters = summary.get("counters", {})
    pattern_counts = {
        k.replace("executions_started_", ""): v
        for k, v in counters.items()
        if k.startswith("executions_started_") and k != "executions_started"
    }

    if pattern_counts:
        st.markdown("**Executions by Pattern Type**")
        import pandas as pd

        df = pd.DataFrame(
            {
                "Pattern": list(pattern_counts.keys()),
                "Count": list(pattern_counts.values()),
            }
        )
        st.bar_chart(df.set_index("Pattern"))


def render_parallel_status_sidebar() -> None:
    """Render parallel execution status in sidebar."""
    st.sidebar.divider()
    st.sidebar.subheader("Parallel Status")

    try:
        tracker = get_parallel_tracker()
        summary = tracker.get_summary()

        active = summary.get("active_executions", 0)
        active_branches = summary.get("active_branches", 0)

        if active > 0:
            st.sidebar.markdown(f"🟡 **{active}** active executions")
            st.sidebar.caption(f"{active_branches} branches running")
        else:
            st.sidebar.markdown("🟢 No active parallel executions")

        # Rate limit status
        rate_limits = summary.get("rate_limit_states", {})
        throttled = [p for p, s in rate_limits.items() if s.get("is_throttled")]

        if throttled:
            st.sidebar.warning(f"⚠️ Throttled: {', '.join(throttled)}")

    except Exception:
        st.sidebar.markdown("⚪ Parallel monitoring unavailable")
