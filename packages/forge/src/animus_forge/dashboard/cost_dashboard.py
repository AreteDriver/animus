"""Cost dashboard page for Streamlit."""

from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

try:
    from animus_forge.metrics.cost_tracker import CostTracker, get_cost_tracker
except ImportError:
    CostTracker = None
    get_cost_tracker = None


def render_cost_dashboard() -> None:
    """Render the cost tracking dashboard page."""
    st.title("💰 Cost Dashboard")

    if CostTracker is None:
        st.error("Cost tracker module not available")
        return

    # Initialize or get tracker
    tracker = _get_tracker()

    # Top metrics
    _render_top_metrics(tracker)

    st.divider()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Overview",
            "🤖 By Agent",
            "🧠 By Model",
            "📋 Details",
        ]
    )

    with tab1:
        _render_overview_tab(tracker)

    with tab2:
        _render_agent_tab(tracker)

    with tab3:
        _render_model_tab(tracker)

    with tab4:
        _render_details_tab(tracker)


@st.cache_resource
def _get_tracker() -> CostTracker:
    """Get or create cost tracker instance."""
    if get_cost_tracker:
        return get_cost_tracker()

    # Fallback: create new tracker
    storage_path = Path("data/costs.json")
    return CostTracker(storage_path=storage_path)


def _render_top_metrics(tracker: CostTracker) -> None:
    """Render top-level cost metrics."""
    summary = tracker.get_summary(days=30)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Monthly Cost",
            f"${summary['total_cost_usd']:.2f}",
            delta=None,
        )

    with col2:
        st.metric(
            "Total Tokens",
            f"{summary['total_tokens']:,}",
        )

    with col3:
        st.metric(
            "API Calls",
            f"{summary['total_calls']:,}",
        )

    with col4:
        avg_cost = summary.get("avg_cost_per_call", 0)
        st.metric(
            "Avg Cost/Call",
            f"${avg_cost:.4f}",
        )

    # Budget progress
    budget = summary.get("budget", {})
    if budget.get("limit_usd"):
        st.markdown("### 📊 Budget Status")
        percent_used = budget.get("percent_used", 0) or 0
        st.progress(
            min(percent_used / 100, 1.0),
            text=f"${budget.get('monthly_used_usd', 0):.2f} / ${budget['limit_usd']:.2f} ({percent_used:.1f}%)",
        )

        if percent_used >= 90:
            st.error("⚠️ Budget limit nearly reached!")
        elif percent_used >= 75:
            st.warning("⚠️ Budget at 75%")


def _render_overview_tab(tracker: CostTracker) -> None:
    """Render overview tab with charts."""
    st.subheader("📈 Cost Overview (Last 30 Days)")

    summary = tracker.get_summary(days=30)

    # Cost by provider pie chart
    by_provider = summary.get("by_provider", {})
    if by_provider:
        st.markdown("#### By Provider")

        col1, col2 = st.columns(2)

        with col1:
            # Create simple bar representation
            total = sum(p.get("cost", 0) for p in by_provider.values())
            for provider, data in by_provider.items():
                cost = data.get("cost", 0)
                pct = (cost / total * 100) if total > 0 else 0
                st.markdown(f"**{provider.upper()}**: ${cost:.4f} ({pct:.1f}%)")
                st.progress(pct / 100 if total > 0 else 0)

        with col2:
            st.markdown("**Token Usage by Provider:**")
            for provider, data in by_provider.items():
                tokens = data.get("tokens", 0)
                calls = data.get("calls", 0)
                st.write(f"- {provider}: {tokens:,} tokens ({calls} calls)")

    # Daily cost trend (simulated with available data)
    st.markdown("#### Daily Trend")
    _render_daily_trend(tracker)


def _render_daily_trend(tracker: CostTracker) -> None:
    """Render daily cost trend."""
    # Get costs for last 7 days
    daily_costs = []
    for i in range(7):
        date = datetime.now() - timedelta(days=i)
        cost = tracker.get_daily_cost(date)
        daily_costs.append(
            {
                "date": date.strftime("%m/%d"),
                "cost": cost,
            }
        )

    daily_costs.reverse()

    if any(d["cost"] > 0 for d in daily_costs):
        # Simple bar chart representation
        max_cost = max(d["cost"] for d in daily_costs) or 1

        cols = st.columns(7)
        for i, day in enumerate(daily_costs):
            with cols[i]:
                height = int((day["cost"] / max_cost) * 100) if max_cost > 0 else 0
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="
                            width: 30px;
                            height: {max(height, 5)}px;
                            background: linear-gradient(to top, #4CAF50, #8BC34A);
                            margin: 0 auto {100 - height}px auto;
                            border-radius: 4px 4px 0 0;
                        "></div>
                        <div style="font-size: 10px;">{day["date"]}</div>
                        <div style="font-size: 10px;">${day["cost"]:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No cost data available for the last 7 days")


def _render_agent_tab(tracker: CostTracker) -> None:
    """Render agent cost breakdown tab."""
    st.subheader("🤖 Cost by Agent Role")

    days = st.selectbox(
        "Time Period", [7, 14, 30, 90], index=2, format_func=lambda x: f"Last {x} days"
    )

    agent_costs = tracker.get_agent_costs(days=days)

    if agent_costs:
        # Sort by cost descending
        sorted_agents = sorted(agent_costs.items(), key=lambda x: x[1].get("cost", 0), reverse=True)

        total_cost = sum(data.get("cost", 0) for _, data in sorted_agents)

        for agent, data in sorted_agents:
            cost = data.get("cost", 0)
            tokens = data.get("tokens", 0)
            calls = data.get("calls", 0)
            pct = (cost / total_cost * 100) if total_cost > 0 else 0

            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    icon = _get_agent_icon(agent)
                    st.markdown(f"{icon} **{agent}**")
                    st.progress(pct / 100 if total_cost > 0 else 0)

                with col2:
                    st.metric("Cost", f"${cost:.4f}")

                with col3:
                    st.metric("Tokens", f"{tokens:,}")

                with col4:
                    st.metric("Calls", str(calls))

                st.divider()
    else:
        st.info("No agent cost data available")


def _render_model_tab(tracker: CostTracker) -> None:
    """Render model cost breakdown tab."""
    st.subheader("🧠 Cost by Model")

    days = st.selectbox(
        "Time Period ",
        [7, 14, 30, 90],
        index=2,
        format_func=lambda x: f"Last {x} days",
        key="model_days",
    )

    model_costs = tracker.get_model_costs(days=days)

    if model_costs:
        # Sort by cost descending
        sorted_models = sorted(model_costs.items(), key=lambda x: x[1].get("cost", 0), reverse=True)

        total_cost = sum(data.get("cost", 0) for _, data in sorted_models)

        for model, data in sorted_models:
            cost = data.get("cost", 0)
            tokens = data.get("tokens", 0)
            calls = data.get("calls", 0)
            pct = (cost / total_cost * 100) if total_cost > 0 else 0

            # Get pricing info
            pricing = CostTracker.PRICING.get(model, {})
            input_price = pricing.get("input", "N/A")
            output_price = pricing.get("output", "N/A")

            with st.expander(f"**{model}** - ${cost:.4f} ({pct:.1f}%)"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Total Cost:** ${cost:.4f}")
                    st.write(f"**Total Tokens:** {tokens:,}")
                    st.write(f"**API Calls:** {calls}")

                with col2:
                    st.write(f"**Input Price:** ${input_price}/1M tokens")
                    st.write(f"**Output Price:** ${output_price}/1M tokens")
                    avg = cost / calls if calls > 0 else 0
                    st.write(f"**Avg Cost/Call:** ${avg:.6f}")
    else:
        st.info("No model cost data available")


def _render_details_tab(tracker: CostTracker) -> None:
    """Render detailed cost entries tab."""
    st.subheader("📋 Recent API Calls")

    # Get recent entries
    entries = tracker.entries[-50:]  # Last 50
    entries.reverse()

    if entries:
        # Export button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📥 Export CSV"):
                export_path = Path("data/cost_export.csv")
                tracker.export_csv(export_path, days=30)
                st.success(f"Exported to {export_path}")

        # Entries table
        for entry in entries:
            with st.expander(
                f"{entry.timestamp.strftime('%Y-%m-%d %H:%M')} - {entry.model} - ${entry.cost_usd:.6f}"
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Provider:** {entry.provider.value}")
                    st.write(f"**Model:** {entry.model}")
                    st.write(f"**Cost:** ${entry.cost_usd:.6f}")

                with col2:
                    st.write(f"**Input Tokens:** {entry.tokens.input_tokens:,}")
                    st.write(f"**Output Tokens:** {entry.tokens.output_tokens:,}")
                    st.write(f"**Total Tokens:** {entry.tokens.total_tokens:,}")

                if entry.workflow_id:
                    st.write(f"**Workflow:** {entry.workflow_id}")
                if entry.agent_role:
                    st.write(f"**Agent:** {entry.agent_role}")
    else:
        st.info("No cost entries recorded yet")

    # Clear old entries
    st.divider()
    st.markdown("### 🗑️ Maintenance")

    days_to_keep = st.number_input("Days to keep", min_value=7, max_value=365, value=90)
    if st.button("Clear Old Entries"):
        removed = tracker.clear_old_entries(days=days_to_keep)
        st.success(f"Removed {removed} old entries")


def _get_agent_icon(agent: str) -> str:
    """Get icon for agent role."""
    icons = {
        "planner": "📋",
        "builder": "🔨",
        "tester": "🧪",
        "reviewer": "👁️",
        "model_builder": "🎮",
        "data_analyst": "📊",
        "devops": "🔧",
        "security_auditor": "🔒",
        "migrator": "🔄",
    }
    return icons.get(agent, "🤖")


def render_cost_widget() -> None:
    """Render a small cost widget for sidebar."""
    if CostTracker is None:
        return

    tracker = _get_tracker()
    daily_cost = tracker.get_daily_cost()
    monthly_cost = tracker.get_monthly_cost()

    st.sidebar.markdown("### 💰 Costs")
    st.sidebar.write(f"Today: ${daily_cost:.2f}")
    st.sidebar.write(f"Month: ${monthly_cost:.2f}")
