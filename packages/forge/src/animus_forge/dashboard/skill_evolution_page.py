"""Skill evolution dashboard — metrics, A/B tests, versions, deprecations."""

from __future__ import annotations

import streamlit as st


def render_skill_evolution_page() -> None:
    """Main entry point for the skill evolution dashboard page."""
    st.title("Skills Evolution")

    try:
        from animus_forge.skills.evolver.ab_test import ABTestManager
        from animus_forge.skills.evolver.metrics import SkillMetricsAggregator
        from animus_forge.state.database import get_database

        db = get_database()
        aggregator = SkillMetricsAggregator(db)
        ab_manager = ABTestManager(db, aggregator)
    except Exception as e:
        st.error(f"Failed to connect to skill evolution store: {e}")
        return

    tab_metrics, tab_experiments, tab_versions, tab_deprecations = st.tabs(
        ["Metrics", "A/B Tests", "Versions", "Deprecations"]
    )

    with tab_metrics:
        _render_metrics_tab(aggregator)

    with tab_experiments:
        _render_experiments_tab(ab_manager, aggregator)

    with tab_versions:
        _render_versions_tab(db)

    with tab_deprecations:
        _render_deprecations_tab(db)


def _render_metrics_tab(aggregator: object) -> None:
    """Render skill performance metrics overview."""
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        days = st.slider("Look-back (days)", 7, 90, 30, key="metrics_days")
    with col_filter2:
        skill_filter = st.text_input("Skill filter", placeholder="e.g. code_review")

    metrics = aggregator.get_all_skill_metrics(days=days)

    if not metrics:
        st.info("No skill metrics recorded yet.")
        return

    if skill_filter:
        metrics = [m for m in metrics if skill_filter.lower() in m.skill_name.lower()]

    # Summary row
    total_invocations = sum(m.total_invocations for m in metrics)
    avg_success = (
        sum(m.success_rate * m.total_invocations for m in metrics) / total_invocations
        if total_invocations
        else 0
    )
    total_cost = sum(m.total_cost_usd for m in metrics)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Skills Tracked", len(metrics))
    with c2:
        st.metric("Total Invocations", f"{total_invocations:,}")
    with c3:
        st.metric("Avg Success Rate", f"{avg_success:.0%}")
    with c4:
        st.metric("Total Cost", f"${total_cost:.2f}")

    st.divider()

    # Per-skill table
    st.subheader("Per-Skill Breakdown")
    rows = []
    for m in metrics:
        trend_icon = {"improving": "^", "declining": "v", "stable": "-"}.get(m.trend, "-")
        rows.append(
            {
                "Skill": m.skill_name,
                "Version": m.skill_version or "all",
                "Invocations": m.total_invocations,
                "Success": f"{m.success_rate:.0%}",
                "Quality": f"{m.avg_quality_score:.2f}",
                "Avg Cost": f"${m.avg_cost_usd:.4f}",
                "Avg Latency": f"{m.avg_latency_ms:.0f}ms",
                "Trend": trend_icon,
            }
        )
    st.dataframe(rows, use_container_width=True)

    # Drill-down
    if metrics:
        st.divider()
        skill_names = sorted({m.skill_name for m in metrics})
        selected = st.selectbox("Drill into skill", skill_names)
        if selected:
            detail = aggregator.get_skill_metrics(selected, days=days)
            if detail:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(
                        "Success Rate",
                        f"{detail.success_rate:.1%}",
                    )
                with c2:
                    st.metric(
                        "Total Cost",
                        f"${detail.total_cost_usd:.4f}",
                    )
                with c3:
                    trend = aggregator.get_skill_trend(selected, days=days)
                    st.metric("Trend", trend)


def _render_experiments_tab(ab_manager: object, aggregator: object) -> None:
    """Render A/B test experiments."""
    active = ab_manager.get_active_experiments()

    if not active:
        st.info("No active A/B experiments.")
    else:
        st.subheader(f"Active Experiments ({len(active)})")
        for exp in active:
            eid = str(exp.get("id", ""))[:8]
            skill = exp.get("skill_name", "?")
            control = exp.get("control_version", "?")
            variant = exp.get("variant_version", "?")
            split = float(exp.get("traffic_split", 0.5))

            with st.expander(f"{skill}: v{control} vs v{variant} ({eid})"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**Control:** v{control}")
                with c2:
                    st.write(f"**Variant:** v{variant}")
                with c3:
                    st.write(f"**Split:** {split:.0%} variant")

                st.write(f"**Min invocations:** {exp.get('min_invocations', 100)}")
                st.write(f"**Started:** {str(exp.get('start_date', ''))[:16]}")

                result = ab_manager.evaluate_experiment(eid)
                if result:
                    st.success(
                        f"Ready to conclude: **{result.winner}** wins ({result.conclusion_reason})"
                    )
                else:
                    st.warning("Insufficient data to conclude yet.")

    # Concluded experiments
    st.divider()
    st.subheader("Experiment History")
    try:
        from animus_forge.state.database import get_database

        db = get_database()
        rows = db.fetchall(
            "SELECT * FROM skill_experiments "
            "WHERE status != 'active' "
            "ORDER BY concluded_at DESC LIMIT 20",
            (),
        )
        if rows:
            for r in rows:
                status = r.get("status", "?")
                color = "green" if status == "concluded" else "orange"
                winner = r.get("winner") or "N/A"
                st.write(
                    f":{color}[{status}] **{r.get('skill_name')}** "
                    f"v{r.get('control_version')} vs v{r.get('variant_version')} "
                    f"— Winner: {winner}"
                )
        else:
            st.info("No concluded experiments yet.")
    except Exception:
        st.info("No experiment history available.")


def _render_versions_tab(db: object) -> None:
    """Render skill version history."""
    st.subheader("Version History")

    try:
        rows = db.fetchall(
            "SELECT skill_name, version, previous_version, change_type, "
            "change_description, created_at "
            "FROM skill_versions ORDER BY created_at DESC LIMIT 50",
            (),
        )
    except Exception:
        rows = []

    if not rows:
        st.info("No skill versions recorded yet.")
        return

    skill_names = sorted({r["skill_name"] for r in rows})
    selected = st.selectbox("Filter by skill", ["All"] + skill_names)

    filtered = rows
    if selected != "All":
        filtered = [r for r in rows if r["skill_name"] == selected]

    for r in filtered:
        change_icon = {
            "tune": "wrench",
            "generate": "sparkles",
            "deprecate": "warning",
            "manual": "pencil",
        }.get(r.get("change_type", ""), "pencil")

        with st.expander(
            f":{change_icon}: **{r['skill_name']}** "
            f"v{r.get('previous_version', '?')} -> v{r['version']} "
            f"({r.get('change_type', '?')}) — "
            f"{str(r.get('created_at', ''))[:16]}"
        ):
            st.write(r.get("change_description", "No description"))


def _render_deprecations_tab(db: object) -> None:
    """Render skill deprecation lifecycle."""
    st.subheader("Deprecation Tracker")

    try:
        rows = db.fetchall(
            "SELECT * FROM skill_deprecations ORDER BY flagged_at DESC",
            (),
        )
    except Exception:
        rows = []

    if not rows:
        st.info("No skills flagged for deprecation.")
        return

    # Status counts
    statuses = [r.get("status", "?") for r in rows]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Flagged", statuses.count("flagged"))
    with c2:
        st.metric("Deprecated", statuses.count("deprecated"))
    with c3:
        st.metric("Retired", statuses.count("retired"))

    st.divider()

    for r in rows:
        status = r.get("status", "?")
        color = {
            "flagged": "orange",
            "deprecated": "red",
            "retired": "gray",
        }.get(status, "blue")

        with st.expander(
            f":{color}[{status}] **{r.get('skill_name', '?')}** — {r.get('reason', 'No reason')}"
        ):
            c1, c2 = st.columns(2)
            with c1:
                sr = r.get("success_rate_at_flag", 0)
                st.write(f"**Success rate at flag:** {float(sr):.0%}")
                st.write(f"**Invocations at flag:** {r.get('invocations_at_flag', 0)}")
            with c2:
                st.write(f"**Flagged:** {str(r.get('flagged_at', ''))[:16]}")
                dep = r.get("deprecated_at")
                if dep:
                    st.write(f"**Deprecated:** {str(dep)[:16]}")
                ret = r.get("retired_at")
                if ret:
                    st.write(f"**Retired:** {str(ret)[:16]}")
            replacement = r.get("replacement_skill")
            if replacement:
                st.write(f"**Replacement:** {replacement}")
