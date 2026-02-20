"""Evaluation dashboard page — quality trends and benchmarking results."""

from __future__ import annotations

import streamlit as st


def render_eval_page() -> None:
    """Main entry point for the evaluation dashboard page."""
    st.title("Evaluations")

    try:
        from animus_forge.evaluation.store import EvalStore
        from animus_forge.state.database import get_database

        store = EvalStore(get_database())
    except Exception as e:
        st.error(f"Failed to connect to eval store: {e}")
        return

    # ── Filters ──
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        agent_role = st.text_input("Agent Role Filter", value="", placeholder="e.g. planner")
    with col_filter2:
        days = st.slider("Days of history", min_value=1, max_value=90, value=30)

    agent_filter = agent_role.strip() or None

    # ── Summary metrics ──
    runs = store.query_runs(agent_role=agent_filter, limit=1000)

    if not runs:
        st.info("No evaluation runs found. Run `gorgon eval run <suite> --mock` to get started.")
        return

    total_runs = len(runs)
    avg_score = sum(r.get("avg_score", 0) for r in runs) / total_runs if total_runs else 0
    avg_pass_rate = sum(r.get("pass_rate", 0) for r in runs) / total_runs if total_runs else 0
    total_cases = sum(r.get("total_cases", 0) for r in runs)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", total_runs)
    with col2:
        st.metric("Avg Score", f"{avg_score:.0%}")
    with col3:
        st.metric("Avg Pass Rate", f"{avg_pass_rate:.0%}")
    with col4:
        st.metric("Total Cases", total_cases)

    st.divider()

    # ── Quality trend chart ──
    st.subheader("Quality Trend")

    suite_names = sorted({r["suite_name"] for r in runs})
    selected_suite = st.selectbox("Suite", suite_names) if suite_names else None

    if selected_suite:
        trend = store.get_suite_trend(selected_suite, days=days)
        if trend:
            chart_data = {
                "Date": [t["completed_at"][:10] for t in trend],
                "Avg Score": [t["avg_score"] for t in trend],
                "Pass Rate": [t["pass_rate"] for t in trend],
            }
            st.line_chart(chart_data, x="Date", y=["Avg Score", "Pass Rate"])
        else:
            st.info(f"No trend data for '{selected_suite}' in the last {days} days.")

    st.divider()

    # ── Recent runs table ──
    st.subheader("Recent Runs")

    for r in runs[:20]:
        pass_rate = r.get("pass_rate", 0)
        color = "green" if pass_rate >= 0.7 else "orange" if pass_rate >= 0.5 else "red"
        passed = r.get("passed", 0)
        total = r.get("total_cases", 0)

        with st.expander(
            f":{color}[{r['suite_name']}] — "
            f"{passed}/{total} passed ({pass_rate:.0%}) — "
            f"{r.get('run_mode', '?')} — "
            f"{str(r.get('completed_at', ''))[:16]}"
        ):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write(f"**Run ID:** `{r['id'][:8]}`")
                st.write(f"**Agent:** {r.get('agent_role') or 'N/A'}")
            with col_b:
                st.write(f"**Model:** {r.get('model') or 'N/A'}")
                st.write(f"**Mode:** {r.get('run_mode', 'N/A')}")
            with col_c:
                st.write(f"**Duration:** {r.get('duration_ms', 0):.0f}ms")
                st.write(f"**Avg Score:** {r.get('avg_score', 0):.2%}")

            # Show case-level details
            full_run = store.get_run(r["id"])
            if full_run and full_run.get("case_results"):
                case_rows = []
                for cr in full_run["case_results"]:
                    case_rows.append(
                        {
                            "Case": cr["case_name"],
                            "Status": cr["status"],
                            "Score": f"{cr['score']:.2%}",
                            "Latency": f"{cr.get('latency_ms', 0):.0f}ms",
                        }
                    )
                st.dataframe(case_rows, use_container_width=True)
