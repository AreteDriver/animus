"""Graph visualization — text tables, DOT, HTML reports, overlap matrices.

All outputs are pure stdlib, no external dependencies. Each function takes
an IntentResolver as input and produces a string representation.
"""

from __future__ import annotations

import html as html_mod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from convergent.resolver import IntentResolver


def text_table(resolver: IntentResolver, *, show_evidence: bool = False) -> str:
    """Render a text table of intents grouped by agent.

    Columns: Agent | Intent | Stability | Provides | Requires

    Args:
        resolver: IntentResolver with published intents.
        show_evidence: If True, include evidence details per intent.
    """
    intents = resolver.backend.query_all(min_stability=0.0)
    if not intents:
        return "(empty graph)"

    # Group by agent
    by_agent: dict[str, list] = {}
    for intent in intents:
        by_agent.setdefault(intent.agent_id, []).append(intent)

    lines: list[str] = []
    header = f"{'Agent':<16} {'Intent':<30} {'Stab':>5} {'Provides':<25} {'Requires':<25}"
    lines.append(header)
    lines.append("-" * len(header))

    for agent_id in sorted(by_agent):
        for intent in by_agent[agent_id]:
            stab = f"{intent.compute_stability():.2f}"
            provides = ", ".join(s.name for s in intent.provides) or "-"
            requires = ", ".join(s.name for s in intent.requires) or "-"
            lines.append(
                f"{agent_id:<16} {intent.intent:<30} {stab:>5} {provides:<25} {requires:<25}"
            )
            if show_evidence and intent.evidence:
                for ev in intent.evidence:
                    lines.append(f"{'':>16}   [{ev.kind.value}] {ev.description}")
        lines.append("")

    return "\n".join(lines)


def dot_graph(resolver: IntentResolver, *, min_stability: float = 0.0) -> str:
    """Render a DOT format graph for graphviz.

    Nodes are intents, edges represent overlaps, subgraphs group by agent.
    Node color intensity reflects stability (darker = higher).

    Args:
        resolver: IntentResolver with published intents.
        min_stability: Filter intents below this stability.
    """
    intents = resolver.backend.query_all(min_stability=min_stability)

    lines: list[str] = ["digraph convergent {", "  rankdir=LR;"]

    # Group by agent for subgraphs
    by_agent: dict[str, list] = {}
    for intent in intents:
        by_agent.setdefault(intent.agent_id, []).append(intent)

    # Subgraphs
    for idx, (agent_id, agent_intents) in enumerate(sorted(by_agent.items())):
        lines.append(f"  subgraph cluster_{idx} {{")
        lines.append(f'    label="{agent_id}";')
        for intent in agent_intents:
            stab = intent.compute_stability()
            # Map stability 0..1 to grayscale (1.0=dark, 0.0=light)
            gray = 1.0 - stab
            color = f"{gray:.2f} {gray:.2f} {gray:.2f}"
            label = f"{intent.intent}\\n({stab:.2f})"
            node_id = intent.id.replace("-", "_")
            lines.append(f'    "{node_id}" [label="{label}", style=filled, fillcolor="{color}"];')
        lines.append("  }")

    # Edges: overlaps between intents from different agents
    for i, a in enumerate(intents):
        for b in intents[i + 1 :]:
            if a.agent_id == b.agent_id:
                continue
            a_specs = a.provides + a.requires
            b_specs = b.provides + b.requires
            overlap = any(sa.structurally_overlaps(sb) for sa in a_specs for sb in b_specs)
            if overlap:
                a_id = a.id.replace("-", "_")
                b_id = b.id.replace("-", "_")
                lines.append(f'  "{a_id}" -> "{b_id}" [dir=both, style=dashed];')

    lines.append("}")
    return "\n".join(lines)


def html_report(resolver: IntentResolver) -> str:
    """Generate a self-contained HTML report with summary stats and agent table.

    Args:
        resolver: IntentResolver with published intents.
    """
    intents = resolver.backend.query_all(min_stability=0.0)

    by_agent: dict[str, list] = {}
    for intent in intents:
        by_agent.setdefault(intent.agent_id, []).append(intent)

    total = len(intents)
    agent_count = len(by_agent)
    avg_stab = sum(i.compute_stability() for i in intents) / total if total else 0.0

    # Build HTML
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Convergent Report</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 2em; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }",
        "th { background: #f0f0f0; }",
        ".stab { text-align: right; }",
        "</style>",
        "</head><body>",
        "<h1>Convergent Intent Graph Report</h1>",
        "<h2>Summary</h2>",
        "<ul>",
        f"<li>Total intents: {total}</li>",
        f"<li>Agents: {agent_count}</li>",
        f"<li>Average stability: {avg_stab:.2f}</li>",
        "</ul>",
        "<h2>Intents by Agent</h2>",
        "<table>",
        (
            "<tr><th>Agent</th><th>Intent</th><th class='stab'>Stability</th>"
            "<th>Provides</th><th>Requires</th></tr>"
        ),
    ]

    for agent_id in sorted(by_agent):
        for intent in by_agent[agent_id]:
            stab = intent.compute_stability()
            provides = ", ".join(html_mod.escape(s.name) for s in intent.provides) or "-"
            requires = ", ".join(html_mod.escape(s.name) for s in intent.requires) or "-"
            parts.append(
                f"<tr><td>{html_mod.escape(agent_id)}</td>"
                f"<td>{html_mod.escape(intent.intent)}</td>"
                f"<td class='stab'>{stab:.2f}</td>"
                f"<td>{provides}</td>"
                f"<td>{requires}</td></tr>"
            )

    parts.extend(["</table>", "</body></html>"])
    return "\n".join(parts)


def overlap_matrix(resolver: IntentResolver) -> str:
    """Render a text matrix showing which intents overlap.

    Rows and columns are intent labels (agent:name). Overlaps are marked
    with 'X', self-intersections with '.'.

    Args:
        resolver: IntentResolver with published intents.
    """
    intents = resolver.backend.query_all(min_stability=0.0)
    if not intents:
        return "(empty graph)"

    # Build labels
    labels = [f"{i.agent_id}:{i.intent[:15]}" for i in intents]
    max_label = max(len(label) for label in labels)

    # Build overlap matrix
    n = len(intents)
    matrix = [["." if i == j else " " for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if intents[i].agent_id == intents[j].agent_id:
                continue
            a_specs = intents[i].provides + intents[i].requires
            b_specs = intents[j].provides + intents[j].requires
            overlap = any(sa.structurally_overlaps(sb) for sa in a_specs for sb in b_specs)
            if overlap:
                matrix[i][j] = "X"
                matrix[j][i] = "X"

    # Render
    lines: list[str] = []
    # Header row with column indices
    header = " " * (max_label + 1) + " ".join(f"{i:>2}" for i in range(n))
    lines.append(header)

    for i, label in enumerate(labels):
        row = " ".join(f"{matrix[i][j]:>2}" for j in range(n))
        lines.append(f"{label:<{max_label}} {row}")

    # Legend
    lines.append("")
    for i, label in enumerate(labels):
        lines.append(f"  {i:>2}: {label}")

    return "\n".join(lines)
