"""
Animus Web Dashboard

Serves a self-contained HTML dashboard for memory browsing,
learning transparency, entity visualization, and proactive nudges.
No external JS/CSS dependencies — fully self-contained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from animus.logging import get_logger

if TYPE_CHECKING:
    from animus.entities import EntityMemory
    from animus.memory import MemoryLayer
    from animus.proactive import ProactiveEngine

logger = get_logger("dashboard")


# =========================================================================
# Dashboard HTML template (self-contained, no CDN dependencies)
# =========================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Animus Dashboard</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --border: #30363d;
  --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
  --green: #3fb950; --yellow: #d29922; --red: #f85149; --purple: #bc8cff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.5; }
.container { max-width: 1200px; margin: 0 auto; padding: 16px; }
header { display: flex; align-items: center; justify-content: space-between;
         padding: 12px 0; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
header h1 { font-size: 20px; font-weight: 600; }
header h1 span { color: var(--accent); }
.refresh-btn { background: var(--surface); border: 1px solid var(--border);
               color: var(--text); padding: 6px 14px; border-radius: 6px;
               cursor: pointer; font-size: 13px; }
.refresh-btn:hover { border-color: var(--accent); }

/* Grid layout */
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
        gap: 16px; margin-bottom: 20px; }
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 16px; }
.card h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;
           color: var(--muted); margin-bottom: 12px; }

/* Stats */
.stat-row { display: flex; justify-content: space-between; padding: 6px 0;
            border-bottom: 1px solid var(--border); }
.stat-row:last-child { border: none; }
.stat-label { color: var(--muted); font-size: 13px; }
.stat-value { font-weight: 600; font-size: 14px; }

/* Nudges */
.nudge { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
         padding: 10px 12px; margin-bottom: 8px; }
.nudge-header { display: flex; justify-content: space-between; align-items: center; }
.nudge-title { font-weight: 600; font-size: 13px; }
.badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-urgent { background: var(--red); color: #fff; }
.badge-high { background: var(--yellow); color: #000; }
.badge-medium { background: var(--accent); color: #000; }
.badge-low { background: var(--border); color: var(--text); }
.nudge-content { color: var(--muted); font-size: 12px; margin-top: 4px;
                 white-space: pre-wrap; max-height: 80px; overflow-y: auto; }

/* Entity list */
.entity { display: flex; justify-content: space-between; align-items: center;
          padding: 8px 0; border-bottom: 1px solid var(--border); }
.entity:last-child { border: none; }
.entity-name { font-weight: 500; }
.entity-type { color: var(--muted); font-size: 12px; }
.entity-meta { text-align: right; font-size: 12px; color: var(--muted); }

/* Memory list */
.memory { padding: 8px 0; border-bottom: 1px solid var(--border); }
.memory:last-child { border: none; }
.memory-content { font-size: 13px; }
.memory-meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
.tag { display: inline-block; background: var(--border); padding: 1px 6px;
       border-radius: 10px; font-size: 11px; margin-right: 4px; }

/* Timeline */
.timeline-item { display: flex; gap: 12px; padding: 8px 0;
                 border-bottom: 1px solid var(--border); }
.timeline-item:last-child { border: none; }
.timeline-date { color: var(--muted); font-size: 12px; min-width: 70px; }
.timeline-text { font-size: 13px; }

/* Tabs */
.tabs { display: flex; gap: 4px; margin-bottom: 16px; }
.tab { background: var(--surface); border: 1px solid var(--border); color: var(--muted);
       padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.tab.active { border-color: var(--accent); color: var(--accent); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Search */
.search-box { width: 100%; background: var(--bg); border: 1px solid var(--border);
              color: var(--text); padding: 8px 12px; border-radius: 6px;
              font-size: 14px; margin-bottom: 12px; }
.search-box:focus { outline: none; border-color: var(--accent); }
.empty { color: var(--muted); font-size: 13px; text-align: center; padding: 20px; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1><span>&#9670;</span> Animus Dashboard</h1>
    <button class="refresh-btn" onclick="location.reload()">Refresh</button>
  </header>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('overview')">Overview</div>
    <div class="tab" onclick="switchTab('memories')">Memories</div>
    <div class="tab" onclick="switchTab('entities')">Entities</div>
    <div class="tab" onclick="switchTab('nudges')">Nudges</div>
  </div>

  <!-- Overview Tab -->
  <div id="tab-overview" class="tab-content active">
    <div class="grid">
      <div class="card">
        <h2>Memory Stats</h2>
        <div id="memory-stats"></div>
      </div>
      <div class="card">
        <h2>Entity Stats</h2>
        <div id="entity-stats"></div>
      </div>
      <div class="card">
        <h2>Active Nudges</h2>
        <div id="nudge-summary"></div>
      </div>
      <div class="card">
        <h2>Recent Activity</h2>
        <div id="recent-activity"></div>
      </div>
    </div>
  </div>

  <!-- Memories Tab -->
  <div id="tab-memories" class="tab-content">
    <input type="text" class="search-box" id="memory-search"
           placeholder="Search memories..." onkeyup="searchMemories(this.value)">
    <div id="memory-list"></div>
  </div>

  <!-- Entities Tab -->
  <div id="tab-entities" class="tab-content">
    <input type="text" class="search-box" id="entity-search"
           placeholder="Search entities..." onkeyup="searchEntities(this.value)">
    <div id="entity-list"></div>
  </div>

  <!-- Nudges Tab -->
  <div id="tab-nudges" class="tab-content">
    <div id="nudge-list"></div>
  </div>
</div>

<script>
const DATA = __DASHBOARD_DATA__;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

function statRow(label, value) {
  return `<div class="stat-row"><span class="stat-label">${label}</span>` +
         `<span class="stat-value">${value}</span></div>`;
}

function renderMemoryStats() {
  const s = DATA.memory_stats;
  let html = statRow('Total memories', s.total);
  if (s.by_type) Object.entries(s.by_type).forEach(([k,v]) => html += statRow(k, v));
  html += statRow('Unique tags', s.unique_tags || 0);
  html += statRow('Avg confidence', (s.avg_confidence || 0).toFixed(2));
  document.getElementById('memory-stats').innerHTML = html;
}

function renderEntityStats() {
  const s = DATA.entity_stats;
  let html = statRow('Total entities', s.total_entities);
  html += statRow('Relationships', s.total_relationships);
  html += statRow('Interactions', s.total_interactions);
  html += statRow('Active (7d)', s.recently_active || 0);
  html += statRow('Dormant (30d+)', s.dormant || 0);
  if (s.by_type) Object.entries(s.by_type).forEach(([k,v]) => html += statRow(k, v));
  document.getElementById('entity-stats').innerHTML = html;
}

function badgeClass(priority) {
  return 'badge badge-' + priority;
}

function renderNudgeSummary() {
  const nudges = DATA.nudges.filter(n => !n.dismissed && !n.acted_on);
  if (!nudges.length) {
    document.getElementById('nudge-summary').innerHTML = '<div class="empty">No active nudges</div>';
    return;
  }
  let html = '';
  nudges.slice(0, 5).forEach(n => {
    html += `<div class="nudge"><div class="nudge-header">` +
      `<span class="nudge-title">${n.title}</span>` +
      `<span class="${badgeClass(n.priority)}">${n.priority}</span>` +
      `</div><div class="nudge-content">${n.content.substring(0, 200)}</div></div>`;
  });
  if (nudges.length > 5) html += `<div class="empty">+${nudges.length - 5} more</div>`;
  document.getElementById('nudge-summary').innerHTML = html;
}

function renderRecentActivity() {
  const memories = DATA.memories.slice(0, 8);
  if (!memories.length) {
    document.getElementById('recent-activity').innerHTML = '<div class="empty">No recent activity</div>';
    return;
  }
  let html = '';
  memories.forEach(m => {
    const date = new Date(m.created_at).toLocaleDateString();
    html += `<div class="timeline-item"><span class="timeline-date">${date}</span>` +
      `<span class="timeline-text">${m.content.substring(0, 120)}</span></div>`;
  });
  document.getElementById('recent-activity').innerHTML = html;
}

function renderMemories(filter) {
  let memories = DATA.memories;
  if (filter) {
    const q = filter.toLowerCase();
    memories = memories.filter(m =>
      m.content.toLowerCase().includes(q) ||
      (m.tags && m.tags.some(t => t.includes(q)))
    );
  }
  if (!memories.length) {
    document.getElementById('memory-list').innerHTML = '<div class="empty">No memories found</div>';
    return;
  }
  let html = '';
  memories.slice(0, 50).forEach(m => {
    const date = new Date(m.created_at).toLocaleDateString();
    const tags = (m.tags || []).map(t => `<span class="tag">${t}</span>`).join('');
    html += `<div class="memory"><div class="memory-content">${m.content.substring(0, 300)}</div>` +
      `<div class="memory-meta">${m.memory_type} | ${date} | conf: ${(m.confidence || 1).toFixed(1)} ${tags}</div></div>`;
  });
  document.getElementById('memory-list').innerHTML = html;
}
function searchMemories(q) { renderMemories(q); }

function renderEntities(filter) {
  let entities = DATA.entities;
  if (filter) {
    const q = filter.toLowerCase();
    entities = entities.filter(e =>
      e.name.toLowerCase().includes(q) ||
      (e.aliases && e.aliases.some(a => a.toLowerCase().includes(q)))
    );
  }
  if (!entities.length) {
    document.getElementById('entity-list').innerHTML = '<div class="empty">No entities found</div>';
    return;
  }
  let html = '';
  entities.forEach(e => {
    const lastSeen = e.last_mentioned ? new Date(e.last_mentioned).toLocaleDateString() : 'never';
    const aliases = e.aliases.length ? ` (${e.aliases.join(', ')})` : '';
    html += `<div class="entity"><div><span class="entity-name">${e.name}${aliases}</span>` +
      `<br><span class="entity-type">${e.entity_type}</span></div>` +
      `<div class="entity-meta">${e.mention_count} mentions<br>Last: ${lastSeen}</div></div>`;
  });
  document.getElementById('entity-list').innerHTML = html;
}
function searchEntities(q) { renderEntities(q); }

function renderNudges() {
  const nudges = DATA.nudges;
  if (!nudges.length) {
    document.getElementById('nudge-list').innerHTML = '<div class="empty">No nudges</div>';
    return;
  }
  let html = '';
  nudges.forEach(n => {
    const date = new Date(n.created_at).toLocaleString();
    const status = n.dismissed ? ' (dismissed)' : n.acted_on ? ' (acted)' : '';
    html += `<div class="nudge"><div class="nudge-header">` +
      `<span class="nudge-title">${n.title}${status}</span>` +
      `<span class="${badgeClass(n.priority)}">${n.nudge_type}</span>` +
      `</div><div class="nudge-content">${n.content}</div>` +
      `<div class="memory-meta">${date}</div></div>`;
  });
  document.getElementById('nudge-list').innerHTML = html;
}

// Initial render
renderMemoryStats();
renderEntityStats();
renderNudgeSummary();
renderRecentActivity();
renderMemories();
renderEntities();
renderNudges();
</script>
</body>
</html>"""


def collect_dashboard_data(
    memory: MemoryLayer,
    entity_memory: EntityMemory | None = None,
    proactive: ProactiveEngine | None = None,
    learning: Any = None,
) -> dict[str, Any]:
    """
    Collect all data needed for the dashboard.

    Args:
        memory: Memory layer for memory data
        entity_memory: Entity memory for entity/relationship data
        proactive: Proactive engine for nudge data
        learning: Optional learning layer for learning stats

    Returns:
        Dict with all dashboard data
    """
    # Memory stats and recent memories
    memory_stats = memory.get_statistics()
    all_memories = memory.store.list_all()
    all_memories.sort(key=lambda m: m.created_at, reverse=True)
    memories_data = [m.to_dict() for m in all_memories[:100]]

    # Entity data
    if entity_memory:
        entity_stats = entity_memory.get_statistics()
        entities_data = [e.to_dict() for e in entity_memory.list_entities(limit=100)]
    else:
        entity_stats = {
            "total_entities": 0,
            "total_relationships": 0,
            "total_interactions": 0,
            "by_type": {},
            "recently_active": 0,
            "dormant": 0,
        }
        entities_data = []

    # Nudge data
    if proactive:
        nudges_data = [n.to_dict() for n in proactive._nudges[-50:]]
    else:
        nudges_data = []

    return {
        "memory_stats": memory_stats,
        "memories": memories_data,
        "entity_stats": entity_stats,
        "entities": entities_data,
        "nudges": nudges_data,
    }


def render_dashboard(
    memory: MemoryLayer,
    entity_memory: EntityMemory | None = None,
    proactive: ProactiveEngine | None = None,
    learning: Any = None,
) -> str:
    """
    Render the complete dashboard HTML with embedded data.

    Args:
        memory: Memory layer
        entity_memory: Entity memory
        proactive: Proactive engine
        learning: Learning layer

    Returns:
        Complete HTML string
    """
    import json

    data = collect_dashboard_data(memory, entity_memory, proactive, learning)
    data_json = json.dumps(data, default=str)
    return DASHBOARD_HTML.replace("__DASHBOARD_DATA__", data_json)


def add_dashboard_routes(app: Any, get_state: Any, verify_api_key: Any) -> None:
    """
    Add dashboard routes to an existing FastAPI app.

    Args:
        app: FastAPI application
        get_state: Dependency function that returns AppState
        verify_api_key: Dependency function for auth
    """
    try:
        from fastapi import Depends
        from starlette.responses import HTMLResponse
    except ImportError:
        logger.warning("FastAPI not available, dashboard routes not added")
        return

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(_auth: bool = Depends(verify_api_key)):
        """Serve the web dashboard."""
        state = get_state()
        entity_mem = getattr(state, "entity_memory", None)
        proactive_eng = getattr(state, "proactive", None)
        learning_layer = getattr(state, "learning", None)
        html = render_dashboard(
            state.memory,
            entity_mem,
            proactive_eng,
            learning_layer,
        )
        return HTMLResponse(content=html)

    @app.get("/dashboard/data")
    async def dashboard_data(_auth: bool = Depends(verify_api_key)):
        """Get raw dashboard data as JSON."""
        state = get_state()
        entity_mem = getattr(state, "entity_memory", None)
        proactive_eng = getattr(state, "proactive", None)
        learning_layer = getattr(state, "learning", None)
        return collect_dashboard_data(
            state.memory,
            entity_mem,
            proactive_eng,
            learning_layer,
        )

    logger.info("Dashboard routes added")
