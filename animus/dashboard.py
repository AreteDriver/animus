"""
Animus Learning Dashboard

Simple HTML dashboard for learning transparency, served by the API server.
"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Animus — Learning Dashboard</title>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #c9d1d9;
    --muted: #8b949e; --accent: #58a6ff; --green: #3fb950; --yellow: #d29922;
    --red: #f85149; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5; padding: 1.5rem; }
  h1 { font-size: 1.5rem; color: var(--accent); margin-bottom: 0.25rem; }
  h2 { font-size: 1.1rem; color: var(--text); margin-bottom: 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; }
  .card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
  .card .value { font-size: 1.75rem; font-weight: 600; margin-top: 0.25rem; }
  .card .value.green { color: var(--green); }
  .card .value.yellow { color: var(--yellow); }
  .card .value.red { color: var(--red); }
  .card .value.accent { color: var(--accent); }
  .section { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; margin-bottom: 1.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { text-align: left; color: var(--muted); font-weight: 500; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
  td { padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
  .badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500; }
  .badge.active { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge.pending { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .badge.blocked { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge.immutable { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge.user { background: rgba(88,166,255,0.15); color: var(--accent); }
  .bar-chart { display: flex; gap: 0.5rem; align-items: flex-end; height: 80px; margin-top: 0.5rem; }
  .bar-item { display: flex; flex-direction: column; align-items: center; flex: 1; }
  .bar { background: var(--accent); border-radius: 3px 3px 0 0; min-height: 4px; width: 100%; }
  .bar-label { font-size: 0.65rem; color: var(--muted); margin-top: 0.25rem; white-space: nowrap; }
  .bar-value { font-size: 0.7rem; color: var(--text); margin-bottom: 0.15rem; }
  #error { display: none; background: rgba(248,81,73,0.15); border: 1px solid var(--red); color: var(--red); padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem; }
  .refresh-btn { background: var(--surface); border: 1px solid var(--border); color: var(--accent); padding: 0.4rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem; float: right; }
  .refresh-btn:hover { background: var(--border); }
</style>
</head>
<body>
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div><h1>Animus</h1><p class="subtitle">Learning Transparency Dashboard</p></div>
  <button class="refresh-btn" onclick="loadAll()">Refresh</button>
</div>
<div id="error"></div>

<div class="grid" id="summary-cards">
  <div class="card"><div class="label">Total Learned</div><div class="value accent" id="total-learned">—</div></div>
  <div class="card"><div class="label">Pending Approval</div><div class="value yellow" id="pending-approval">—</div></div>
  <div class="card"><div class="label">Events Today</div><div class="value green" id="events-today">—</div></div>
  <div class="card"><div class="label">Guardrail Violations</div><div class="value red" id="violations">—</div></div>
</div>

<div class="grid">
  <div class="section">
    <h2>By Category</h2>
    <div class="bar-chart" id="category-chart"></div>
  </div>
  <div class="section">
    <h2>Confidence Distribution</h2>
    <div class="bar-chart" id="confidence-chart"></div>
  </div>
</div>

<div class="section">
  <h2>Learned Items</h2>
  <table>
    <thead><tr><th>ID</th><th>Category</th><th>Content</th><th>Confidence</th><th>Status</th></tr></thead>
    <tbody id="items-table"></tbody>
  </table>
</div>

<div class="section">
  <h2>Guardrails</h2>
  <table>
    <thead><tr><th>ID</th><th>Type</th><th>Rule</th><th>Scope</th></tr></thead>
    <tbody id="guardrails-table"></tbody>
  </table>
</div>

<div class="section">
  <h2>Recent Events</h2>
  <table>
    <thead><tr><th>Time</th><th>Event</th><th>Item</th></tr></thead>
    <tbody id="events-table"></tbody>
  </table>
</div>

<script>
const BASE = window.location.origin;
const KEY = document.cookie.match(/api_key=([^;]+)/)?.[1] || '';
const headers = KEY ? {'X-API-Key': KEY} : {};

function showError(msg) {
  const el = document.getElementById('error');
  el.textContent = msg; el.style.display = 'block';
}

async function fetchJSON(path) {
  const resp = await fetch(BASE + path, {headers});
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json();
}

function renderBar(container, data, colorFn) {
  const el = document.getElementById(container);
  el.innerHTML = '';
  const max = Math.max(...Object.values(data), 1);
  for (const [label, count] of Object.entries(data)) {
    const pct = (count / max) * 100;
    el.innerHTML += `<div class="bar-item"><div class="bar-value">${count}</div><div class="bar" style="height:${Math.max(pct, 5)}%;${colorFn ? colorFn(label) : ''}"></div><div class="bar-label">${label}</div></div>`;
  }
}

async function loadDashboard() {
  try {
    const d = await fetchJSON('/learning/status');
    document.getElementById('total-learned').textContent = d.total_learned;
    document.getElementById('pending-approval').textContent = d.pending_approval;
    document.getElementById('events-today').textContent = d.events_today;
    document.getElementById('violations').textContent = d.guardrail_violations;
    if (d.by_category) renderBar('category-chart', d.by_category);
    if (d.confidence_distribution) renderBar('confidence-chart', d.confidence_distribution);
  } catch (e) { showError('Failed to load dashboard: ' + e.message); }
}

async function loadItems() {
  try {
    const d = await fetchJSON('/learning/items?status=all');
    const tb = document.getElementById('items-table');
    tb.innerHTML = '';
    for (const item of (d.items || [])) {
      const badge = item.applied ? 'active' : 'pending';
      const label = item.applied ? 'Active' : 'Pending';
      tb.innerHTML += `<tr><td><code>${item.id.slice(0,8)}</code></td><td>${item.category}</td><td>${item.content.slice(0,60)}${item.content.length > 60 ? '…' : ''}</td><td>${(item.confidence * 100).toFixed(0)}%</td><td><span class="badge ${badge}">${label}</span></td></tr>`;
    }
    if (!d.items?.length) tb.innerHTML = '<tr><td colspan="5" style="color:var(--muted)">No learned items yet</td></tr>';
  } catch (e) { /* dashboard may not have learning */ }
}

async function loadGuardrails() {
  try {
    const d = await fetchJSON('/guardrails');
    const tb = document.getElementById('guardrails-table');
    tb.innerHTML = '';
    for (const g of (d.guardrails || [])) {
      const badge = g.immutable ? 'immutable' : 'user';
      const label = g.immutable ? 'Core' : 'User';
      tb.innerHTML += `<tr><td><code>${g.id.slice(0,12)}</code></td><td>${g.guardrail_type}</td><td>${g.rule}</td><td><span class="badge ${badge}">${label}</span></td></tr>`;
    }
  } catch (e) { /* ignore */ }
}

async function loadHistory() {
  try {
    const d = await fetchJSON('/learning/history?limit=20');
    const tb = document.getElementById('events-table');
    tb.innerHTML = '';
    for (const ev of (d.events || [])) {
      const ts = new Date(ev.timestamp).toLocaleString();
      tb.innerHTML += `<tr><td style="white-space:nowrap">${ts}</td><td>${ev.event_type}</td><td><code>${(ev.learned_item_id || '').slice(0,8)}</code></td></tr>`;
    }
    if (!d.events?.length) tb.innerHTML = '<tr><td colspan="3" style="color:var(--muted)">No events yet</td></tr>';
  } catch (e) { /* ignore */ }
}

function loadAll() {
  document.getElementById('error').style.display = 'none';
  loadDashboard(); loadItems(); loadGuardrails(); loadHistory();
}

loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>"""
