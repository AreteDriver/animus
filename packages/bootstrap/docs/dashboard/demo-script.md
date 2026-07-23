# Animus Operations Center — Demo Script

A 2-minute walkthrough for stakeholders. Read this aloud while navigating the dashboard.

---

## Opening (0:00–0:15)

"This is the Animus Operations Center. It is not a static admin panel — it is a live instrumented surface that shows exactly what the Animus runtime is doing right now, and gives the operator direct control over it."

**Action:** Open the dashboard at `http://localhost:7700`.

---

## System Health (0:15–0:45)

"At the top of the home page, we see a composite health score. It is calculated from three live factors: error rate, tool failure rate, and recent alerts. Right now the system is healthy — 100 out of 100. If something degrades, this card turns yellow. If it becomes critical, it turns red."

**Action:** Point to the health score card. If alerts are present, point to the red alert banner and demonstrate the Acknowledge button.

---

## Live Telemetry (0:45–1:15)

"Every operational action is recorded in the Event Ledger. Tool executions, task completions, config changes, feedback — everything. The home page shows the last five events, and the Events page shows a live feed that refreshes every five seconds via HTMX. There is also a Server-Sent Events stream at `/events/stream` for external consumers."

**Action:** Navigate to `/events`. Watch the feed update. Mention the SSE stream indicator.

---

## Operational Control (1:15–1:45)

"This is not read-only. The operator can pause the runtime, kill a stuck task, clear memory, re-run a tool from history, export the full event log as JSON or CSV, and acknowledge alerts. Every action is itself recorded in the ledger, so there is a full audit trail."

**Action:** Demonstrate one control:
- Pause the runtime from the home page, then resume it.
- Or navigate to Tasks, create a task, then kill it.
- Or navigate to Events and click Export JSON.

---

## Security (1:45–2:00)

"Every state-changing action is protected by CSRF token validation. The middleware enforces this automatically — there is no way to accidentally leave an endpoint unprotected. The dashboard is also CDN-free; all assets are served locally."

**Action:** Mention the security test suite: 65 tests covering controls, events, alerting, and CSRF protection.

---

## Closing

"The Operations Center turns the Animus dashboard from a configuration viewer into a genuine control surface. It is live, instrumented, and operator-safe."
