# TASK-007: Mobile HTML UI

## Objective
Vanilla-JS single-page chat UI, mobile-responsive, hitting the FastAPI endpoint.

## Constraints
- No build step. Pure HTML/CSS/JS.
- Total payload < 50KB (uncompressed).
- Must work on iPhone Safari and Chrome Android.
- Must show build queue and budget bar.
- Budget: 600 ET.

## Inputs
- `packages/kernel/src/animus_kernel/server/app.py`
- SSE event format from TASK-006.

## Outputs
- `packages/kernel/src/animus_kernel/server/static/index.html`
- `packages/kernel/src/animus_kernel/server/static/style.css`
- `packages/kernel/src/animus_kernel/server/static/app.js`

## Acceptance Criteria
1. Loads on iPhone Safari (iOS 17+) without horizontal scroll.
2. Sends message and displays streaming response with markdown-like formatting.
3. Shows a budget bar (green/yellow/red) synced with kernel budget.
4. Shows a build queue list with status icons.
5. Works without internet (local network only).

## Rubric
- format_compliance [2.0] — CSS works on mobile.
- schema_valid [1.0] — JS consumes SSE correctly.
- concision [1.0] — under 50KB, no bloat.

## Exclusions
- No React/Vue/Angular.
- No WebSocket fallback.
- No dark mode toggle.
- No file drag-and-drop.

## Dependencies
- BLOCKS: none
- BLOCKED_BY: TASK-006
