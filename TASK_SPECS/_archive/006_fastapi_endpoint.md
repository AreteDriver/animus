# TASK-006: FastAPI Chat Endpoint

## Objective
Serve a `POST /chat` endpoint that routes to `TerminalAgent` via `OllamaProvider`.

## Constraints
- Single-file FastAPI app. No frontend framework.
- Must stream SSE responses.
- Must include CORS for local development.
- Must gracefully degrade if Ollama is unreachable.
- Budget: 800 ET.

## Inputs
- `packages/kernel/src/animus_kernel/providers/ollama_provider.py`
- `packages/kernel/src/animus_kernel/builder/terminal_agent.py`

## Outputs
- `packages/kernel/src/animus_kernel/server/app.py` (new)
- `packages/kernel/src/animus_kernel/server/__init__.py`

## Acceptance Criteria
1. `curl -X POST http://localhost:8000/chat -d '{"message":"Add OAuth"}'` streams tokens via SSE.
2. Response ends with `event: done\ndata: {"status":"complete"}`.
3. Errors return JSON with `{"error":"...","detail":"..."}`.
4. Ollama unreachable returns `503` with clear message.
5. CORS allows `http://localhost:3000` and `http://localhost:8080`.

## Rubric
- correctness [3.0] — endpoint responds, streams, errors correctly.
- schema_valid [1.5] — OpenAPI schema generated automatically.
- format_compliance [1.0] — SSE format matches spec.

## Exclusions
- No auth / rate limiting (serves localhost only).
- No persistent chat history.
- No file upload endpoint.

## Dependencies
- BLOCKS: TASK-007
- BLOCKED_BY: TASK-005
