"""Animus Head Session Daemon (JSON-RPC over stdio).

A long-running background process that manages one or more HeadREPL
sessions via a lightweight JSON-RPC protocol.  Each session runs in its
own thread and communicates through thread-safe queues.

Protocol (JSON-RPC 2.0 over stdio, one request per line):

    → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"project_root":"/path/to/repo"}}
    ← {"jsonrpc":"2.0","id":1,"result":{"session_id":"sess_abc123","status":"ready"}}

    → {"jsonrpc":"2.0","id":2,"method":"process_message","params":{"session_id":"sess_abc123","message":"read the readme"}}
    ← {"jsonrpc":"2.0","id":2,"result":{"response":"...","tokens_used":42,"fallback_used":false}}

    → {"jsonrpc":"2.0","id":3,"method":"get_status","params":{"session_id":"sess_abc123"}}
    ← {"jsonrpc":"2.0","id":3,"result":{"turns":7,"tokens_used":1240,"last_active":"2026-07-02T14:23:00Z"}}

    → {"jsonrpc":"2.0","id":4,"method":"list_sessions"}
    ← {"jsonrpc":"2.0","id":4,"result":{"sessions":["sess_abc123"],"total":1}}

    → {"jsonrpc":"2.0","id":5,"method":"shutdown","params":{"session_id":"sess_abc123"}}
    ← {"jsonrpc":"2.0","id":5,"result":{"status":"shutting_down"}}

For notification-style (no response) use `"id": null`.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

from animus_kernel.head.checkpoint import HeadCheckpointStore
from animus_kernel.head.repl import HeadREPL
from animus_kernel.head.session_controller import SessionController, SessionPolicy


@dataclass
class SessionState:
    """Runtime state for a daemon-managed session."""

    session_id: str
    project_root: Path
    repl: HeadREPL | None = None
    thread: threading.Thread | None = None
    request_queue: Queue = field(default_factory=Queue)
    response_queue: Queue = field(default_factory=Queue)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    total_messages: int = 0
    shutdown_requested: bool = False
    error: str | None = None


class HeadDaemon:
    """JSON-RPC daemon managing HeadREPL sessions."""

    def __init__(self, checkpoint_dir: Path | None = None, model: str = "local") -> None:
        self.sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        self.checkpoint_dir = checkpoint_dir or Path(".animus/head/checkpoints")
        self.default_model = model
        self._store = HeadCheckpointStore(self.checkpoint_dir)

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT, self._on_sigterm)

    # ------------------------------------------------------------------
    # JSON-RPC transport
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Read JSON-RPC requests from stdin and dispatch."""
        self._write_jsonrpc(None, {"status": "ready", "version": "2.3"})
        for line in sys.stdin:
            if self._shutdown:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                self._write_jsonrpc(None, error={"code": -32700, "message": f"Parse error: {exc}"})
                continue

            resp = self._dispatch(req)
            if resp is not None:
                self._write_jsonrpc(resp.get("id"), resp.get("result"), resp.get("error"))

        # Final checkpoint of all sessions
        self._checkpoint_all()
        self._write_jsonrpc(None, {"status": "stopped"})

    @staticmethod
    def _write_jsonrpc(
        req_id: Any,
        result: dict | None = None,
        error: dict | None = None,
    ) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0"}
        if req_id is not None:
            payload["id"] = req_id
        if error:
            payload["error"] = error
        else:
            payload["result"] = result or {}
        print(json.dumps(payload), flush=True)

    def _dispatch(self, req: dict) -> dict | None:
        method = req.get("method")
        params = req.get("params", {})
        req_id = req.get("id")

        if not isinstance(method, str):
            return {"id": req_id, "error": {"code": -32600, "message": "Invalid request"}}

        handler = getattr(self, f"_rpc_{method}", None)
        if handler is None:
            return {
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        try:
            result = handler(params)
            return {"id": req_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {"id": req_id, "error": {"code": -32000, "message": f"Server error: {exc}"}}

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    def _rpc_initialize(self, params: dict) -> dict:
        """Create a new HeadREPL session and start its worker thread."""
        import datetime

        project_root = Path(params.get("project_root", "."))
        session_id = params.get("session_id") or f"sess_{uuid.uuid4().hex[:8]}"
        model = params.get("model", self.default_model)

        # Session policy overrides from params
        timer_minutes = params.get("session_timer_minutes")
        wrapup_threshold = params.get("wrapup_threshold")
        auto_restart = params.get("auto_restart")

        policy = SessionPolicy()
        if timer_minutes is not None:
            policy.session_timer = datetime.timedelta(minutes=timer_minutes)
        if wrapup_threshold is not None:
            policy.wrapup_threshold = wrapup_threshold
        if auto_restart is not None:
            policy.auto_restart = bool(auto_restart)

        with self._lock:
            if session_id in self.sessions:
                return {"session_id": session_id, "status": "already_exists"}

            state = SessionState(session_id=session_id, project_root=project_root)
            self.sessions[session_id] = state

        # Build the REPL on the worker thread to avoid blocking stdin
        def _worker() -> None:
            try:
                controller = SessionController(policy=policy)
                state.repl = HeadREPL(
                    project_root=project_root,
                    model=model,
                    checkpoint_store=self._store,
                    session_timer=policy.session_timer,
                    wrapup_threshold=policy.wrapup_threshold,
                    session_controller=controller,
                )
                state.repl._restore_session()
            except Exception as exc:  # noqa: BLE001
                state.error = str(exc)
                return

            # Pump messages
            while not state.shutdown_requested:
                try:
                    msg = state.request_queue.get(timeout=0.5)
                except Exception:  # noqa: BLE001
                    continue
                if msg is None:
                    break
                state.last_active = time.time()
                try:
                    result = state.repl.process_message(msg["text"])
                except Exception as exc:  # noqa: BLE001
                    result = {"error": str(exc)}
                state.total_messages += 1
                state.response_queue.put({"id": msg.get("id"), **result})

            # Checkpoint before exit
            if state.repl:
                state.repl._checkpoint()

        state.thread = threading.Thread(target=_worker, daemon=True)
        state.thread.start()
        return {"session_id": session_id, "status": "initializing"}

    def _rpc_process_message(self, params: dict) -> dict:
        session_id = params.get("session_id")
        message = params.get("message")
        if not session_id or not message:
            raise ValueError("session_id and message are required")

        with self._lock:
            state = self.sessions.get(session_id)
        if not state:
            raise ValueError(f"Unknown session: {session_id}")
        if state.error:
            raise ValueError(f"Session error: {state.error}")

        msg_id = f"msg_{uuid.uuid4().hex[:6]}"
        state.request_queue.put({"id": msg_id, "text": message})

        # Block until response (with timeout)
        deadline = time.time() + 300  # 5 minutes
        while time.time() < deadline:
            try:
                resp = state.response_queue.get(timeout=0.5)
                if resp.get("id") == msg_id:
                    return resp
                # Otherwise re-queue if it belongs to someone else
                state.response_queue.put(resp)
            except Exception:  # noqa: BLE001
                continue
        raise TimeoutError(f"No response within 5 minutes for session {session_id}")

    def _rpc_get_status(self, params: dict) -> dict:
        session_id = params.get("session_id")
        with self._lock:
            state = self.sessions.get(session_id)
        if not state:
            raise ValueError(f"Unknown session: {session_id}")

        return {
            "session_id": session_id,
            "turns": state.total_messages,
            "last_active": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(state.last_active)),
            "project_root": str(state.project_root),
            "error": state.error,
        }

    def _rpc_list_sessions(self, _params: dict) -> dict:
        with self._lock:
            ids = list(self.sessions.keys())
        return {"sessions": ids, "total": len(ids)}

    def _rpc_shutdown(self, params: dict) -> dict:
        session_id = params.get("session_id")
        with self._lock:
            state = self.sessions.pop(session_id, None)
        if not state:
            raise ValueError(f"Unknown session: {session_id}")

        state.shutdown_requested = True
        state.request_queue.put(None)
        if state.thread:
            state.thread.join(timeout=5.0)
        return {"status": "shutting_down", "session_id": session_id}

    def _rpc_get_session_policy(self, params: dict) -> dict:
        session_id = params.get("session_id")
        with self._lock:
            state = self.sessions.get(session_id)
        if not state:
            raise ValueError(f"Unknown session: {session_id}")

        if state.repl and state.repl._session_controller:
            policy = state.repl._session_controller.policy
            return {
                "wrapup_threshold": policy.wrapup_threshold,
                "session_timer_seconds": (
                    policy.session_timer.total_seconds() if policy.session_timer else None
                ),
                "auto_restart": policy.auto_restart,
            }
        return {"policy": None}

    def _rpc_set_session_policy(self, params: dict) -> dict:
        import datetime

        session_id = params.get("session_id")
        with self._lock:
            state = self.sessions.get(session_id)
        if not state:
            raise ValueError(f"Unknown session: {session_id}")

        if not (state.repl and state.repl._session_controller):
            raise ValueError(f"Session {session_id} has no session controller")

        policy = state.repl._session_controller.policy
        if "wrapup_threshold" in params:
            policy.wrapup_threshold = float(params["wrapup_threshold"])
        if "session_timer_minutes" in params:
            policy.session_timer = datetime.timedelta(minutes=int(params["session_timer_minutes"]))
        if "auto_restart" in params:
            policy.auto_restart = bool(params["auto_restart"])

        return {"status": "updated", "session_id": session_id}

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def _on_sigterm(self, _signum: int, _frame: Any) -> None:
        self._shutdown = True
        self._checkpoint_all()

    def _checkpoint_all(self) -> None:
        with self._lock:
            states = list(self.sessions.values())
        for state in states:
            if state.repl:
                try:
                    state.repl._checkpoint()
                except Exception:  # noqa: S112
                    pass


def main() -> None:
    """CLI entry point for the daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="Animus Head Session Daemon")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".animus/head/checkpoints"),
        help="Directory for session checkpoints",
    )
    parser.add_argument(
        "--model",
        default="local",
        help="Default model identifier",
    )
    args = parser.parse_args()

    daemon = HeadDaemon(checkpoint_dir=args.checkpoint_dir, model=args.model)
    daemon.run()


if __name__ == "__main__":
    main()
