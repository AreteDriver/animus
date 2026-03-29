#!/usr/bin/env python3
"""Fleet Remediation Runner — processes pending fleet alerts from fleet-monitor.

Reads alert context files from ~/.animus/fleet_alerts/, determines the
appropriate remediation workflow, and executes it via shell commands
(flyctl restart, health checks, log pulls).

Designed to run as:
  - A cron job (every 5 minutes)
  - A one-shot command: python tools/fleet_remediate.py
  - Called by the Animus proactive engine

Does NOT require Forge — uses direct shell commands for reliability.
Forge workflows are available for deeper triage when simple remediation fails.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fleet_remediate")

ANIMUS_DIR = Path.home() / ".animus"
ALERTS_DIR = ANIMUS_DIR / "fleet_alerts"
TASKS_FILE = ANIMUS_DIR / "tasks.json"
RESULTS_DIR = ANIMUS_DIR / "fleet_results"
FLYCTL = "/home/arete/.fly/bin/flyctl"

# Services where auto-restart is safe
SAFE_RESTART_APPS = {
    "benchgoblins-backend",
    "aegismonolith",
    "watchtower-evefrontier",
    "frontier-tribe-os",
    "aicards-mint",
    "staycards-mint",
    "cmdf-license",
}

# Track what we've already attempted (prevent restart loops)
_attempted: dict[str, float] = {}
ATTEMPT_COOLDOWN = 600  # 10 minutes between restart attempts per service


def load_pending_alerts() -> list[dict]:
    """Load all pending alert context files."""
    if not ALERTS_DIR.exists():
        return []

    alerts = []
    for f in ALERTS_DIR.glob("*.json"):
        if f.name == "active_alerts.json":
            continue
        try:
            data = json.loads(f.read_text())
            data["_file"] = str(f)
            alerts.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", f, e)

    return sorted(alerts, key=lambda a: a.get("timestamp", ""), reverse=True)


def run_cmd(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run a shell command and return (returncode, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout}s"
    except FileNotFoundError:
        return -1, f"Command not found: {cmd[0]}"


def check_health(url: str, timeout: int = 15) -> tuple[bool, int, str]:
    """Check a health endpoint. Returns (is_healthy, http_code, details)."""
    code, output = run_cmd(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(timeout), url],
        timeout=timeout + 5,
    )
    try:
        http_code = int(output.strip())
    except ValueError:
        return False, 0, output

    return http_code == 200, http_code, output


def attempt_fly_restart(fly_app: str) -> tuple[bool, str]:
    """Attempt to restart a Fly.io app. Returns (success, details)."""
    if fly_app not in SAFE_RESTART_APPS:
        return False, f"App {fly_app} not in safe restart list"

    # Check cooldown
    last_attempt = _attempted.get(fly_app, 0)
    if time.time() - last_attempt < ATTEMPT_COOLDOWN:
        remaining = int(ATTEMPT_COOLDOWN - (time.time() - last_attempt))
        return False, f"Cooldown active — {remaining}s remaining since last attempt"

    _attempted[fly_app] = time.time()

    logger.info("Restarting Fly.io app: %s", fly_app)

    # Get machine IDs first, then restart each one
    list_code, list_output = run_cmd(
        [FLYCTL, "machine", "list", "-a", fly_app, "--json"], timeout=15
    )
    if list_code != 0:
        # Fallback: try app-level restart via deploy restart
        code, output = run_cmd([FLYCTL, "apps", "restart", fly_app], timeout=60)
        if code == 0:
            return True, f"App restart succeeded: {output}"
        return False, f"Could not list machines or restart app: {output}"

    import json as _json
    try:
        machines = _json.loads(list_output)
        machine_ids = [m["id"] for m in machines if m.get("state") in ("started", "stopped", "replacing")]
    except (ValueError, KeyError):
        machine_ids = []

    if not machine_ids:
        code, output = run_cmd([FLYCTL, "apps", "restart", fly_app], timeout=60)
        if code == 0:
            return True, f"App restart (no machines found): {output}"
        return False, f"No machines found and app restart failed: {output}"

    results = []
    for mid in machine_ids:
        code, output = run_cmd([FLYCTL, "machine", "restart", mid, "-a", fly_app], timeout=60)
        results.append(f"machine {mid}: {'ok' if code == 0 else 'failed'}")

    all_ok = all("ok" in r for r in results)
    return all_ok, "; ".join(results)


def get_fly_logs(fly_app: str, lines: int = 20) -> str:
    """Pull recent logs from a Fly.io app."""
    _, output = run_cmd([FLYCTL, "logs", "-a", fly_app, "--no-tail"], timeout=15)
    log_lines = output.strip().split("\n")
    return "\n".join(log_lines[-lines:])


def remediate_alert(alert: dict) -> dict:
    """Process a single alert and attempt remediation.

    Returns a result dict with actions taken and outcomes.
    """
    service = alert.get("service_name", "unknown")
    fly_app = alert.get("fly_app")
    hint = alert.get("remediation_hint", "monitor")
    task_id = alert.get("task_id", "")
    status = alert.get("status", "unknown")

    result = {
        "task_id": task_id,
        "service": service,
        "fly_app": fly_app,
        "hint": hint,
        "actions": [],
        "recovered": False,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info("Processing alert for %s (hint=%s, status=%s)", service, hint, status)

    # Step 1: Re-check health — maybe it already recovered
    health_url = alert.get("health_url", "")
    if not health_url:
        # Reconstruct from service config if missing
        from fleet_monitor_urls import SERVICE_URLS
        health_url = SERVICE_URLS.get(service, "")

    if health_url:
        is_healthy, http_code, _ = check_health(health_url)
        result["actions"].append(f"Health recheck: HTTP {http_code}")

        if is_healthy:
            result["recovered"] = True
            result["actions"].append("Service already recovered — no action needed")
            logger.info("%s already recovered (HTTP 200)", service)
            return result

    # Step 2: Attempt remediation based on hint
    if hint == "fly_restart" and fly_app:
        success, details = attempt_fly_restart(fly_app)
        result["actions"].append(f"Fly restart: {'SUCCESS' if success else 'FAILED'} — {details}")

        if success:
            # Wait and verify
            time.sleep(15)
            if health_url:
                is_healthy, http_code, _ = check_health(health_url)
                result["actions"].append(f"Post-restart health check: HTTP {http_code}")
                result["recovered"] = is_healthy

    elif hint == "check_logs" and fly_app:
        logs = get_fly_logs(fly_app)
        result["actions"].append(f"Retrieved logs:\n{logs}")
        # Don't auto-restart on 500s — could be a code bug
        result["actions"].append("500 error detected — needs manual investigation (possible code bug)")

    elif hint == "check_database":
        result["actions"].append("Database issue detected — checking if Fly Postgres is responsive")
        if fly_app:
            _, pg_output = run_cmd(
                [FLYCTL, "postgres", "connect", "-a", f"{fly_app}-db", "-c", "SELECT 1"],
                timeout=10,
            )
            result["actions"].append(f"Postgres check: {pg_output}")

    elif hint == "check_external_dependency":
        result["actions"].append(
            "External dependency issue (e.g., Sui GraphQL) — cannot auto-remediate, monitoring"
        )

    elif hint == "check_connectivity":
        result["actions"].append("Connectivity timeout — checking from this host")
        if health_url:
            _, trace = run_cmd(["curl", "-v", "--max-time", "10", health_url], timeout=15)
            result["actions"].append(f"Verbose check:\n{trace[:500]}")

    else:
        result["actions"].append(f"Hint '{hint}' — monitoring only, no auto-remediation")

    # Step 3: If still not recovered and we have a Fly app, grab logs for context
    if not result["recovered"] and fly_app:
        logs = get_fly_logs(fly_app, lines=10)
        result["actions"].append(f"Diagnostic logs:\n{logs}")

    return result


def update_task_with_result(task_id: str, result: dict) -> None:
    """Update the Animus task with remediation results."""
    if not TASKS_FILE.exists():
        return

    try:
        tasks = json.loads(TASKS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return

    if task_id not in tasks:
        return

    now = datetime.now().isoformat()
    actions_summary = "\n".join(f"  - {a}" for a in result["actions"])
    note = f"[{now}] Remediation run:\n{actions_summary}"

    if result["recovered"]:
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["completed_at"] = now
        note += "\n  RESULT: RECOVERED"

    if tasks[task_id]["notes"]:
        tasks[task_id]["notes"] += f"\n{note}"
    else:
        tasks[task_id]["notes"] = note

    tasks[task_id]["updated_at"] = now
    TASKS_FILE.write_text(json.dumps(tasks, indent=2))


def save_result(result: dict) -> None:
    """Save remediation result for audit trail."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = RESULTS_DIR / f"{result['service']}_{ts}.json"
    result_file.write_text(json.dumps(result, indent=2))


def main() -> None:
    alerts = load_pending_alerts()
    if not alerts:
        logger.info("No pending fleet alerts")
        return

    logger.info("Found %d pending alert(s)", len(alerts))

    for alert in alerts:
        result = remediate_alert(alert)
        save_result(result)

        task_id = result.get("task_id", "")
        if task_id:
            update_task_with_result(task_id, result)

        if result["recovered"]:
            # Clean up the alert context file
            alert_file = Path(alert.get("_file", ""))
            if alert_file.exists():
                alert_file.unlink()
            logger.info("%s: RECOVERED", result["service"])
        else:
            logger.warning("%s: still unhealthy after remediation", result["service"])

    logger.info("Remediation complete: %d alert(s) processed", len(alerts))


# Health URL fallback (when alert context doesn't include it)
# This avoids importing fleet_monitor which may not be installed
class _ServiceURLs(dict):
    """Lazy URL map that doesn't crash if fleet_monitor isn't available."""

    _URLS = {
        "benchgoblins_api": "https://backend.benchgoblins.com/health",
        "benchgoblins_web": "https://benchgoblins.com",
        "gatekeeper_api": "https://edengk.com/api/health",
        "gatekeeper_web": "https://edengk.com",
        "monolith_api": "https://aegismonolith.fly.dev/api/v1/health",
        "witness_api": "https://watchtower-evefrontier.fly.dev/health",
        "frontier_tribe_os": "https://frontier-tribe-os.fly.dev/health",
        "aicards_mint": "https://aicards-mint.fly.dev/health",
        "staycards_mint": "https://staycards-mint.fly.dev/health",
        "license_server": "https://cmdf-license.fly.dev/v1/health",
    }

    def get(self, key, default=None):
        return self._URLS.get(key, default)


# Module-level stub so the remediate function can reference it
# without importing fleet_monitor
sys.modules["fleet_monitor_urls"] = type(sys)("fleet_monitor_urls")
sys.modules["fleet_monitor_urls"].SERVICE_URLS = _ServiceURLs()


if __name__ == "__main__":
    main()
