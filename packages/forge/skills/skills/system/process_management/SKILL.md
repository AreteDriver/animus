---
name: process_management
version: 1.0.0
agent: system
risk_level: variable
description: "Monitor, control, and manage system processes. Includes starting, stopping, and monitoring services, background jobs, and system health metrics."
---

# Process Management Skill

## Purpose

Monitor and control system processes on the Gorgon host machine. This skill enables Gorgon agents to launch applications, manage services, monitor system health, and handle background jobs.

## Safety Rules

### PROTECTED PROCESSES - NEVER KILL
```
init (PID 1)
systemd
sshd
Gorgon supervisor (self)
kernel threads [kthreadd, kworker, etc.]
```

### MANDATORY PRACTICES
1. **Never kill processes by pattern without verification** - always list first
2. **Prefer graceful shutdown (SIGTERM) before force kill (SIGKILL)**
3. **Wait for confirmation before killing processes with open files/network connections**
4. **Log all process terminations with reason**
5. **Never modify OOM killer settings on system processes**

### CONSENSUS REQUIREMENTS
| Operation | Risk Level | Consensus Required |
|-----------|------------|-------------------|
| list_processes | low | any |
| get_process_info | low | any |
| monitor_resources | low | any |
| start_process | medium | majority |
| stop_process (graceful) | medium | majority |
| kill_process (force) | high | unanimous |
| manage_service | high | unanimous |
| set_priority | medium | majority |

## Capabilities

### list_processes
List running processes with optional filtering.

**Usage:**
```bash
# All processes
ps aux

# Process tree
pstree -p

# By name
pgrep -la python

# By user
ps -u gorgon

# Top CPU consumers
ps aux --sort=-%cpu | head -20

# Top memory consumers
ps aux --sort=-%mem | head -20

# With full command line
ps aux --width 200
```

---

### get_process_info
Get detailed information about a specific process.

**Usage:**
```bash
# Basic info by PID
ps -p 1234 -o pid,ppid,user,%cpu,%mem,etime,comm

# Full details
cat /proc/1234/status

# Open files
lsof -p 1234

# Network connections
ss -tulpn | grep 1234

# Memory map
pmap 1234

# Environment variables
cat /proc/1234/environ | tr '\0' '\n'

# Current working directory
ls -la /proc/1234/cwd

# File descriptors
ls -la /proc/1234/fd
```

---

### monitor_resources
Get system resource utilization.

**Usage:**
```bash
# Overall system stats
vmstat 1 5

# CPU per core
mpstat -P ALL 1 3

# Memory detailed
free -h
cat /proc/meminfo

# Disk I/O
iostat -x 1 3

# Network I/O
sar -n DEV 1 3

# Combined snapshot (for Gorgon metrics)
echo "=== CPU ===" && top -bn1 | head -5
echo "=== MEMORY ===" && free -h
echo "=== DISK ===" && df -h
echo "=== LOAD ===" && uptime
```

**Python helper for structured output:**
```python
import psutil
import json

def get_system_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_per_core": psutil.cpu_percent(interval=1, percpu=True),
        "memory": dict(psutil.virtual_memory()._asdict()),
        "swap": dict(psutil.swap_memory()._asdict()),
        "disk": {p.mountpoint: dict(psutil.disk_usage(p.mountpoint)._asdict()) 
                 for p in psutil.disk_partitions()},
        "network": dict(psutil.net_io_counters()._asdict()),
        "load_average": psutil.getloadavg(),
        "boot_time": psutil.boot_time(),
    }

print(json.dumps(get_system_metrics(), indent=2))
```

---

### start_process
Launch a new process.

**Usage:**
```bash
# Simple foreground
python3 /path/to/script.py

# Background with nohup
nohup python3 /path/to/script.py > /var/log/gorgon/script.log 2>&1 &
echo $! > /var/run/gorgon/script.pid

# With resource limits
systemd-run --user --scope \
  -p MemoryMax=512M \
  -p CPUQuota=50% \
  python3 /path/to/script.py

# With environment variables
env VAR1=value1 VAR2=value2 python3 /path/to/script.py

# With specific working directory
cd /path/to/workdir && python3 script.py

# Detached with logging
setsid python3 /path/to/script.py > /var/log/gorgon/script.log 2>&1 &
```

**Safety:**
- Always capture PID for tracking
- Redirect output to logs
- Set resource limits for untrusted scripts
- Verify executable exists and is permitted

---

### stop_process
Gracefully stop a process (SIGTERM).

**Usage:**
```bash
# By PID
kill -TERM 1234

# Wait for termination (timeout 30s)
timeout 30 tail --pid=1234 -f /dev/null

# Check if still running
ps -p 1234 > /dev/null && echo "Still running" || echo "Stopped"

# By name (careful - lists first!)
pgrep -la myprocess  # List first!
pkill -TERM myprocess

# By PID file
kill -TERM $(cat /var/run/gorgon/myprocess.pid)
rm /var/run/gorgon/myprocess.pid
```

**Safety:**
- Always try SIGTERM first
- Wait reasonable time before escalating
- Check for child processes
- Clean up PID files

---

### kill_process
Force kill a process (SIGKILL).

**Usage:**
```bash
# By PID - REQUIRES UNANIMOUS CONSENSUS
kill -9 1234

# Verify termination
sleep 1 && ps -p 1234 > /dev/null && echo "FAILED TO KILL" || echo "Killed"

# Kill process group
kill -9 -$(ps -o pgid= -p 1234 | tr -d ' ')
```

**Safety:**
- REQUIRES UNANIMOUS CONSENSUS
- Only use after graceful stop fails
- Log reason for force kill
- Check for orphaned child processes
- May cause data loss - warn user

---

### manage_service
Control systemd services.

**Usage:**
```bash
# Status
systemctl status myservice

# Start
sudo systemctl start myservice

# Stop
sudo systemctl stop myservice

# Restart
sudo systemctl restart myservice

# Enable at boot
sudo systemctl enable myservice

# Disable at boot
sudo systemctl disable myservice

# Check if active
systemctl is-active myservice

# View logs
journalctl -u myservice -n 50 --no-pager

# User services (no sudo)
systemctl --user status myservice
systemctl --user start myservice
```

**Safety:**
- REQUIRES UNANIMOUS CONSENSUS
- Never disable critical services (sshd, systemd, networking)
- Check dependencies before stopping
- Verify service file exists

---

### set_priority
Change process priority (nice/ionice).

**Usage:**
```bash
# CPU priority (nice: -20 highest to 19 lowest)
renice 10 -p 1234

# Start process with low priority
nice -n 15 python3 script.py

# I/O priority
ionice -c 3 -p 1234  # Idle class

# Start with I/O priority
ionice -c 2 -n 7 python3 script.py  # Best-effort, low priority

# Combined
nice -n 10 ionice -c 2 -n 7 python3 script.py
```

**Safety:**
- Only increase priority (lower number) with unanimous consensus
- Never set negative nice values without explicit approval

---

### background_job_control
Manage background jobs in current session.

**Usage:**
```bash
# List background jobs
jobs -l

# Send to background
# (from running process: Ctrl+Z then)
bg %1

# Bring to foreground
fg %1

# Disown (survives shell exit)
disown %1
```

---

## Examples

### Example 1: Find and stop a runaway process
**Intent:** "Something is using 100% CPU, find and stop it"

**Execution:**
```bash
# Identify the process
ps aux --sort=-%cpu | head -10

# Output shows PID 5678 at 98% CPU
# Get more info
ps -p 5678 -o pid,ppid,user,%cpu,%mem,etime,comm,args

# Check what files it has open
lsof -p 5678 | head -20

# Try graceful stop first (REQUIRES MAJORITY CONSENSUS)
kill -TERM 5678

# Wait up to 30 seconds
for i in {1..30}; do
  ps -p 5678 > /dev/null 2>&1 || break
  sleep 1
done

# Check if stopped
ps -p 5678 > /dev/null 2>&1 && echo "Still running - may need force kill" || echo "Successfully stopped"
```

---

### Example 2: Launch a monitored background task
**Intent:** "Run the data sync script in background with resource limits"

**Execution:**
```bash
# Create log directory if needed
mkdir -p /var/log/gorgon

# Launch with limits and logging
systemd-run --user --scope \
  --unit=gorgon-datasync \
  -p MemoryMax=1G \
  -p CPUQuota=50% \
  bash -c 'python3 /home/gorgon/scripts/datasync.py >> /var/log/gorgon/datasync.log 2>&1'

# Get the PID
systemctl --user show gorgon-datasync --property=MainPID

# Store for tracking
systemctl --user show gorgon-datasync --property=MainPID --value > /var/run/gorgon/datasync.pid

# Monitor
systemctl --user status gorgon-datasync
```

---

### Example 3: System health check for Gorgon metrics
**Intent:** "Get current system metrics for supervisor decision making"

**Execution:**
```bash
python3 << 'EOF'
import psutil
import json
from datetime import datetime

metrics = {
    "timestamp": datetime.now().isoformat(),
    "cpu": {
        "percent": psutil.cpu_percent(interval=1),
        "count": psutil.cpu_count(),
        "count_physical": psutil.cpu_count(logical=False),
        "load_avg": psutil.getloadavg()
    },
    "memory": {
        "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "percent": psutil.virtual_memory().percent
    },
    "disk": {
        "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
        "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
        "percent": psutil.disk_usage('/').percent
    },
    "processes": {
        "total": len(psutil.pids()),
        "running": len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'running'])
    }
}

print(json.dumps(metrics, indent=2))
EOF
```

---

## Integration with Gorgon Supervisor

The process_management skill provides critical data for the Triumvirate's resource governance:

```python
# Supervisor calls this periodically
def collect_metrics_for_triumvirate():
    """Collect metrics for supervisor decision making."""
    
    # Uses monitor_resources capability
    metrics = execute_skill("process_management", "monitor_resources")
    
    # Feed to triumvirate for scaling decisions
    triumvirate.evaluate_resources(metrics)
```

## Error Handling

| Error | Response |
|-------|----------|
| Process not found | Report, verify PID was correct |
| Permission denied | Report, check if sudo required |
| Process is protected | Refuse operation, explain why |
| Kill failed | Report, may be zombie - check parent |
| Service not found | List available services, suggest alternatives |

## Output Format

```json
{
  "success": true,
  "operation": "stop_process",
  "pid": 1234,
  "details": {
    "signal": "SIGTERM",
    "wait_time_seconds": 3.2,
    "clean_shutdown": true
  },
  "timestamp": "2026-01-27T10:30:00Z"
}
```
