# macOS Installation

> Install and run Animus on macOS (Apple Silicon or Intel). The core system is fully cross-platform; only the systemd-based autonomous daemon is Linux-only.

---

## Prerequisites

| Component | Version | macOS Notes |
|---|---|---|
| macOS | ≥12 (Monterey) | Apple Silicon (M1–M4) or Intel |
| Python | ≥3.11 | Install via [Homebrew](https://brew.sh) or [python.org](https://python.org) |
| Ollama | latest | [Native macOS app](https://ollama.com/download/mac) |
| psutil | *optional* | `pip install psutil` — enables cross-platform resource monitoring |

### Install Ollama

```bash
# Via Homebrew (recommended)
brew install --cask ollama

# Or download from https://ollama.com/download/mac
```

Ollama on macOS automatically uses **Metal (Apple Silicon)** or **Intel UHD/AMD graphics** for GPU acceleration when available.

---

## Step-by-Step Install

### 1. Clone the repo

```bash
git clone https://github.com/your-org/animus.git
cd animus
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Install shared types FIRST
pip install -e packages/types/

# Install the kernel (includes Head REPL)
pip install -e packages/kernel/

# Optional: install core for full operating environment features
pip install -e packages/core/

# Optional: install bootstrap for daemon + dashboard
pip install -e packages/bootstrap/
```

### 4. Verify

```bash
python -c "import animus_kernel; print('Kernel OK')"
python -m animus_kernel.head --help
```

---

## Running Animus Head (REPL)

```bash
# Start the local-first agentic REPL
python -m animus_kernel.head --model qwen2.5:14b --project .

# List available models
animus > /model

# Swap models mid-session
animus > /model llama3.2

# Get hardware-aware recommendations
animus > /model recommend
```

The Head REPL is fully functional on macOS, including:
- Model swap (`/model <name>`)
- Hardware detection (`/model` shows Apple Silicon unified memory)
- Performance telemetry (`/model stats`)
- Warm-swap (`/model <name> --warm`)
- Session persistence (SQLite checkpoints)

---

## Running the Daemon (Background Service)

Unlike Linux (which uses systemd), macOS uses **launchd** for background services. Animus includes a built-in launchd agent generator.

### Generate and install the launchd plist

```bash
# Activate your venv
source .venv/bin/activate

# Generate the plist
python -c "
from animus_bootstrap.daemon.platforms.macos import MacOSService
svc = MacOSService()
svc.install_plist()
"
```

This creates `~/Library/LaunchAgents/dev.animus.plist`.

### Load (start) the daemon

```bash
launchctl load ~/Library/LaunchAgents/dev.animus.plist
```

### Check status

```bash
launchctl list dev.animus
```

### Stop the daemon

```bash
launchctl unload ~/Library/LaunchAgents/dev.animus.plist
```

### View logs

```bash
tail -f ~/Library/Logs/Animus/animus.out.log
tail -f ~/Library/Logs/Animus/animus.err.log
```

---

## macOS-Specific Notes

### Apple Silicon Unified Memory

On Apple Silicon (M1/M2/M3/M4), `detect_hardware()` treats total RAM as available VRAM because the GPU shares unified memory with the CPU. This means:

- A Mac mini with **16 GB RAM** can run **14B-parameter models** comfortably
- A Mac Studio with **32 GB RAM** can run **32B-parameter models**
- A Mac Pro with **64+ GB RAM** can run **70B-parameter models**

The hardware profiler automatically detects `Darwin + arm64` and maps memory to the appropriate model tier.

### Intel Macs

Intel Macs without discrete GPUs fall back to **CPU-only** recommendations. You can still run small models (3B–7B) but inference will be slower than Apple Silicon.

### Missing Linux-Only Features

The following features are **not available on macOS**:

| Feature | Linux Equivalent | macOS Status |
|---|---|---|
| Encrypted vault (`gocryptfs`) | `setup-gocryptfs-vault.sh` | Not available — macOS has FileVault |
| systemd timers | `systemd/` directory | Not needed — use `launchd` or `cron` |
| TPM-sealed credentials | `systemd-creds` | Not available |
| `/proc` monitoring | `monitoring/watchers.py` | Falls back to `psutil` |

---

## Troubleshooting

### "Ollama is not running"

Ensure Ollama.app is running in the menu bar:
```bash
# Or start via CLI
/Applications/Ollama.app/Contents/MacOS/Ollama serve
```

### Permission denied on `~/.animus`

```bash
mkdir -p ~/.animus/sessions ~/.animus/memory
chmod 755 ~/.animus
```

### Slow inference on Intel Mac

Intel Macs lack unified memory. Try smaller models:
```bash
python -m animus_kernel.head --model qwen2.5:3b --project .
```

---

## See Also

- [Installation](installation.md) — Generic install guide
- [Quickstart](quickstart.md) — Get running in 10 minutes
- [Operators → Configuration](../operators/configuration.md) — Configure after install
