# Head REPL Commands

> Reference for the Animus Head REPL — the local-first agentic conversation loop.

---

## Overview

The Head REPL is a turn-by-turn conversation loop that runs local Ollama models. It auto-loads context, executes tools, persists state to SQLite checkpoints, and supports live model swapping.

Launch:

```bash
python -m animus_kernel.head --model qwen2.5:32b --project .
```

---

## Slash Commands

### Session

| Command | Description |
|---|---|
| `exit`, `quit`, `Ctrl+D` | Leave the REPL and save checkpoint |
| `!!` | List available tools |

### Model Management

| Command | Description |
|---|---|
| `/model` | Show current model, installed models, running VRAM models, hardware profile |
| `/model <name>` | Swap to an installed model (exact match or unique prefix) |
| `/model <name> --warm` | Swap and trigger a warmup completion to preload into VRAM |
| `/model recommend` | Show hardware-aware model recommendations filtered to installed models |
| `/model stats` | Show per-model telemetry (calls, avg latency, tokens/sec, fallbacks) |
| `/model pin <name>` | Fetch Ollama digest and store it for tamper detection |
| `/model unpin <name>` | Remove a stored pin |
| `/model pins` | List all pinned models with digest verification status |
| `/hardware` | Show GPU/CPU detection results and VRAM estimate |

### Examples

```
animus > /model
  Current:   qwen2.5:32b
  Installed: qwen2.5:32b, qwen2.5:14b, phi4:14b, llama3.2
  Running:   qwen2.5:32b (VRAM)
  Hardware:  NVIDIA RTX 4090 (24 GB)

animus > /model phi4:14b
  Model swapped to: phi4:14b
   Context window: 16,384 tokens

animus > /model recommend
  Recommended models for your hardware:
  1. qwen2.5:32b  — 32K context, coding
  2. phi4:14b      — 16K context, reasoning

animus > /model stats
  Model               Calls  Avg ms   Tok/sec  Fallbacks
  ───────────────────────────────────────────────────────
  qwen2.5:32b         12     850.0    180.2    0
  phi4:14b            3      1200.0   200.0    1

animus > /model pin qwen2.5:32b
  Pinned qwen2.5:32b → sha256:abc123...

animus > /model pins
  qwen2.5:32b  sha256:abc123...  ✓ verified
```

---

## Checkpoint Persistence

The active model is saved to the SQLite checkpoint on every auto-save and graceful exit. When you restart Animus within 24 hours, the previous model is automatically restored if it is still installed.

```
🧠 Animus Head — local-first agentic loop
   Model: qwen2.5:32b
   📥 Restoring previous model: qwen2.5:32b
```

If the previously used model is no longer installed, Animus falls back to the CLI default with a warning.

---

## Context-Budget Guard

When swapping to a model with a smaller context window, Animus proactively prunes excess conversation history to fit the new limit. The pruning preserves system messages and the most recent user/assistant pairs.

```
animus > /model tiny:1b
  Model swapped to: tiny:1b
   Context window: 8,192 tokens
   ⚠️ Pruned 42 messages to fit new context window.
```

---

## Cross-Platform Hardware Detection

The Head REPL detects your hardware automatically on startup and uses it for `/model recommend`:

| Platform | Detection Method |
|---|---|
| Linux NVIDIA | `nvidia-smi` |
| Linux AMD | `rocm-smi`, fallback to `/sys/class/kfd/` sysfs |
| macOS Apple Silicon | `psutil` virtual memory (unified memory architecture) |
| Windows | `nvidia-smi.exe`, PowerShell WMI `Win32_VideoController` |
| CPU fallback | Physical core count + RAM |

See [macOS Install Guide](../getting-started/macos-install.md) for Apple Silicon specifics.
