# CUDA Setup Guide for Animus AI Box

**Target:** Ubuntu 24.04 LTS · NVIDIA GeForce RTX 30/40-series or Tesla · 24+ GB VRAM  
**Goal:** Local LLM inference for Ollama · Speed: 40–120 tokens/sec for 30B models  
**Scope:** Driver → CUDA → Docker GPU passthrough → Ollama → Animus wiring  
**Author:** ARETE · Canonical: update this doc when driver/CUDA major versions change

---

## 1. Prerequisites

```bash
# Start with a clean Ubuntu 24.04 Server or Desktop install.
# Desktop is easier initially; flip to headless after validating.

uname -a                    # Should be 6.8+ kernel (24.04 default)
lspci | grep -i nvidia      # Should list your GPU(s)
ubuntu-drivers devices      # Lists available proprietary drivers
```

**Check your GPU compute capability** (determines CUDA features):

| GPU | Compute | Max CUDA |
|---|---|---|
| RTX 3090/3090 Ti | 8.6 | 12.x |
| RTX 4090 | 8.9 | 12.x |
| RTX 4070 Ti Super | 8.9 | 12.x |
| A100 | 8.0 | 12.x |
| H100 | 9.0 | 12.x |
| RTX 3060 | 8.6 | 12.x |

All RTX 30/40-series and newer are fine for LLM inference with CUDA 12.x.

---

## 2. NVIDIA Driver (Step Zero — Do This First)

**Use the Ubuntu-packaged driver.** The runfile from nvidia.com is a trap — it breaks DKMS on kernel updates.

```bash
# Method A: Ubuntu's recommended (easiest)
sudo ubuntu-drivers autoinstall

# Method B: Explicit version (pick a 550 or 560 series driver)
sudo apt update
sudo apt install -y linux-headers-$(uname -r)
sudo apt install -y nvidia-driver-560  # or 550 for older stability
```

**Reboot and verify:**

```bash
sudo reboot
# After reboot
nvidia-smi
```

You should see GPU name, temperature, VRAM, driver version, and CUDA version (driver-bundled). If `nvidia-smi` fails, stop here. Nothing below works without this.

**Common failure:** Secure Boot enabled. Ubuntu's packaged driver signs the kernel module via `mokutil`, but if Secure Boot is active and you skipped the MOK enrollment step, the driver won't load.

```bash
# Fix: either enroll the MOK or disable Secure Boot
sudo mokutil --sb-state    # Check state
# If enabled and driver fails: reboot → UEFI setup → Secure Boot OFF
```

---

## 3. CUDA Toolkit (Minimal Install)

**You probably DO NOT need the full CUDA toolkit.** Ollama ships its own CUDA runtime via llama.cpp's cuBLAS bindings. The only reason to install system CUDA is if you're:

- Compiling Python packages with CUDA extensions from source
- Running vLLM (needs PyTorch with CUDA)
- Custom-building llama.cpp with specific CUDA flags

**If Ollama is your only inference engine, skip to Section 4.** Ollama uses the driver-bundled CUDA libs.

**If you want system CUDA (e.g., for vLLM or custom builds):**

```bash
# Use the apt repo (NOT the runfile installer)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

**Verify:**

```bash
/usr/local/cuda/bin/nvcc --version  # Should print 12.4.x
```

**Add to PATH (add to `~/.bashrc`):**

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 4. Docker with GPU Passthrough

**This is the cleanest way to run Ollama + Animus.** The host stays clean. Everything runs in containers with explicit GPU access.

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit (this is the magic)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify Docker can see the GPU:**

```bash
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu24.04 nvidia-smi
```

You should see the same output as host `nvidia-smi`.

---

## 5. Ollama with GPU Acceleration

### Option A: Native (simpler for a dedicated box)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

Ollama detects CUDA automatically. If your GPU is visible in `nvidia-smi`, Ollama will use it.

```bash
# Pull a model
ollama pull qwen3:32b       # or deepseek-coder-v2, llama3.3:70b
ollama pull nomic-embed-text

# Run and observe GPU utilization
ollama run qwen3:32b
# In another terminal, watch:
watch -n 1 nvidia-smi
```

**Expected:** `nvidia-smi` shows `ollama` process consuming GPU compute and VRAM. CPU usage stays low. If GPU is at 0% and CPU is pegged, CUDA isn't wired.

### Option B: Docker (recommended for reproducibility)

```bash
# docker-compose.yml for the AI box
mkdir ~/ai-box && cd ~/ai-box
```

```yaml
# ~/ai-box/docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - OLLAMA_KEEP_ALIVE=24h        # Keep model in VRAM for 24h after last use
      - OLLAMA_NUM_GPU=99            # Offload as many layers as fit

  animus-bootstrap:
    image: animus:bootstrap          # You'd build this from Dockerfile
    container_name: animus
    restart: unless-stopped
    ports:
      - "7700:7700"
    volumes:
      - animus_data:/data
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - ANIMUS_OFFLINE=1
      - ANIMUS_MODEL_PROVIDER=ollama
      - ANIMUS_MODEL_NAME=qwen3:32b
    depends_on:
      - ollama

volumes:
  ollama_data:
  animus_data:
```

```bash
docker compose up -d ollama
sleep 5
docker exec -it ollama ollama pull qwen3:32b
docker exec -it ollama ollama pull nomic-embed-text
# Animus later:
# docker compose up -d animus-bootstrap
```

**Key env vars:**

| Var | Default | Effect |
|---|---|---|
| `OLLAMA_KEEP_ALIVE` | 5m | How long to keep model in VRAM after idle. Longer = faster re-use, more VRAM held. |
| `OLLAMA_NUM_GPU` | auto | Layers to offload to GPU. `99` = offload everything that fits. |
| `OLLAMA_MAX_LOADED_MODELS` | 1 | VRAM permitting, keep multiple models resident. |

---

## 6. Animus Wiring for GPU Ollama

Your goal is **offline-default, cloud-gated.** Here are the config changes:

### 6.1 Environment defaults

Add to `~/.profile` (or Docker `environment:` block):

```bash
# Force offline mode — must explicitly opt in to cloud per-session
export ANIMUS_OFFLINE=1

# Point Animus at your local Ollama
export ANIMUS_OLLAMA_URL=http://localhost:11434
export ANIMUS_MODEL_PROVIDER=ollama
export ANIMUS_MODEL_NAME=qwen3:32b

# If using the Docker compose above, these are already wired
# export OLLAMA_HOST=http://ollama:11434
```

### 6.2 HybridBackend tuning (Core `cognitive.py`)

Currently the `HybridBackend` routes `HEAVY` tasks to Anthropic and `LIGHT` to Ollama. For local-first, **invert the default assumption:**

```python
# In packages/core/animus/cognitive.py or config override

# OLD: default heavy → Anthropic
# NEW: default everything → Ollama; heavy/explicit → cloud with gate

class HybridBackend:
    """Routes to Ollama by default. Cloud is the exception, gated."""

    def route(self, prompt: str, mode: ReasoningMode) -> str:
        # Offline gate: if ANIMUS_OFFLINE=1, never reach cloud
        if os.environ.get("ANIMUS_OFFLINE") == "1":
            return "ollama"

        # Heavy tasks can opt into cloud IF user explicitly allows
        if classify_task(prompt) == TaskWeight.HEAVY:
            # Check for an explicit "--cloud" flag or approval token
            if not getattr(self, "_cloud_approved", False):
                logger.info("Heavy task routed to Ollama (local-only mode)")
                return "ollama"
            return "anthropic"

        return "ollama"
```

**Practical impact:** With Qwen3 32B at Q4_K_M, ~80% of your daily tasks (summaries, formatting, tagging, code review, simple tool use) run entirely locally. Only frontier-grade reasoning (architecture design, eval judging) would prompt for cloud.

### 6.3 Bootstrap wizard first-run for local-first

Modify `packages/bootstrap/src/animus_bootstrap/setup/steps/api_keys.py` to **default to Ollama** and make API keys truly optional:

```python
# In the wizard — change from "API Keys (required)" to "API Keys (optional)"
# If user skips Anthropic key, set provider=ollama and model=qwen3:32b
```

---

## 7. Benchmarking Your Box

**Before trusting local models for Animus, measure.**

```bash
#!/bin/bash
# bench_models.sh

MODELS=("qwen3:32b" "deepseek-coder-v2" "llama3.3:70b" "qwen2.5:14b")
PROMPT="Explain the concept of recursion in programming."

for model in "${MODELS[@]}"; do
    echo "=== $model ==="
    # Check if model exists, pull if not
    docker exec ollama ollama list | grep -q "$model" || docker exec ollama ollama pull "$model"

    # Warm-up run
    curl -s http://localhost:11434/api/generate \
      -d "{\"model\":\"$model\",\"prompt\":\"$PROMPT\",\"stream\":false}" > /dev/null

    # Timed run
    time curl -s http://localhost:11434/api/generate \
      -d "{\"model\":\"$model\",\"prompt\":\"$PROMPT\",\"stream\":false}" \
      | jq '.eval_count, .eval_duration'
    echo ""
done
```

**Interpret results:**

| Tokens/sec | Verdict |
|---|---|
| 0–5 | Wrong. Model running on CPU. Check CUDA wiring. |
| 5–20 | Okay for async/batch. Too slow for interactive chat. |
| 20–50 | Usable for interactive. Summaries, taggings, light chat. |
| 50–100 | Good. Comfortable daily driver for most tasks. |
| 100+ | Excellent. Comparable to small cloud models. |

For Qwen3 32B on RTX 4090, expect **40–70 tok/s** at Q4_K_M.

---

## 8. Tailscale Remote Access (Headless Setup)

If the box lives in a closet:

```bash
# On the AI box
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh

# From your laptop
ssh username@ai-box    # Or use Tailscale IP for dashboard
open http://ai-box:7700  # PWA/dashboard
```

**Expose Ollama securely:**

```bash
# Ollama should NOT be exposed to 0.0.0.0 without auth.
# Use Tailscale's SOCKS5 proxy instead:
tailscale serve --bg --set-path=/ollama http://localhost:11434
# Access from laptop: http://ai-box/ollama (Tailscale auth)
```

---

## 9. Maintenance

### Kernel update safety

Ubuntu updates the kernel periodically. DKMS should auto-rebuild the NVIDIA module. Sometimes it fails.

```bash
# After any kernel update:
sudo apt update && sudo apt upgrade
sudo reboot
nvidia-smi  # Verify
```

If `nvidia-smi` fails post-update:

```bash
sudo dkms status | grep nvidia    # Check build status
sudo dkms autoinstall             # Force rebuild
sudo modprobe nvidia              # Load module
```

### Ollama update

```bash
docker compose pull ollama
docker compose up -d ollama
# Or native:
curl -fsSL https://ollama.com/install.sh | sh
```

### VRAM management

```bash
# See what's loaded
curl http://localhost:11434/api/ps

# Unload all models to free VRAM
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen3:32b","keep_alive":0}'
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi` → "NVIDIA-SMI has failed" | Driver not loaded | Reinstall driver. Check Secure Boot. |
| `nvidia-smi` works but `docker --gpus` fails | nvidia-container-toolkit not configured | Re-run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Ollama runs but uses 0% GPU, 100% CPU | Model offloaded to CPU | Check VRAM — model too big? Quantize. Check `OLLAMA_NUM_GPU`. |
| Intermittent CUDA out-of-memory | Multiple models loaded | Set `OLLAMA_MAX_LOADED_MODELS=1` or unload unused models. |
| Slow after idle period | Cold start from system RAM | Increase `OLLAMA_KEEP_ALIVE`. |
| Animus can't reach Ollama | Wrong host/port | Verify `ANIMUS_OLLAMA_URL`. If Docker, use service name `ollama:11434`. |
| Build failures on `pip install` with CUDA | Missing nvcc | Install `cuda-toolkit-12-4` or use pre-built wheels. |

---

## 11. Cost Comparison (Why This Matters)

| Scenario | 1 Year Cost | Notes |
|---|---|---|
| Cloud API (Claude Opus tier) | $2,000–5,000 | Depends on volume. At 30-project scale, this is realistic. |
| RTX 4090 build (one-time) | $2,500–3,500 | No recurring inference cost. Electricity: ~$15/mo. |
| Used RTX 3090 build | $1,500–2,000 | Same VRAM as 4090, slower, more power. Still fine. |
| Cloud GPU rental (continuous) | $3,000–6,000 | RunPod/Vast.ai. Flexible but defeats sovereignty. |

**Break-even:** ~100,000 heavy queries or ~500,000 light queries. At your usage rate, 2–3 years. The real win is not price — it's **no dependency**, **no rate limits**, **no key rotations**, **no network required**.

---

## 12. Quick-Start Checklist

```bash
# Day 1: Bare box → running inference
□ Ubuntu 24.04 installed
□ sudo apt update && sudo apt upgrade
□ sudo ubuntu-drivers autoinstall && sudo reboot
□ nvidia-smi passes
□ curl -fsSL https://ollama.com/install.sh | sh
□ ollama pull qwen3:32b
□ ollama run qwen3:32b  # Verify GPU % in nvidia-smi
□ curl -fsSL https://get.docker.com | sh
□ nvidia-container-toolkit installed + docker restarted
□ docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu24.04 nvidia-smi
□ docker compose up -d ollama
□ curl http://localhost:11434/api/tags  # Lists pulled models
□ export ANIMUS_OFFLINE=1
□ animus-bootstrap install --skip-wizard  # If not yet installed
□ python -m animus  # Talk to your local model
```

---

## Related

- `docs/PERSONAL_ROADMAP.md` Track 2 — cost-efficiency audit
- `docs/INTERFACE_BOOTSTRAP_VISION.md` Phase IV — ambient presence on the AI box
- `docs/ROADMAP_TO_10.md` Session 7 — supply-chain hardening (pin model digests)
- Ollama docs: https://github.com/ollama/ollama/blob/main/docs/

---

*Canonical. Update when driver/CUDA major versions change or when new GPU architectures ship.*
