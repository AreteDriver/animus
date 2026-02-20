#!/usr/bin/env bash
cd ~/projects/animus
source packages/core/.venv/bin/activate
set -a; source .env 2>/dev/null; set +a
python3 scripts/chat.py
