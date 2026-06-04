#!/usr/bin/env bash
# Deploy the Animus memory-sync systemd --user timer from this repo (source of
# truth) into the live locations. COPIES the units (does not symlink) so they
# survive branch switches of this working tree — a symlinked unit would dangle
# whenever the tree is on a branch without deploy/, silently breaking sync.
# Idempotent: re-run after editing any unit here. See deploy/README.md.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../deploy
UNIT_DIR="$HOME/.config/systemd/user"
BIN_DIR="$HOME/.local/bin"
mkdir -p "$UNIT_DIR" "$BIN_DIR"

install -m 0644 "$SRC/systemd/user/animus-sync.service"      "$UNIT_DIR/animus-sync.service"
install -m 0644 "$SRC/systemd/user/animus-sync-fail.service" "$UNIT_DIR/animus-sync-fail.service"
install -m 0644 "$SRC/systemd/user/animus-sync.timer"        "$UNIT_DIR/animus-sync.timer"
install -m 0755 "$SRC/bin/animus-sync-notify.sh"             "$BIN_DIR/animus-sync-notify.sh"

systemctl --user daemon-reload
systemctl --user enable --now animus-sync.timer

echo "Deployed animus-sync timer. Next run:"
systemctl --user list-timers animus-sync.timer --no-pager | grep -i animus-sync || true
