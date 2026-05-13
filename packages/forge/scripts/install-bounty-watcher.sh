#!/usr/bin/env bash
# Install the Bounty Watcher as a user-level systemd timer.
#
# Idempotent: safe to re-run. Does NOT auto-enable — prints the enable
# command at the end so the user explicitly opts in.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"

mkdir -p "$UNIT_DIR"
cp "$SCRIPT_DIR/animus-bounty-watcher.service" "$UNIT_DIR/"
cp "$SCRIPT_DIR/animus-bounty-watcher.timer" "$UNIT_DIR/"

systemctl --user daemon-reload

echo "Installed unit files to $UNIT_DIR"
echo ""
echo "To enable and start the daily timer:"
echo "  systemctl --user enable --now animus-bounty-watcher.timer"
echo ""
echo "To run once on demand:"
echo "  systemctl --user start animus-bounty-watcher.service"
echo ""
echo "To check status / next firing:"
echo "  systemctl --user list-timers animus-bounty-watcher.timer"
echo ""
echo "To follow the log:"
echo "  tail -F ~/.animus/bounty_intel/cron.log"
