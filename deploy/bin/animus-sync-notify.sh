#!/usr/bin/env bash
# OnFailure handler for animus-sync.service — makes a sync failure VISIBLE.
#  - writes ~/.animus/sync.FAILED (the statusline surfaces it as ⚠sync:FAIL)
#  - best-effort desktop notification
# Why: the 4h memory-sync cron failed silently for 5 days (2026-05-28→06-02)
# because errors only appended to ~/.animus/sync.log, which nobody reads.
mkdir -p "$HOME/.animus"
{
  echo "animus-sync FAILED at $(date -Iseconds)"
  echo "--- last 25 journal lines ---"
  journalctl --user -u animus-sync.service -n 25 --no-pager 2>/dev/null
} > "$HOME/.animus/sync.FAILED"
notify-send -u critical "Animus sync FAILED" "Check: journalctl --user -u animus-sync" 2>/dev/null || true
exit 0
