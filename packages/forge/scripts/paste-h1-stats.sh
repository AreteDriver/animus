#!/usr/bin/env bash
# Manual paste path for HackerOne program stats.
#
# HackerOne's program page (https://hackerone.com/anthropic) is a JS-rendered
# SPA, so curl returns only the shell. To refresh program-stats intel, paste
# the rendered page text into this script's stdin.
#
# Usage:
#   1. Open https://hackerone.com/anthropic in a browser
#   2. Select All + Copy (Ctrl+A, Ctrl+C)
#   3. Run: scripts/paste-h1-stats.sh
#      then paste, then Ctrl+D
#
# Saves to: ~/.animus/bounty_intel/raw/h1-paste.txt
# The next bounty-watcher run will parse stats from this file.

set -euo pipefail

STATE_DIR="${STATE_DIR:-$HOME/.animus/bounty_intel}"
mkdir -p "$STATE_DIR/raw"
DEST="$STATE_DIR/raw/h1-paste.txt"

echo "Paste HackerOne page text, then Ctrl+D when done:"
cat > "$DEST"

BYTES=$(wc -c < "$DEST")
echo "Saved $BYTES bytes to $DEST"

# Quick sanity check
python3 - "$DEST" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="ignore")
patterns = {
    "total_paid": r"Total bounties paid\s*\$([0-9,]+)",
    "paid_90d": r"Bounties paid \| 90 days\s*\$([0-9,]+)",
    "reports_90d": r"Reports received \| 90 days\s*([0-9,]+)",
    "reports_resolved": r"Reports resolved\s*([0-9,]+)",
    "hackers_thanked": r"Hackers thanked\s*([0-9,]+)",
    "response_efficiency": r"Response efficiency:\s*([0-9]+)%",
    "assets_in_scope": r"Assets In Scope\s*([0-9,]+)",
}
found = 0
for key, pat in patterns.items():
    m = re.search(pat, text)
    if m:
        print(f"  {key}: {m.group(1)}")
        found += 1
print(f"\nParsed {found}/{len(patterns)} stat fields")
if found == 0:
    print("WARNING: no stats parsed — did you paste the program landing page?")
    sys.exit(2)
PY
