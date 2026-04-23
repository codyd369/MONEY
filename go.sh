#!/usr/bin/env bash
# Resume live actions by removing the .kill sentinel. 5-second countdown so a
# misclick can be aborted with Ctrl-C.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$HERE/.kill" ]]; then
  echo "[go.sh] No .kill file at $HERE/.kill. Nothing to do."
  exit 0
fi
echo "[go.sh] About to remove .kill in 5 seconds. Ctrl-C to abort."
for i in 5 4 3 2 1; do
  echo "  $i..."
  sleep 1
done
rm -f "$HERE/.kill"
echo "[go.sh] .kill removed. Live actions are re-armed (subject to DRY_RUN, KALSHI_ENV, and strategy state)."
