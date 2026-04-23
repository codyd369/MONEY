#!/usr/bin/env bash
# Emergency stop: places a .kill sentinel at the repo root.
# The scheduler and executor poll for this file on every iteration and
# short-circuit before any order placement or publishing when it exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
touch "$HERE/.kill"
echo "[kill.sh] .kill created at $HERE/.kill"
echo "[kill.sh] All live actions (orders, publishing) are now blocked."
echo "[kill.sh] Run ./go.sh to resume (5-second countdown)."
