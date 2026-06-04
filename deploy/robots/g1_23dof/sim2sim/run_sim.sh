#!/usr/bin/env bash
# Run the table-tennis sim. MUST use `conda run --no-capture-output` so tt_sim's
# keyboard reader gets a real terminal stdin (plain `conda run` pipes stdin, so
# keypresses echo to the shell instead of reaching tt_sim -> FSM keys do nothing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec conda run --no-capture-output -n g1tt_sim2sim python tt_sim.py "$@"
