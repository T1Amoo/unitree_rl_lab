#!/usr/bin/env bash
# Run the table-tennis sim (forked from unitree_mujoco's proven loop). Keyboard
# input goes through the MuJoCo VIEWER window (glfw key_callback): focus the
# window, then f=FixStand g=TableTennis p=Passive | 7/8 band raise/lower | 9 release | q quit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
exec conda run --no-capture-output -n g1tt_sim2sim python tt_sim_mujoco.py "$@"
