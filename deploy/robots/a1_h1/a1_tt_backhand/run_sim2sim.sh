#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd "$script_dir/../../../../.." && pwd)

python_bin=${A1_TT_BACKHAND_PYTHON:-/home/woan/.conda/envs/g1tt_sim2sim/bin/python}
policy_path=${A1_TT_BACKHAND_POLICY:-$workspace_root/Pingpong_TTRL/logs/a1_tt_backhand_real_v1_y055h105/2026-08-01_11-05-58_scratch_backhand_camera_age35_tau_delay_dr_servey055h105_10k10k10k/exported/policy.onnx}

exec "$python_bin" -u "$script_dir/../sim2sim/run_a1_tt_sim2sim.py" \
  --policy "$policy_path" \
  --diag-every 50 \
  "$@"
