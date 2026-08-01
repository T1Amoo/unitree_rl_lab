#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd "$script_dir/../../../../.." && pwd)

python_bin=${A1_TT_BACKHAND_PYTHON:-/home/woan/.conda/envs/g1tt_sim2sim/bin/python}
policy_path=${A1_TT_BACKHAND_POLICY:-$workspace_root/Pingpong_TTRL/pretrained/a1_tt_backhand/base_9700_hitplane020/policy/policy.onnx}

export A1_SIM2SIM_PROFILE=${A1_SIM2SIM_PROFILE:-v1_3_backhand_low_arm_hitplane020}

exec "$python_bin" -u "$script_dir/sim2sim/run_a1_tt_sim2sim.py" \
  --policy "$policy_path" \
  --actuator-mode damiao_mit \
  --real-response-model \
  --serve-pause-steps 100 \
  --diag-every 50 \
  "$@"
