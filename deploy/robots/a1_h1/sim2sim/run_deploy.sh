#!/usr/bin/env bash
# Run the C++ policy bridge in dry-run mode by default. It subscribes ROS topics
# and publishes diagnostics, but does not publish /model_action unless
# PUBLISH_ACTIONS=true is set. Policy inference/action output is also gated by
# POLICY_ENABLED; keep it false when the FSM supervisor is running.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$HERE/build"
ONNX="$HERE/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib"

if [ ! -x "$BUILD/a1_policy_bridge_cpp" ]; then
  echo "ERROR: $BUILD/a1_policy_bridge_cpp not found. Run sim2sim/build_deploy.sh first." >&2
  exit 1
fi

set +u; source /opt/ros/humble/setup.bash; set -u
export LD_LIBRARY_PATH="$ONNX:${LD_LIBRARY_PATH:-}"

PUBLISH_ACTIONS="${PUBLISH_ACTIONS:-false}"
POLICY_ENABLED="${POLICY_ENABLED:-false}"
MAX_DELTA_PER_TICK="${MAX_DELTA_PER_TICK:-[0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160]}"
if [ -z "${POLICY_PATH:-}" ]; then
  ROOT="$HERE"
  while [ "$ROOT" != "/" ]; do
    if [ -d "$ROOT/Pingpong_TTRL" ] && [ -d "$ROOT/unitree_rl_lab" ]; then
      break
    fi
    ROOT="$(dirname "$ROOT")"
  done
  if [ ! -d "$ROOT/Pingpong_TTRL" ] || [ ! -d "$ROOT/unitree_rl_lab" ]; then
    echo "ERROR: could not find lgy root. Set POLICY_PATH explicitly." >&2
    exit 1
  fi
  POLICY_PATH="$ROOT/Pingpong_TTRL/logs/a1_tt_v13/2026-07-08_12-40-15/exported/policy.onnx"
fi
PREDICTOR_PATH="${PREDICTOR_PATH:-}"

ARGS=(
  --ros-args
  -p "publish_actions:=$PUBLISH_ACTIONS"
  -p "policy_enabled_on_start:=$POLICY_ENABLED"
  -p "policy_path:=$POLICY_PATH"
  -p "max_delta_per_tick:=$MAX_DELTA_PER_TICK"
)
if [ -n "$PREDICTOR_PATH" ]; then
  ARGS+=(-p "predictor_path:=$PREDICTOR_PATH")
fi

exec "$BUILD/a1_policy_bridge_cpp" "${ARGS[@]}"
