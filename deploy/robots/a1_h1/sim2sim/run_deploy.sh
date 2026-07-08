#!/usr/bin/env bash
# Run the C++ policy bridge in dry-run mode by default. It subscribes ROS topics
# and publishes diagnostics, but does not publish /model_action unless
# PUBLISH_ACTIONS=true is set.
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
  POLICY_PATH="$ROOT/Pingpong_TTRL/logs/a1_tt_v11/2026-07-07_10-48-31/exported/policy.onnx"
fi
PREDICTOR_PATH="${PREDICTOR_PATH:-}"

ARGS=(
  --ros-args
  -p "publish_actions:=$PUBLISH_ACTIONS"
  -p "policy_path:=$POLICY_PATH"
)
if [ -n "$PREDICTOR_PATH" ]; then
  ARGS+=(-p "predictor_path:=$PREDICTOR_PATH")
fi

exec "$BUILD/a1_policy_bridge_cpp" "${ARGS[@]}"
