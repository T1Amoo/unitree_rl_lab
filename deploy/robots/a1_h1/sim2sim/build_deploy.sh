#!/usr/bin/env bash
# Build the A1/H1 C++ policy bridge against system ROS2 Humble and the vendored
# ONNX Runtime under unitree_rl_lab/deploy/thirdparty. Keep conda out of this
# build, matching the G1 deploy convention.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: a conda env is active ($CONDA_PREFIX)." >&2
  echo "       Run 'conda deactivate' until CONDA_PREFIX is unset, then re-run." >&2
  exit 1
fi

set +u; source /opt/ros/humble/setup.bash; set -u

rm -rf build
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/opt/ros/humble \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j4
echo "[build_deploy] built $HERE/build/a1_policy_bridge_cpp"
