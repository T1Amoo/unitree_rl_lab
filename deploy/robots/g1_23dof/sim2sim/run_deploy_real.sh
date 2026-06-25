#!/usr/bin/env bash
# Launch g1_ctrl on the REAL G1 against the SYSTEM ROS2 humble (rmw_fastrtps_cpp),
# matching the system-ROS vrpn_mocap publisher. This is the sim2sim run_deploy.sh
# but with the real NIC instead of `lo`:
#   --network enx6c1ff76cb7d7  (USB->G1, 192.168.123.169)  -> main.cpp picks
#   Unitree DDS domain 0 (real robot) automatically.
# LD_LIBRARY_PATH must include /usr/local/lib (Unitree SDK CycloneDDS ddsc/ddscxx)
# and the onnxruntime dir; /opt/ros/humble/lib is added by setup.bash.
#
# >>> REAL ROBOT MOVES. Have the e-stop ready. Enter FixStand (LT+up) before
#     TableTennis (RB+Y). Ball is blocked in config (ball_topic disabled) for the
#     no-ball standing test. <<<
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # robots/g1_23dof
BUILD="$HERE/build"
ONNX="$HERE/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib"
NIC="${1:-enx6c1ff76cb7d7}"   # override by passing a NIC name as $1

pkill -9 -x g1_ctrl 2>/dev/null || true   # one instance only (exact binary name)
sleep 1

# ROS setup.bash references unset vars; relax nounset just for the source.
set +u; source /opt/ros/humble/setup.bash; set -u
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"          # ROS/mocap domain (matches vrpn)
export LD_LIBRARY_PATH="/usr/local/lib:$ONNX:${LD_LIBRARY_PATH:-}"

cd "$BUILD"
echo "[run_deploy_real] g1_ctrl --network $NIC  (RMW=$RMW_IMPLEMENTATION ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
exec stdbuf -oL -eL ./g1_ctrl --network "$NIC"
