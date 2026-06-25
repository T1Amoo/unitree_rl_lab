#!/usr/bin/env bash
# Run the g1_ctrl deploy binary for sim2sim. g1_ctrl links conda's libfmt.so.11
# and the robostack ROS2 libs (librclcpp etc.), which live in $CONDA_PREFIX/lib;
# conda run does not put that on the loader path, so export it here.
# DDS domain is 1 (set in main.cpp); pass --network lo.
set -euo pipefail
BUILD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)"
LOOPBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cyclonedds_loopback.xml"
# Kill any previous g1_ctrl first. The deploy's init_fsm_state() detects another
# lowcmd writer and calls go2::shutdown() then keeps running on a dead DDS handle
# (the exit(0) is commented out) -> segfault / "cannot connect". One instance only.
pkill -9 -x g1_ctrl 2>/dev/null || true   # match by exact binary name: catches ./g1_ctrl AND build/g1_ctrl
sleep 1
conda run --no-capture-output -n g1tt_sim2sim bash -c "
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:/usr/local/lib:\${LD_LIBRARY_PATH:-}
  # Override the conda env's CYCLONEDDS_URI (it pins enp8s0, the wired LAN, which
  # is DOWN for sim2sim) with the loopback config so DDS runs on lo.
  export CYCLONEDDS_URI='file://$LOOPBACK'
  cd '$BUILD'
  exec stdbuf -oL -eL ./g1_ctrl --network lo \"\$@\"
" -- "$@"
