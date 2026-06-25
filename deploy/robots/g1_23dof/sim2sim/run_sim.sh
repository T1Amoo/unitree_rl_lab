#!/usr/bin/env bash
# Run the table-tennis sim (forked from unitree_mujoco's proven loop). Keyboard
# input goes through the MuJoCo VIEWER window (glfw key_callback): focus the
# window, then f=FixStand g=TableTennis p=Passive | 7/8 band raise/lower | 9 release | q quit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# sim2sim is ALL CycloneDDS: this process initializes the Unitree SDK's bundled
# CycloneDDS in-process (UnitreeSdk2Bridge), so rclpy must ALSO be CycloneDDS
# (conda's robostack default) — forcing rmw_fastrtps_cpp here makes rclpy node
# creation fail with "rmw handle is invalid". The sim2sim controller must
# therefore also be the CycloneDDS (conda) g1_ctrl, not the system-ROS build.
exec conda run --no-capture-output -n g1tt_sim2sim bash -c "
  # Override the conda env's CYCLONEDDS_URI (pins enp8s0, DOWN for sim2sim) with
  # the loopback config so the Unitree DDS + rclpy mocap both run on lo.
  export CYCLONEDDS_URI='file://$HERE/cyclonedds_loopback.xml'
  exec python tt_sim_mujoco.py \"\$@\"
" -- "$@"
