#!/usr/bin/env bash
# Switch the g1_23dof deploy between SIM2SIM and REAL. These two modes differ in
# THREE places that must all match, or the robot mis-maps / DDS fails / diverges:
#
#   axis            | SIM2SIM (sim)                  | REAL (robot)
#   ----------------|--------------------------------|------------------------------
#   joint_ids_map   | identity [0..22] (mujoco has   | sparse [0-11,12,15..19,22..26]
#                   |   23 contiguous actuators)     |   (G1 SDK 29-slot enum)
#   build/g1_ctrl   | conda build (CycloneDDS)       | system build (/opt/ros/humble
#                   |                                |   FastDDS)
#   DDS / RMW       | CycloneDDS on lo (loopback xml)| FastDDS; vrpn must also be
#                   |   run_sim.sh + run_deploy.sh   |   FastDDS. run_deploy_real.sh
#
# Backups: _conda_build_backup/build.conda/g1_ctrl , _system_build_backup/g1_ctrl.system
# This script swaps the binary + joint_ids_map. The run scripts are already
# split (run_deploy.sh = sim, run_deploy_real.sh = real) and not touched here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # robots/g1_23dof
cd "$HERE"
MODE="${1:-}"
[ "$MODE" = sim ] || [ "$MODE" = real ] || { echo "usage: $(basename "$0") sim|real"; exit 1; }

FILES=(config/config.yaml config/policy/table_tennis/v*/params/deploy.yaml)

set_map() {  # $1 = from-tail  $2 = to-tail  (handles both spaced and non-spaced yaml)
  for f in "${FILES[@]}"; do
    sed -i "s/12,${1}]/12,${2}]/"                         "$f"   # non-spaced
    sed -i "s/12, ${1//,/, }]/12, ${2//,/, }]/"           "$f"   # spaced
  done
}
SPARSE_TAIL='15,16,17,18,19,22,23,24,25,26'
IDENT_TAIL='13,14,15,16,17,18,19,20,21,22'

# Velocity policy v2 (2026-06-23, OUR g1_locomotion_v2) uses the SAME leg-first
# joint order as TableTennis (G1_JOINT_NAMES) — deliberately, so it deploys under
# the same joint_ids_map. So its sim/real maps are IDENTICAL to the TT pattern
# (sim = identity [0..22] on mujoco's 23 contiguous actuators; real = sparse G1
# SDK motor enum). (The old foreign policy was IsaacLab-interleaved -> different.)
VEL="config/policy/velocity/params/deploy.yaml"
VEL_SIM='joint_ids_map: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]'
VEL_REAL='joint_ids_map: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]'
set_vel_map() { [ -f "$VEL" ] && sed -i "s|^joint_ids_map:.*|$1|" "$VEL"; }

if [ "$MODE" = sim ]; then
  cp -a _conda_build_backup/build.conda/g1_ctrl build/g1_ctrl
  set_map "$SPARSE_TAIL" "$IDENT_TAIL"
  set_vel_map "$VEL_SIM"
  echo "[switch] SIM2SIM: conda binary + identity joint_ids_map (+ velocity sim map)"
  echo "  run:  TT_NO_SERVE=1 bash sim2sim/run_sim.sh   (terminal 1)"
  echo "        bash sim2sim/run_deploy.sh              (terminal 2)"
else
  cp -a _system_build_backup/g1_ctrl.system build/g1_ctrl
  set_map "$IDENT_TAIL" "$SPARSE_TAIL"
  set_vel_map "$VEL_REAL"
  echo "[switch] REAL: system binary + sparse joint_ids_map (+ velocity real map)"
  echo "  run:  source /opt/ros/humble/setup.bash; ros2 launch vrpn_mocap client.launch.yaml server:=10.1.1.198 port:=3883   (terminal 1)"
  echo "        bash sim2sim/run_deploy_real.sh         (terminal 2, --network enx6c1ff76cb7d7)"
  echo "  NOTE: vrpn MUST be FastDDS (system ROS) to match the controller, or robot_pos=0 -> policy diverges."
fi
echo "[switch] verify:"
grep -h "^joint_ids_map:" config/config.yaml config/policy/table_tennis/v5/params/deploy.yaml | sed 's/^/  TT  /'
[ -f "$VEL" ] && grep -h "^joint_ids_map:" "$VEL" | sed 's/^/  VEL /'
