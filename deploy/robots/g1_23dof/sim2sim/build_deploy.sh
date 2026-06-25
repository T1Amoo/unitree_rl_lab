#!/usr/bin/env bash
# Build g1_ctrl (and tt_replay) against the SYSTEM ROS2 humble (/opt/ros/humble,
# rmw_fastrtps_cpp) -- NOT conda. This keeps the whole deploy on one DDS stack
# that matches the system vrpn_mocap publisher (no conda CycloneDDS in the mix).
#   - rclcpp / geometry_msgs : /opt/ros/humble (ros-base)
#   - fmt / spdlog / yaml-cpp / boost::program_options : system apt packages
#       (libfmt8, libspdlog1, libyaml-cpp0.7, libboost-program-options-dev)
#   - onnxruntime            : vendored in ../../thirdparty
#   - Unitree SDK + its CycloneDDS (ddsc/ddscxx, iceoryx) : /usr/local
#       (this is the ROBOT-side DDS, independent of the ROS RMW above)
# The previous conda-built toolchain is backed up under _conda_build_backup/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # robots/g1_23dof
cd "$HERE"

# A live conda env would inject its own rclcpp / rmw_cyclonedds / libfmt.so.11
# and silently win over the system ones. Refuse to build until it is gone.
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: a conda env is active ($CONDA_PREFIX)." >&2
  echo "       Run 'conda deactivate' (possibly twice) until CONDA_PREFIX is unset, then re-run." >&2
  exit 1
fi

# ROS setup.bash references unset vars; relax nounset just for the source.
set +u; source /opt/ros/humble/setup.bash; set -u

rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_PREFIX_PATH="/opt/ros/humble" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/lib"
make g1_ctrl tt_replay -j4
echo "[build_deploy] g1_ctrl built (system ROS /opt/ros/humble) at $HERE/build/g1_ctrl"
