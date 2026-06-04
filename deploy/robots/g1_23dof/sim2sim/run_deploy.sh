#!/usr/bin/env bash
# Run the g1_ctrl deploy binary for sim2sim. g1_ctrl links conda's libfmt.so.11
# and the robostack ROS2 libs (librclcpp etc.), which live in $CONDA_PREFIX/lib;
# conda run does not put that on the loader path, so export it here.
# DDS domain is 1 (set in main.cpp); pass --network lo.
set -euo pipefail
BUILD="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)"
conda run -n g1tt_sim2sim bash -c "
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:/usr/local/lib:\${LD_LIBRARY_PATH:-}
  cd '$BUILD'
  exec ./g1_ctrl --network lo \"\$@\"
" -- "$@"
