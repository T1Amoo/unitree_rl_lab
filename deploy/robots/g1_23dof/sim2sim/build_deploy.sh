#!/usr/bin/env bash
# Build g1_ctrl (and tt_replay) for sim2sim inside the g1tt_sim2sim conda env.
# The conda gcc-14 toolchain cannot reconcile the /usr/local unitree SDK headers
# with its sandboxed glibc, and the bare `fmt` token needs conda's libfmt.so.11
# (main.cpp.o is compiled against conda spdlog/fmt) to win over system libfmt.so.8.
# Hence: system compiler + explicit -L ordering ($CONDA_PREFIX/lib first).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # robots/g1_23dof
cd "$HERE"
rm -rf build && mkdir build && cd build
conda run -n g1tt_sim2sim bash -c '
  cmake .. \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$CONDA_PREFIX/share" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_EXE_LINKER_FLAGS="-L$CONDA_PREFIX/lib -L/usr/local/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$CONDA_PREFIX/lib -L/usr/local/lib"
  make g1_ctrl tt_replay -j4
'
echo "[build_deploy] g1_ctrl built at $HERE/build/g1_ctrl"
