#!/bin/bash
# One-shot environment setup on the WARD workstation (non-SLURM, no sudo/apt,
# everything inside the project folder). Run from the repo root:
#     bash env/setup_env_ward.sh
# FASRC Cannon uses env/setup_env.sh instead (module load + lab paths).
#
# Host WARD: Threadripper 7970X (64t), 125 GiB RAM, RTX 5090 (sm_120), Ubuntu
# 24.04 (glibc 2.39, gcc 13). System Python 3.12.3 -> project-local .venv.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# --- venv (Ubuntu's system python lacks ensurepip -> bootstrap pip) ----------
if [ ! -d .venv ]; then
    python3 -m venv --without-pip .venv
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$PIP_CACHE_DIR/get-pip.py"
    .venv/bin/python "$PIP_CACHE_DIR/get-pip.py"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade -q pip

# --- python stack (torch+cu128 runs on the 5090's sm_120; same pin as FASRC) -
pip install -q -r env/requirements.txt
pip install -q -e .          # editable install -> `import tpu_trigger` anywhere

# --- k40gen native module (gcc 13) -------------------------------------------
# Re-applies the patches in tools/K40GEN_PATCHES.md. 1-2 are platform-agnostic;
# 4-5 are WARD/gcc-13/glibc-2.39 specific.
if ! python -c 'import k40gen' 2>/dev/null; then
    pip install -q pybind11      # build-time dep (find_package(pybind11 REQUIRED))
    [ -d tools/k40gen ] || git clone -q https://gitlab.nikhef.nl/roelaaij/k40gen.git tools/k40gen
    pushd tools/k40gen >/dev/null
    # 1. py3.11+ build break (SO -> EXT_SUFFIX)
    sed -i "s/get_config_var('SO')/get_config_var('EXT_SUFFIX')/" cmake/FindPythonLibsNew.cmake
    # 2. coincidence PMT/time misalignment (physics bug)
    sed -i 's/times\[++idx\]/times[idx++]/g' src/generate/generate.cpp
    # 4. gcc 13: std::domain_error needs <stdexcept> (was transitively included on gcc 8.5)
    grep -q '#include <stdexcept>' lib/generate/generate_common.h || \
        sed -i 's@#include <random>@#include <random>\n#include <stdexcept>@' lib/generate/generate_common.h
    # 5. glibc 2.39: bundled Catch2 v2 unit tests don't compile (MINSIGSTKSZ not constexpr).
    #    We don't need k40gen's C++ tests -> disable them.
    sed -i 's/option(ENABLE_TESTS "Enable tests" TRUE)/option(ENABLE_TESTS "Enable tests" FALSE)/' CMakeLists.txt
    rm -rf build
    popd >/dev/null
    pip install -q --no-build-isolation ./tools/k40gen
fi

# --- edgetpu_compiler v16 (extract .deb; wrapper is self-contained) ----------
if [ ! -x tools/edgetpu_compiler/usr/bin/edgetpu_compiler ]; then
    DEST="$ROOT/tools/edgetpu_compiler"; APT=https://packages.cloud.google.com/apt
    mkdir -p "$DEST"; pushd "$DEST" >/dev/null
    DEB=$(curl -s "$APT/dists/coral-edgetpu-stable/main/binary-amd64/Packages" \
        | awk '/^Package: edgetpu-compiler$/,/^$/' | awk '/^Filename:/{print $2}' | tail -1)
    curl -sO "$APT/$DEB"
    ar x edgetpu-compiler_*.deb
    tar -xf data.tar.xz 2>/dev/null || tar --zstd -xf data.tar.zst
    popd >/dev/null
fi

pip freeze > env/requirements.lock.ward.txt
echo
echo "venv + k40gen + edgetpu_compiler ready under $ROOT"
echo "next:  source env.sh  &&  python smoke_test/convert_and_compile.py"
