#!/bin/bash
# Download and extract edgetpu_compiler v16 without root (FASRC login node).
# The binary is a 2021 Debian build; Rocky 8's glibc 2.28 should satisfy it —
# the ldd check below verifies. If it fails, build the Singularity fallback
# from tools/edgetpu-compiler.def instead.
set -euo pipefail

LAB=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin
DEST=$LAB/tools/edgetpu_compiler
APT=https://packages.cloud.google.com/apt

mkdir -p "$DEST" && cd "$DEST"

DEB_PATH=$(curl -s "$APT/dists/coral-edgetpu-stable/main/binary-amd64/Packages" \
  | awk '/^Package: edgetpu-compiler$/,/^$/' | awk '/^Filename:/{print $2}' | tail -1)
[ -n "$DEB_PATH" ] || { echo "could not find edgetpu-compiler in apt index" >&2; exit 1; }
echo "downloading $APT/$DEB_PATH"
curl -sO "$APT/$DEB_PATH"

ar x edgetpu-compiler_*.deb
tar -xf data.tar.xz 2>/dev/null || tar --zstd -xf data.tar.zst

echo "--- ldd check ---"
ldd usr/bin/edgetpu_compiler
echo "--- version ---"
./usr/bin/edgetpu_compiler --version

echo
echo "OK. Add to PATH: export PATH=$DEST/usr/bin:\$PATH"
