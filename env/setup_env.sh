#!/bin/bash
# One-shot environment setup on FASRC Cannon. Run on a login node from the repo
# root: bash env/setup_env.sh
# Override the python module with PYMOD=python/3.10.13-fasrc01 if cp312 wheels
# are missing for any pin.
set -euo pipefail

LAB=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin
SCRATCH=/n/netscratch/arguelles_delgado_lab/Everyone/miaochenjin
PYMOD=${PYMOD:-python/3.12.8-fasrc01}
VENV=$LAB/envs/tpu-trigger
REPO_ENV_DIR=$(cd "$(dirname "$0")" && pwd)

# pip cache must stay off the nearly-full home directory
export PIP_CACHE_DIR=$SCRATCH/.pip-cache

mkdir -p "$LAB/envs" "$LAB/tools" "$LAB/results/TPU-trigger" \
         "$SCRATCH/TPU-trigger/data" "$SCRATCH/TPU-trigger/runs" "$PIP_CACHE_DIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$PYMOD"

python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$REPO_ENV_DIR/requirements.txt"

python - <<'EOF'
import torch, litert_torch, ai_edge_quantizer, ai_edge_litert
print("imports OK")
print("torch", torch.__version__)
print("litert_torch", getattr(litert_torch, "__version__", "?"))
EOF

pip freeze > "$REPO_ENV_DIR/requirements.lock.txt"
echo "venv ready at $VENV; lockfile written to env/requirements.lock.txt"
