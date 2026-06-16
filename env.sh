#!/usr/bin/env bash
# TPU-trigger -- workstation environment (host: WARD)
# Project-local only; no global installs. Source from the project root:
#     source env.sh
# (FASRC Cannon uses env/setup_env.sh + `module load` instead — see README.)
_here="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"
export TPU_TRIGGER_ROOT="$_here"

# Activate the project-local virtualenv (system Python 3.12 + pip wheels).
if [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$TPU_TRIGGER_ROOT/.venv/bin/activate"
fi

# edgetpu_compiler (extracted .deb; wrapper is self-contained via bundled ld).
if [ -d "$TPU_TRIGGER_ROOT/tools/edgetpu_compiler/usr/bin" ]; then
    case ":$PATH:" in
        *":$TPU_TRIGGER_ROOT/tools/edgetpu_compiler/usr/bin:"*) ;;
        *) export PATH="$TPU_TRIGGER_ROOT/tools/edgetpu_compiler/usr/bin:$PATH" ;;
    esac
fi

# Keep the pip cache inside the project (no writes outside the project folder).
export PIP_CACHE_DIR="$TPU_TRIGGER_ROOT/.cache/pip"

echo "[TPU-trigger@WARD] ready: $(python --version 2>&1)"
echo "  TPU_TRIGGER_ROOT = $TPU_TRIGGER_ROOT"
echo "  edgetpu_compiler = $(command -v edgetpu_compiler 2>/dev/null || echo 'NOT ON PATH')"
echo "  torch CUDA       = $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' 2>/dev/null || echo '?')"
