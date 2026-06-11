#!/bin/bash
#SBATCH -J tpu-trigger-sanity
#SBATCH -A arguelles_delgado_lab
#SBATCH -p arguelles_delgado_gpu_a100
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-00:10
#SBATCH -o /n/netscratch/arguelles_delgado_lab/Everyone/miaochenjin/TPU-trigger/runs/sanity-%j.out

LAB=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin
module load python/3.12.8-fasrc01
source "$LAB/envs/tpu-trigger/bin/activate"
export PATH=$LAB/tools/edgetpu_compiler/usr/bin:$PATH

python - <<'EOF'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
EOF
echo -n "edgetpu_compiler: "; edgetpu_compiler --version
