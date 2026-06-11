#!/bin/bash
# Usage: sbatch scripts/train_gpu.sbatch <python-script> [args...]
# Dev default is the MIG partition; for full training switch to
# -p arguelles_delgado_gpu_mixed (full A100-80GB) or arguelles_delgado_h100.
#SBATCH -J tpu-trigger
#SBATCH -A arguelles_delgado_lab
#SBATCH -p arguelles_delgado_gpu_a100
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-08:00
#SBATCH -o /n/netscratch/arguelles_delgado_lab/Everyone/miaochenjin/TPU-trigger/runs/slurm-%j.out

LAB=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin
module load python/3.12.8-fasrc01
source "$LAB/envs/tpu-trigger/bin/activate"
export PATH=$LAB/tools/edgetpu_compiler/usr/bin:$PATH

python "$@"
