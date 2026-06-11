# TPU-trigger

Trigger-level processing for neutrino telescopes on Google Coral Edge TPUs.
Follow-up to [Two Watts is All You Need (arXiv:2311.04983)](https://arxiv.org/abs/2311.04983)
([DeepInference_on_TPU](https://github.com/MiaochenJin/DeepInference_on_TPU),
[RecoOnEdge](https://github.com/MiaochenJin/RecoOnEdge)), with a modernized toolchain:

```
PyTorch  →  litert-torch (ex ai-edge-torch)  →  float .tflite
         →  ai-edge-quantizer (full int8 PTQ, static_wi8_ai8 recipe)
         →  edgetpu_compiler v16  →  *_edgetpu.tflite  →  Coral device
```

The FASRC cluster handles training, quantization, conversion, compilation, and
CPU-interpreter validation. Only on-device latency/power benchmarking needs the
physical Coral accelerator (kept locally).

## Cluster layout (FASRC Cannon)

Persistent lab share — code, env, tools, blessed artifacts:

```
/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/
├── TPU-trigger/              # this repo (git clone)
├── envs/tpu-trigger/         # python venv (plain venv, no conda)
├── tools/edgetpu_compiler/   # extracted compiler binary
└── results/TPU-trigger/      # final .tflite / *_edgetpu.tflite / compile logs
```

Scratch (purged when unused ~90 days — keep only re-stageable things here):

```
/n/netscratch/arguelles_delgado_lab/Everyone/miaochenjin/TPU-trigger/
├── data/                     # training data
└── runs/                     # checkpoints, intermediate artifacts, SLURM logs
```

## Environment

Plain `venv` on lab storage, built from the cluster python module — no
conda/Mambaforge. One-shot setup (run on a login node from the repo root):

```bash
bash env/setup_env.sh
```

Activation (always `module load` first, so the venv's base interpreter matches):

```bash
module load python/3.12.8-fasrc01
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/envs/tpu-trigger/bin/activate
export PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/tools/edgetpu_compiler/usr/bin:$PATH
```

Pins: `torch` (CUDA runtime bundled in the pip wheel — no CUDA module needed),
`litert-torch` (the renamed successor of ai-edge-torch). Exact resolved versions
are committed in `env/requirements.lock.txt` after install.

## edgetpu_compiler

`bash scripts/install_edgetpu_compiler.sh` downloads the v16 .deb from Google's
apt pool and extracts it (no root). Fallback if the binary won't run on Rocky 8:
`singularity build --fakeroot edgetpu-compiler.sif tools/edgetpu-compiler.def`.

## Milestone 0 — smoke test (go/no-go gate)

Before any real model work, `smoke_test/` must pass end-to-end — see
`smoke_test/README.md` for pass criteria and `smoke_test/RESULTS.md` for the
recorded outcome. The decisive risk it tests: the Edge TPU compiler is frozen at
2021, while litert-converter emits modern TFLite flatbuffers.

## SLURM

- Dev / debug: `arguelles_delgado_gpu_a100` (A100 1g.10gb MIG slices, low queue)
- Full training: `arguelles_delgado_gpu_mixed` (full A100-80GB) or `arguelles_delgado_h100` (H200)
- Conversion / quantization / compilation: CPU-only, login node is fine

Templates in `scripts/`: `train_gpu.sbatch` (wraps any python entrypoint),
`gpu_sanity.sbatch` (10-minute environment check).
