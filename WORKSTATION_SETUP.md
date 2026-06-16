# TPU-trigger — Workstation Setup (host: WARD)

Reproducible setup for running the **entire** TPU-trigger pipeline — background
generation, GPU training, int8 quantization, and Edge TPU compilation — on the
local **WARD** workstation. **Design constraint (shared with AtmNuDataFit):
everything lives inside this project folder; no global installs, no
`sudo`/`apt`.** FASRC Cannon remains available; see `README.md` / `env/setup_env.sh`
for the cluster path. Per the cross-project convention, log entries are tagged
**[WARD]** (workstation) or **[FASRC]** (cluster).

## Host

| | |
|---|---|
| Host / user | `WARD` (128.103.100.27) / `hideon` — `ssh ward` |
| OS | Ubuntu 24.04.4 LTS (glibc 2.39, gcc/g++ 13.3) |
| CPU | AMD Threadripper 7970X — 32c / **64t** |
| RAM | 125 GiB |
| GPU | **NVIDIA RTX 5090 32 GB (Blackwell, sm_120), driver 580 / CUDA 13.0** |
| Project root | `/home/hideon/Desktop/Projects/TPU-trigger` |

Unlike AtmNuDataFit (CPU-only nuSQuIDS, GPU idle), TPU-trigger **uses the
RTX 5090** for training: `torch==2.11.0+cu128` ships sm_120 kernels, validated
running real CUDA compute on the 5090. The 64-thread CPU drives background
generation (k40gen) and the x86-64 Edge TPU compiler. **The whole pipeline runs
on one box with no SLURM queue**, and gcc 13 (vs FASRC's gcc 8.5) is expected to
ease the phase-2b PROPOSAL source build.

## Environment design

- **System Python 3.12.3** → project-local `.venv/`. No pyenv/conda. Ubuntu's
  system Python lacks `ensurepip`, so the venv is created `--without-pip` and pip
  is bootstrapped from `get-pip.py` (no `apt`).
- **Python stack** = the same `env/requirements.txt` as FASRC (`torch 2.11.0+cu128`,
  `litert-torch 0.9.1`, numpy/scipy/h5py/…). The cu128 pin works here too: the
  CUDA 13.0 driver runs the cu128 runtime (backward-compatible).
- **k40gen** is built from source against gcc 13 (`pip install --no-build-isolation
  ./tools/k40gen`), re-applying the patches in `tools/K40GEN_PATCHES.md` —
  including two WARD-only ones (#4 `<stdexcept>`, #5 disable Catch2 tests).
- **edgetpu_compiler v16** is the extracted `.deb`; its wrapper bundles its own
  `ld-linux` + libs, so it **runs natively on Ubuntu 24.04** (no Singularity).
- pip cache redirected to `.cache/pip` (stays in-project).

## Daily use

```bash
cd ~/Desktop/Projects/TPU-trigger
source env.sh        # activates .venv, puts edgetpu_compiler on PATH, reports GPU
python smoke_test/convert_and_compile.py   # go/no-go gate: 100% op mapping + parity
```

`env.sh` is the single entry point — source it in every shell before running.

## Validation (2026-06-16) [WARD]

- **GPU**: `torch.cuda.is_available()` True, device RTX 5090, capability (12,0);
  a real `sm_120` matmul kernel executes.
- **k40gen end-to-end** via `tpu_trigger.backgrounds.k40.generate_full_detector`
  (10 ms full reference detector): 4,595,709 hits over 2070 DOMs × 31 PMT,
  effective **7,162 Hz/PMT** (expected ~7000–7800 incl. coincidences) → PASS.
- **Smoke test** (PyTorch → litert → int8 PTQ → `edgetpu_compiler -s` → CPU-interp
  parity): **5/5 ops Mapped to Edge TPU, 1 subgraph, 0 CPU ops**, int8 parity
  r=0.99892 (nrmse 0.014). Reproduces the FASRC result on Ubuntu 24.04.

## Reproduce from scratch

```bash
cd ~/Desktop/Projects && git clone git@github.com:MiaochenJin/TPU-Trigger.git TPU-trigger
cd TPU-trigger && bash env/setup_env_ward.sh   # venv + stack + k40gen + edgetpu_compiler
source env.sh && python smoke_test/convert_and_compile.py
```

`env/setup_env_ward.sh` is idempotent (skips steps already done) and writes
`env/requirements.lock.ward.txt`.

## Remotes / sync

WARD has working **GitHub SSH** (`git@github.com` authenticates as MiaochenJin),
so WARD ↔ GitHub directly — no lab bare repo needed (that is for FASRC). Flow:
commit on Mac → push to GitHub → `git pull` on WARD (and vice-versa). GitHub
`MiaochenJin/TPU-Trigger` stays canonical.

## Known gaps / follow-ups

- **Physical Coral**: on-device latency/power benchmarking still needs the USB
  Coral accelerator. WARD does everything up to the compiled `*_edgetpu.tflite`;
  whether the device is plugged into WARD or stays on the Mac is TBD.
- **Data/runs**: large generated datasets (`bg_v1.h5`, etc.) are gitignored —
  regenerate on WARD or copy from FASRC `/n/holylfs05/.../results/TPU-trigger/`.
- **Build artifacts** (`.venv/`, `.cache/`, `tools/k40gen/`,
  `tools/edgetpu_compiler/`) are gitignored — they are rebuilt by the setup script.
