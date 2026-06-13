# TPU-trigger — work log

Trigger-level signal/noise classification for neutrino telescopes on Google
Coral Edge TPUs. Follow-up to *Two Watts is All You Need* (arXiv:2311.04983).
Detailed results live in `reports/`; phase plans in `docs/`; this file is the
chronological narrative + design decisions not derivable from the code.

---

## Status — 2026-06-12

**Done:** environment + Edge TPU toolchain (phase 0), mock train→quantize→
compile pipeline (phase 1), and the full KM3NeT background-generation stack
(phase 2a) — K40 (k40gen), PMT dark noise, bioluminescence, composed into
`(31, 256)` windows, all physics-validated, with the 31-channel pipeline
re-verified at 100% Edge TPU mapping. Event displays exist
(`tpu_trigger.display`).

**Next up:** the unified photon→PE response interface (designed 2026-06-12,
see below — **not yet implemented**), then the muon signal source on top of
it (phase 2b).

**Handoff (2026-06-12):** development continues under **Claude Opus 4.8**,
taking over from **Claude Fable 5** because of a Fable 5 service outage. No
change to plan, repo, or conventions — see "Operational notes" for how to
resume on the cluster. The immediately actionable item is the response-module
build order at the end of the design section.

---

## Design — unified photon→PE response interface (decided 2026-06-12, NOT yet implemented)

Concluded in discussion; supersedes the placeholder biolum acceptance. The
goal is one response model shared by **all light-based sources** (bioluminescence,
muons, and the phase-2b Prometheus signal), so a classifier learns physics, not
per-class simulation artifacts.

**Interface.** A light source delivers photons at the DOM as
`(arrival_time, direction, wavelength)`. The response module converts each
photon to a photoelectron with probability `p(θ, λ)` built from two *measured*
curves, then applies TTS smearing and feeds the existing front-end → ToT →
binning stack:

1. **A_eff(θ)** — absolute per-PMT effective area vs incidence angle
   (photocathode projection × glass/gel transmission × collection efficiency).
   Source: KM3NeT multi-PMT DOM paper; OMSim/Geant4 can regenerate it.
2. **QE(λ)** — quantum efficiency of the 3″ Hamamatsu R12199-02 (~25–30% peak
   near 380–400 nm), folded with glass+gel transmission vs wavelength.

K40 and dark noise **bypass** this interface — they are data-driven and inject
PE-level hits directly (k40gen samples measured rates/coincidence
probabilities; dark noise is per-PMT Poisson). This asymmetry is the whole
reason backgrounds needed no DOM response but muons do.

**Why biolum needs upgrading.** The current `biolum.py` uses a *shape-only*
acceptance (`max(0,cosθ)^1.5`, no absolute scale, no wavelength) — adequate
only because burst intensity was marginalized (log-uniform). Retrofit it onto
the real `A_eff(θ)·QE(λ)` interface once that exists.

**Wavelength matters twice:** in QE(λ) at detection, and upstream in the
λ-dependent seawater attenuation length (muon Cherenkov ~1/λ² vs biolum
~470–490 nm see different effective attenuation).

**Free closure test.** With real A_eff and QE, the K40 singles rate becomes
*predictable* from the known seawater K40 activity (~5–7 kHz/PMT). Checking
that prediction against the data-driven k40gen rate validates the absolute
scale of the entire response chain — otherwise the hardest thing to trust.

**Muon plan (decided).**
- No flux model: this is a single-DOM *dataset*, so inject **isotropically** —
  direction uniform on the sphere, impact parameter uniform in area out to a
  few attenuation lengths, energy log-uniform (≈10 GeV–100 TeV), truth stored
  so any flux can be reweighted later. (MUTE is therefore dropped.)
- **PROPOSAL** for muon transport: continuous dE/dx + stochastic cascades
  (brems/pair/photonuclear) as positioned energy depositions. v7.6.2 is
  source-only on PyPI (no wheels) → cluster source build needing a newer gcc
  module than the system gcc 8.5; budget an iteration or two (like k40gen).
- **Light:** analytic Cherenkov for scale — Frank–Tamm (~340 photons/cm in
  band), θ_C ≈ 42°, intensity ∝ 1/(r·sinθ_C)·exp(−r/λ_att); cascades as point
  sources ∝ deposited energy. Validate against a small Prometheus sample
  rather than using Prometheus as the factory. Defensible in seawater (long
  scattering length; mirrors KM3NeT's JSirene fast sim).
- Single muons only; bundles are an acknowledged approximation (MUPAGE not
  public), fine for a single-DOM trigger study.

**Recommended build order for the next phase:**
1. Response module with `A_eff(θ)` + `QE(λ)` + the **K40 closure test**.
2. Retrofit `biolum.py` onto that interface.
3. PROPOSAL source build on the cluster (the real unknown).
4. Isotropic muon injector + analytic Cherenkov light atop the response module.
5. Validation gate vs published KM3NeT direct-light / time-residual PDFs.

---

## Phase log

### Phase 0 — environment + Edge TPU toolchain (2026-06-11)
FASRC Cannon, plain `venv` on lab storage (no conda — user preference; home
dir ~90% full). Toolchain validated end-to-end via a smoke test (the go/no-go
gate): PyTorch 2.11.0+cu128 → litert-torch 0.9.1 → ai-edge-quantizer 0.7.0
`static_wi8_ai8` int8 PTQ → edgetpu_compiler 16.0.384591198 → **100% ops
mapped**. Key gotchas: default PyPI torch ships CUDA-13 runtime that the
FASRC 12.9 driver rejects (pin `+cu128`); avoid the PT2E in-converter
quantization route (CPU-fallback, ai-edge-torch #450); prefer fixed-kernel
`AvgPool2d` over `AdaptiveAvgPool2d`. See `smoke_test/RESULTS.md`.

### Phase 1 — mock pipeline (2026-06-11)
`tpu_trigger` package: three `TriggerNet` variants (plain / dilated /
depthwise temporal CNN), mock data, train + export pipeline. Sweep result:
all three train, quantize, compile at **100% TPU mapping**, int8 AUC == float
AUC to 3 decimals; dilated convs + residual ADDs confirmed to compile on the
frozen v16 compiler. See `reports/mock_pipeline_2026-06-11.md`.

### Phase 2a — KM3NeT background generation (2026-06-12)
Public-software-only generators for a single 31-PMT DOM, all emitting a common
`(t_ns, pmt, tot)` hit stream → binned to `(31, 256)` @ 8 ns:
- **K40** via k40gen (gitlab.nikhef.nl/roelaaij/k40gen). Two upstream bugs
  patched in our clone (`tools/K40GEN_PATCHES.md`): a py3.11+ build break, and
  a **real physics bug** — a pre/post-increment mismatch in
  `fill_coincidences` scrambled coincidence PMT identities (rates looked fine,
  pair angles were flat). After the fix, pair-angle log-correlation vs the
  cross_prob parameterization = **0.992**; all 5 physics gates pass.
- **Dark noise** — per-PMT Poisson + afterpulses, validated uncorrelated.
- **Bioluminescence** — directional burst (shape-only acceptance; to be
  upgraded per the design above), validated facing-hemisphere pattern.
- Composed `bg_v1.h5` (100k windows). 31-channel pipeline re-verified: **100%
  TPU mapping survives 16→31 channels**, int8 AUC == float. Placeholder-task
  AUC ≈ 0.83 (physics-limited: weak bursts add <1 hit per 2 µs window).
See `reports/backgrounds_phase2a_2026-06-12.md`. A 1M-window (10-shard) job
was prepared and then cancelled per user request; resubmit is one
`sbatch --array=0-9`.

### Event visualization (2026-06-12)
`tpu_trigger.display`: raster (network input, PMTs ring-ordered) + DOM map
(directions unrolled in azimuth/cos-zenith, marker ∝ hits). Examples in
`reports/event_displays/examples.png` show the expected morphologies (isolated
singles, time-clustered K40 coincidences on adjacent PMTs, directional biolum
bursts brightest on the facing hemisphere).

---

## Operational notes

- **Remotes:** GitHub `MiaochenJin/TPU-Trigger` is canonical (`origin`, push
  from Mac). `cluster` → lab bare repo for Mac↔cluster sync. Cluster clone has
  a `github` HTTPS remote (pull only; no ssh auth there). Keep all three in
  sync after commits.
- **Cluster paths:** code/env/tools/results on
  `/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/` (persistent);
  data/runs on `/n/netscratch/.../miaochenjin/TPU-trigger/` (purged ~90 d).
- **Activate env:** `module load python/3.12.8-fasrc01 && source
  /n/holylfs05/.../miaochenjin/envs/tpu-trigger/bin/activate`; put
  `/n/holylfs05/.../tools/edgetpu_compiler/usr/bin` on PATH. Source builds need
  `module load cmake/3.25.2-fasrc01` and `unset CONDA_EXE CONDA_PREFIX`,
  `pip install --no-build-isolation`.
- **SLURM:** account `arguelles_delgado_lab`; GPU dev on
  `arguelles_delgado_gpu_a100`; CPU work on `shared`. `module load` must be
  inside the job script (no `/etc/profile.d/modules.sh` on compute nodes).
