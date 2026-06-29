# TPU-trigger — work log

Trigger-level signal/noise classification for neutrino telescopes on Google
Coral Edge TPUs. Follow-up to *Two Watts is All You Need* (arXiv:2311.04983).
Detailed results live in `reports/`; phase plans in `docs/`; this file is the
chronological narrative + design decisions not derivable from the code.

---

## Status — 2026-06-26

**Platform:** phase-2b work was on **FASRC Cannon** — the KM3NeT **OMGsim**
Geant4 chain (radioactive-background sim, run in Singularity), with code edited
locally; tagged **[FASRC]**. WARD remains the recommended GPU/training host.

**Done (phase 2b):** the unified photon→PE response is now *calibrated against
measured optics* — OMGsim's Geant4 material tables (R12199-02 QE, glass, gel,
NEMOWater absorption) drive `response.py`/`closure_k40.py`, moving the K40
closure ratio **0.27 → 0.34** from real optics alone, and OMGsim's
`singleDOM_OMGsim.detx` independently confirms `geometry.py` PMT directions. Ran
OMGsim K40 on FASRC as an absolute-scale ground truth (**979 Hz/PMT** detected,
truncated by the 10 m water world; ≈ 4,270 Hz de-truncated) and established that
the 10 m world is sized for K40 *coincidence*, not singles, calibration. First
signal-side source added: a **bioluminescence flash mode in OMGsim (GenType 3)**,
built and validated, coincidence-dominated as expected of a bright point source.

**Next up:** the muon signal source atop the response module (PROPOSAL build,
better on WARD's gcc 13); porting Jpp's analytic PE→ToT digitization to Python
(`backgrounds/digitize.py`) to replace the k40gen Gaussian-ToT placeholder, plus
a matched waveform generator on Jpp's pulse shape; and a decoupled OMGsim
emitter-radius command for clean large-water / small-emitter biolum runs (the
current command is shadowed). See the Phase 2b entry below.

---

## Status — 2026-06-16

**Platform:** TPU-trigger now runs on **two platforms** — FASRC Cannon (cluster)
and the **WARD** workstation (`ssh ward`), the latter newly stood up end-to-end:
background generation + **GPU training (RTX 5090, sm_120)** + Edge TPU compile,
all on one box with no SLURM queue. Per the cross-project convention, every log
entry is tagged **[WARD]** (workstation) or **[FASRC]** (cluster); phases 0–2a
below were all **[FASRC]**. Runbook in `WORKSTATION_SETUP.md`; details in the
"Workstation (WARD) enablement" entry under the phase log.

**Next up (unchanged):** the unified photon→PE response interface (phase-2b
starting point). WARD is now the recommended host for it — gcc 13 is expected to
ease the PROPOSAL source build that FASRC's gcc 8.5 blocked.

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

### Workstation (WARD) enablement (2026-06-16) [WARD]
Stood up the full pipeline on the WARD workstation (Ubuntu 24.04, Threadripper
7970X 64t, RTX 5090), mirroring the AtmNuDataFit no-global-installs pattern:
project-local `.venv` from system Python 3.12.3 (bootstrapped `--without-pip` +
`get-pip.py`, since Ubuntu's python lacks `ensurepip`), the same
`env/requirements.txt` as FASRC, plus `pip install -e .`. New scripts:
`env.sh` (activation), `env/setup_env_ward.sh` (idempotent from-scratch standup);
runbook in `WORKSTATION_SETUP.md`. **Three validations all PASS:** (1) **GPU** —
`torch 2.11.0+cu128` runs a real sm_120 kernel on the RTX 5090 (capability 12,0),
so training no longer needs the FASRC A100 queue; the cu128 pin works under the
CUDA-13 driver (backward-compatible). (2) **k40gen** built against gcc 13 — two
new platform patches beyond the FASRC pair (`tools/K40GEN_PATCHES.md` #4
missing `<stdexcept>`, #5 disable the Catch2 v2 unit tests that don't compile on
glibc 2.39); end-to-end via the project wrapper gives 7,162 Hz/PMT (10 ms full
reference detector), in band. (3) **Smoke test** — the v16 edgetpu_compiler runs
natively (self-contained wrapper, no Singularity) and reproduces **5/5 ops mapped
to Edge TPU, 1 subgraph, int8 parity r=0.99892**, matching FASRC on a different
OS/glibc. WARD ↔ GitHub sync directly (GitHub SSH works there). **Notable:**
WARD's gcc 13 should unblock the phase-2b PROPOSAL build that FASRC's gcc 8.5
couldn't do; the only thing WARD can't do is the on-device Coral benchmark (USB
device location TBD).

### Phase 2b — OMGsim response calibration + bioluminescence source (2026-06-26) [FASRC]
First signal-side step on top of the unified photon→PE response: calibrate the
response against *measured* KM3NeT optics, then add the first signal source
(bioluminescence) to the simulation chain. All compute on FASRC Cannon (OMGsim
Geant4 runs); code edited locally.

**Cloned the KM3NeT OMGsim chain.** `git@git.km3net.de:vkulikovskiy/inputs4qefit`
(V. Kulikovskiy) cloned into `external/` — **gitignored** (internal KM3NeT data
must not reach the public GitHub remote; `external/` added to `.gitignore`),
behind a new SSH key for `git.km3net.de`. It is KM3NeT's radioactive-background
sim: **OMGsim** (Geant4 10.01 app, run via the `omgsim_v2.1.4.sif` Singularity
container) for decay→Cherenkov→optical-transport→photocathode, then **Jpp**
(`Jpp_v18.0.0.sif`) for digitization/trigger/QE-fit. Pulled the `omgsim`
submodule (source + `common/data` optical tables + `singleDOM_OMGsim.detx`).

**Response model calibrated from the OMGsim optical tables.** New
`src/tpu_trigger/backgrounds/optical_tables.py` parses the OMGsim Geant4 material
PROPERTY tables — R12199-02 QE (stored as `QE2` = 2×real), AntaresGlass,
`WackerSilGel612_A100B67` gel, NEMOWater absorption. `response.py:qe_eff` now
folds the measured QE with Beer–Lambert glass(14 mm)+gel(2 mm) transmission when
the external tables are present (`MEASURED_OPTICS` flag); a documented placeholder
fallback stays committed so the public repo runs everywhere and carries no
internal data. `closure_k40.py` uses the measured NEMOWater absorption. **The K40
closure ratio moves 0.27 → 0.34** (predicted singles 1,915 → 2,382 Hz/PMT) from
the real optics alone — the remaining gap is dominated by the placeholder
Cherenkov photon yield. New `docs/response_calibration.md`. **Geometry
independently validated:** the OMGsim `singleDOM_OMGsim.detx` PMT directions are
identical to `geometry.py` `PMT_DIRS`, confirming k40gen's geometry.

**Ran OMGsim K40 on FASRC — the absolute-scale ground truth.** OMGsim K40 (1e6
decays, single 31-PMT DOM, NEMOWater, 10 m sphere) on `shared`: detected
single-PMT rate **979 Hz/PMT** (QE2-corrected; raw 1958). This is **truncated** —
the 10 m water world captures only ~23% of the infinite-medium singles
(QE-weighted absorption length ≈ 37 m); de-truncated ≈ **4,270 Hz**, broadly
consistent with the 7,000 Hz k40gen reference once truncation and the four other
radioactive sources (we ran only seawater K40) are accounted for. **Key insight:**
the 10 m world is sized for K40 *coincidence* calibration (decays within ~a few
m), not the singles rate — singles is the wrong observable to close OMGsim
against. K40 single-decay multiplicity: 97.1% singles, **2.9% coincidences (m≥2),
all from r < 4.3 m**.

**Added a bioluminescence generation mode to OMGsim (GenType 3).** Edited
`KM3PrimaryGeneratorAction.{hh,cc}` + `KM3RunMessenger.{hh,cc}` (all additions
marked `// TPU-trigger`): one event = one **instantaneous flash** = M
**unpolarised** optical photons (Gaussian wavelength 450–500 nm, centre 475, with
**random polarisation ⊥ momentum** — the production injection path set none, so
reused the `RandomPolarization` idiom from `tests/MottaTest.cxx`) emitted from a
single vertex sampled uniformly in the water `Target`. Reuses the `XP`/`Type K40`
output and the unchanged propagation + DOM-response chain. New
`/KM3/biolum/Nphotons` command; macro `scripts/DOMbiolum.mac`. **Built clean
against the container's Geant4 and validated** — photons propagate and are
detected (so random-polarisation reception works), 2 m emitter sphere confirmed.
Tuned **M = 3500** (mean **3.2 PE/flash**). Single-flash multiplicity is
**coincidence-dominated: ~70% of hit flashes have m≥2** (vs K40's 3%) — a bright
point source. **Injection range ≳ 5 m at M = 3500** (coincidences fill both the
R = 2 m and R = 5 m spheres; no cutoff within 5 m).

**Runtime feasibility for a variable-brightness biolum source.** Measured Geant4
throughput ≈ **5 µs per injected optical photon** (range 3.9–5.6; ≈ 2e5
photons/s/core), linear and ~independent of R and M. K40 ≈ **520 µs/decay** (~100
photon-equivalents). Per flash = M × 5 µs → M = 1e5 (~DOM saturation) = 0.5 s,
**M = 1e9 (physical) = 1.4 h/flash (infeasible)**. **Decision:** cap M at the
saturation scale (~1e5) — the DOM response is M-independent above saturation, so
simulating physical brightness is both infeasible and unnecessary; brighter
flashes are represented by the saturated response with the true M stored for
reweighting. Both cost (∝ M) and injection range (∝ √M) are governed by M.

**Geometry note (for future runs).** `/KM3/det/setTargetLength` and
`/KM3/det/setWorldLength` both alias `SetTargetLength` (the water `Target` is a
sphere of radius `fTargetLength`, default 5 m; the World box is 1.5×). The biolum
emitter-radius command is **shadowed** (the same UI path is registered by two
messengers), so emitters fill the water `Target`. K40 genuinely ran at 10 m
(analysis unaffected); the biolum R = 10 scan point stayed at 5 m. A decoupled
emitter-radius command is needed for clean large-R / small-emitter-in-large-water
runs.

**Cloned the Jpp digitization framework and characterised it** (local — Mac
code exploration, no cluster compute). Shallow-cloned KM3NeT's Jpp
(`git@git.km3net.de:common/jpp`) into `external/jpp` (**gitignored, 87 MB**);
git.km3net.de SSH access already set up. Found the digitization in
`software/JDetector/`: `JPMTSignalProcessorInterface` (the PE→hit pipeline),
`JPMTAnalogueSignalProcessor` (the realistic model), `JPMTParameters`
(constants), wrapped by `JTimeslice/JEventTimesliceWriter` (the tool the
inputs4qefit chain used). **Key finding: Jpp digitizes PE→ToT analytically and
produces NO sampled waveform** — the analog pulse exists only as closed-form
functions; ToT (leading-edge time + width) is the native, final product.
Pipeline: PE times → relative QE + TTS (from the *measured* transit-time
distribution) per PE → merge PEs within one rise-time → sample charge (gain 1.0,
gainSpread 0.4, 5% under-amplified) → discriminator threshold (0.24 pe) → emit
(leading-edge time with **time-slewing**, ToT) → merge overlapping ToT pulses.
Pulse model = Gaussian rising edge (riseTime ≈ 7.24 ns) + exponential decay
tail, amplitude ∝ charge; **ToT(npe)** is linear at **7 ns/npe** in the
high-charge regime then **smoothly saturates toward 210 ns**
(`ToT = sat·tot/√(tot²+sat²)`); threshold-band hits get ToT ~ Gaussian(4.5 ±
1.5 ns). The model is fully invertible (`getNPE(tot)`, charge probability).

**Consequence / decision for reconstruction.** ToT is the official native
product; full waveforms are absent from both OMGsim and Jpp, so a waveform model
is entirely ours — but can be made consistent with the official chain by reusing
Jpp's exact pulse shape (threshold(our-waveform) == Jpp ToT). ToT saturates at
210 ns (≈ a few tens of PE), so the waveform-over-ToT reconstruction gain is
concentrated in the **pile-up regime** (bright bioluminescence flashes — the
saturation we capped M at — and bright signal) plus fine multi-photon timing
(muon direction). Two-path plan adopted: (1) **port Jpp's PE→ToT model to
Python** (`backgrounds/digitize.py`, from `JPMTAnalogueSignalProcessor` +
`JPMTParameters`) — replaces the k40gen Gaussian-ToT placeholder and serves as
the threshold step of the waveform path; (2) build a **waveform generator** on
the same pulse shape + sampling/noise, verified against the Jpp ToT. These feed
the trigger study: a ToT-only trigger vs a waveform algorithm that decides
in-situ whether to save the full waveform or only the ToT.

---

## Operational notes

- **Remotes:** GitHub `MiaochenJin/TPU-Trigger` is canonical (`origin`, push
  from Mac). `cluster` → lab bare repo for Mac↔cluster sync. Cluster clone has
  a `github` HTTPS remote (pull only; no ssh auth there). Keep all three in
  sync after commits.
- **WARD workstation** (`ssh ward` = `hideon@128.103.100.27`): project at
  `/home/hideon/Desktop/Projects/TPU-trigger`, everything in-project (`.venv/`,
  `tools/k40gen`, `tools/edgetpu_compiler`; no global installs). GitHub SSH works
  there, so WARD ↔ GitHub directly (no lab bare repo). Activate: `source env.sh`.
  Standup from scratch: `bash env/setup_env_ward.sh`. Full runbook:
  `WORKSTATION_SETUP.md`.
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
