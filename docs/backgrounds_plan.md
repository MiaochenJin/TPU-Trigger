# Plan: KM3NeT background-noise generation (phase 2a)

Goal: generate physically credible **background** data for a single KM3NeT DOM
(31 × 3″ PMTs) using public software only — K40 decays, bioluminescence, and
PMT dark noise — as the noise classes for the Edge TPU trigger study. Signal
(muons/neutrinos via Prometheus + 31-PMT response) is deferred to phase 2b.

## Data representation (fixed once, used everywhere)

- One sample = one DOM time window: tensor `(31, T)`, T = 256 bins,
  bin width dt = 8 ns (window 2.05 µs) — both configurable; dt/T tradeoff is
  itself a study. KM3NeT L1 coincidence window is 25 ns ≈ 3 bins.
- Channel value = number of PMT hits per bin (KM3NeT readout gives (t, ToT)
  per hit; ToT encoding kept as an option in the binner, default = counts).
- Internal interchange format everywhere: a flat **hit stream**
  `(pmt_id, t_ns, tot_ns)` per DOM — all generators emit it, the binner
  consumes it, and later Prometheus-derived signal hits will be overlaid in
  the same format.

## Module layout — `src/tpu_trigger/backgrounds/`

| file | contents |
|---|---|
| `geometry.py` | 31-PMT direction table for the KM3NeT DOM (from the KM3NeT DOM paper / km3pipe), PMT angular acceptance model |
| `k40.py` | wrapper around `k40gen`: `Generators(seed1, seed2, [singles, 2f, 3f, 4f])` + `generate_k40(t0, t1, ...)` → hit stream |
| `dark.py` | parametric thermal dark counts (Poisson/PMT, uncorrelated) + afterpulses (delayed, few %, from KM3NeT 3″ PMT characterization) |
| `biolum.py` | empirical burst model: Poisson burst arrivals over a baseline, burst amplitude/duration from ANTARES (arXiv:2107.08063) & KM3NeT site studies; each burst is a directional source → per-PMT rate ∝ acceptance toward it (uses `geometry.py`); `fourth_day` integration as a later cross-check, not a dependency |
| `compose.py` | overlay hit streams, sample labeled windows, bin to `(31, T)`, write HDF5 datasets to netscratch |

## Phases (each ends with a validation gate)

**A. k40gen on the cluster.** `git clone` + `pip install ./k40gen` into the
tpu-trigger venv (C++ build; `module load gcc` + cmake as needed). Gate:
reproduce the upstream test (`Generators(seed, seed, [7000, 700, 70, 0])`,
1e8 ns) and confirm output columns/units by inspection against the test suite.
Risk: vectorized build (xsimd/vectorclass) on Rocky 8 — fallback to the
non-vectorized cmake path.

**B. K40 wrapper + physics validation.** `k40.py` + `geometry.py`. Gate:
(1) per-PMT singles rate matches the configured ~5–7 kHz; (2) genuine
coincidence rate vs fold (2-, 3-, 4-fold per DOM) matches the configured
rates and falls ~×10 per fold as in KM3NeT K40 calibration publications;
(3) coincidence pair rate vs PMT-pair opening angle shows the published
monotone decrease. Plots saved to `reports/`.

**C. Dark noise.** `dark.py`, rates from the KM3NeT PMT paper (thermal
O(0.2–1.5 kHz)/PMT + afterpulse component). Gate: rate + inter-PMT
correlation sanity (must be uncorrelated, unlike K40).

**D. Bioluminescence.** `biolum.py` empirical model. Gate: baseline-vs-burst
rate distributions and burst duration/fraction consistent with published
ANTARES/KM3NeT numbers (bursts reaching 100s of kHz per PMT, correlated
across PMTs facing the source — the directional correlation is what
distinguishes biolum from K40 coincidences for the network).

**E. Composition + dataset.** `compose.py`; classes for now:
`k40+dark` (steady-state sea) and `biolum burst` (+ mixtures). Generate a
first labeled dataset (~100k windows) on netscratch. Gate: class-conditional
rate histograms look like the per-generator validations.

**F. Close the ML loop at 31 channels.** Bump `N_CH` 16→31 in
`tpu_trigger.models` (only the first conv changes), retrain the three
variants on the new noise classes (placeholder task: burst vs steady-state),
re-run the full convert→int8→compile pipeline. Gate: 100% Edge TPU mapping
still holds at 31 channels and the int8-vs-float fidelity is unchanged.

## Validation anchors (public)

- KM3NeT K40 coincidence calibration papers (per-DOM fold rates, angular
  correlation).
- KM3NeT 3″ PMT / DOM characterization (dark rate, afterpulses, TTS, PMT
  orientations).
- ANTARES bioluminescence flash study arXiv:2107.08063; KM3NeT site
  bioluminescence/sea-current studies.

## Explicitly deferred

- Prometheus muon/neutrino signal + photon-direction → 31-PMT splitting
  (phase 2b; Prometheus water propagator already stores arrival directions).
- `fourth_day` hydrodynamic biolum model (cross-check of D, not a blocker).
- ToT-based channel encoding and dt/T optimization (after phase 2b, on the
  full signal+background task).
