# DOM response calibration & the K40 closure test

How the optical-response constants in `backgrounds/response.py` are sourced, and
how the K40 closure test (`backgrounds/closure_k40.py`) checks their absolute
scale. Status as of 2026-06-25.

## What the closure test is

The closure test is the one *independent* check on the absolute scale of the
DOM response chain. Seawater K40 activity is a known, measured quantity, so a
correctly normalized response model must **predict** the K40 single-PMT counting
rate from first principles — no tuning. That prediction is compared against the
rate k40gen injects, which matters because every other light source
(bioluminescence, muon Cherenkov, the phase-2b signal) flows through the same
`A_eff(θ) × QE_eff(λ)`, and K40 is the only one with an independent ground truth.

### The reference number (what we reproduce)

The comparison target is **7000 Hz/PMT**, from `backgrounds/k40.py`:

    RATES_DEFAULT = (7000.0, 700.0, 70.0, 0.0)   # singles, 2-/3-/4-fold [Hz]

These are k40gen's reference rates, consistent with published KM3NeT K40
calibration (the prototype DOM measures a ~7–8 kHz single-PMT baseline; the K40
part is ~5–7 kHz). So the reference is *data-driven*: it encodes the measured
K40 singles rate. `closure_k40.empirical_k40gen_rate()` reproduces it by actually
running k40gen (7162 Hz/PMT measured, WARD); on hosts without k40gen the closure
falls back to the 7000 Hz nominal.

### The prediction chain (the steps we take to reproduce it)

Infinite uniform isotropic K40 emitter in seawater:

1. **activity** `a` [decays/s/cm³] from seawater K40 specific activity
2. × **photons/decay** × Cherenkov spectrum (Frank–Tamm, ∝1/λ²) → emission
   density `j(λ)` [photons/cm³/s/nm]
3. × **absorption length** `L_abs(λ)` → scalar photon flux `φ(λ)` [/cm²/s/nm]
4. ÷ 4π → **radiance** `L(λ)` [/cm²/s/sr/nm]
5. fold with **QE_eff(λ)** and integrate over λ
6. × **∫_FOV A_eff(θ) dΩ** (angular effective-area integral) → rate [Hz/PMT]

Pass criterion: predicted/measured within 2× (0.5–2.0) for a first cut.

## Optical inputs: KM3NeT OMGsim tables

The measured curves come from the KM3NeT-internal **`inputs4qefit/omgsim`** repo
(V. Kulikovskiy / C. Hugon), cloned under `external/` (gitignored — internal data
must not reach the public GitHub remote). `backgrounds/optical_tables.py` parses
the Geant4 material `PROPERTY` tables at runtime; `response.py` falls back to
documented placeholder curves when the clone is absent (`MEASURED_OPTICS` flag).

| Quantity | OMGsim table | Notes |
|---|---|---|
| QE(λ) | `KM3MatPMT3inchesQE2.dat` → `EFFICIENCY` | stored as **QE2 = real QE × 2** (oversampling trick); we apply ×0.5. Real peak 0.282 @ 390 nm |
| angular acceptance | same file → `ANGULAR_EFFICIENCY` | measured scan; **not yet wired in** (still cos-law) |
| glass transmission | `KM3MatGlassOM.dat` → `AntaresGlass` ABSLENGTH | Erlangen 2019 measurement; Beer–Lambert over 14 mm |
| gel transmission | `KM3MatOpticalGel.dat` → `WackerSilGel612_A100B67` | KM3NeT gel mix; over 2 mm (`DistSpherePMT`) |
| seawater absorption | `KM3MatWater.dat` → `NEMOWater` ABSLENGTH | 67.5 m @ 440 nm |
| DOM geometry | `singleDOM_OMGsim.detx` | 31 PMT directions — **identical** to `geometry.py` PMT_DIRS (independent validation of k40gen geometry) |

Geant4's default length unit is mm, so ABSLENGTH values are in mm.
Layer thicknesses from `KM3OMDOMQE2.dat`: `GlassThickness=14`, `DistSpherePMT=2`,
gel `WackerSilGel612_A100B67`, reflector ring (expansion cone, 48.33°).

## Calibration history

| Stage | Predicted | ratio pred/7000 | What changed |
|---|---|---|---|
| Placeholder constants | 1,915 Hz | 0.27 | first-cut literature guesses |
| **Measured optics** (this pass) | 2,382 Hz | **0.34** | real QE(λ) + glass/gel/water tables |
| target | ~7,000 Hz | ~1.0 | — |

The measured optics raised QE_eff across the blue band (e.g. 500 nm: 0.19 vs the
placeholder 0.12) but only closed part of the gap. The remaining ~2.9× is
dominated by two levers still on placeholders:

1. **Photon yield per decay** (`N_CHERENKOV_PER_DECAY = 30`). OMGsim does not use
   a constant — it fires Geant4 `ion 19 40` decays and generates Cherenkov from
   the real continuous β-spectrum (endpoint 1.31 MeV). The Geant4-consistent
   in-band yield is ~100–150, i.e. roughly the whole remaining factor. Next step:
   compute it analytically from the β-spectrum + Frank–Tamm, or extract from the
   OMGsim output ROOT files (`sftp.km3net.de/data/k40/DOM*_QE2.root`).
2. **Angular acceptance** `A_eff(θ)` (bare cos-law + hard 90° cut). The measured
   `ANGULAR_EFFICIENCY` is flatter/wider and the expansion-cone reflector adds
   collection; wiring it in (and an absolute area from the OMGsim geometry) is
   the other lever.

## Not reused (by design)

OMGsim is a full Geant4 optical Monte Carlo (3-step Motta–Schönert photocathode
model, Petzold scattering, ray-traced refraction). We do **not** reimplement that
transport in Python — overkill for a single-DOM trigger dataset. We reuse its
*inputs* (the tables above) and, for the photon-yield / angular-acceptance
ground truth, its *outputs* (ROOT files) or a targeted run of its `AA/` angular-
acceptance module on WARD.
