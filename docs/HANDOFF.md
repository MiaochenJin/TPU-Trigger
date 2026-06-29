# TPU-trigger — handoff (2026-06-29)

Orientation for a fresh session. Read this first, then `WORK_LOG.md` for the full
chronological narrative. Branch: **`phase2b-digitization`** (pushed to `origin`
= GitHub and `cluster` = lab bare repo).

## What this project is
Trigger-level signal/noise classifier for neutrino telescopes on Google Coral
**Edge TPUs** — follow-up to *Two Watts is All You Need* (arXiv:2311.04983).
Target: a single **KM3NeT 31-PMT DOM**. The classifier input is a `(31, T)`
time-binned hit image. Backgrounds + the unified photon→PE response are built;
the **muon/neutrino signal source is the main remaining physics piece**.

## Where the project is (Phase 2b)
Phase 2b = unified DOM response + first signal-side source (bioluminescence) +
PMT digitization. Completed this phase:

1. **Response calibrated from KM3NeT OMGsim tables** — `backgrounds/response.py`
   + `backgrounds/optical_tables.py`. Measured QE (R12199-02), glass, gel, water
   curves parsed from the OMGsim Geant4 material tables, loaded at runtime from
   the gitignored `external/` clone (`MEASURED_OPTICS` flag; placeholder fallback
   stays committed so the public repo runs and carries no internal data).
2. **K40 closure** — `backgrounds/closure_k40.py`. Measured optics moved the
   predicted/measured singles ratio **0.27 → 0.34**. The remaining gap is the
   placeholder Cherenkov photon yield.
3. **OMGsim K40 ground-truth run (FASRC)** — detected singles **979 Hz/PMT**
   (QE-corrected) in a 10 m water sphere. This is **truncated**: 10 m world vs
   ~37 m QE-weighted absorption length → only ~23 % of the infinite-medium
   singles; de-truncated ≈ **4270 Hz**, consistent with the 7000 Hz reference
   once the other radioactive sources are added. The 10 m world is sized for
   **coincidence** calibration (decays within a few m), not singles. K40
   single-decay multiplicity: 97 % singles, **~3 % coincidences from r < 4.3 m**.
4. **Bioluminescence `GenType 3` in OMGsim** — a new generation mode added to the
   OMGsim source (in `external/inputs4qefit/omgsim`, gitignored): one event = one
   instantaneous **flash** of M **unpolarised** optical photons (Gaussian
   450–500 nm centre 475, random ⊥ polarisation) from a vertex sampled in the
   water. Built + run on FASRC. M=3500 tuned (~3 PE/flash); multiplicity is
   **coincidence-dominated (~70 %)**; injection range **≳ 5 m at M=3500**.
   Runtime **~5 µs / injected photon** ⇒ physically bright flashes (~10⁹ photons)
   are infeasible (~1.4 h each) and unnecessary — **cap M at the ~10⁵ saturation
   scale**; both cost (∝M) and injection range (∝√M) are governed by M.
5. **PMT digitization** — `backgrounds/digitize.py`. Faithful Python port of
   Jpp's `JPMTAnalogueSignalProcessor` **charge→ToT** model (linear 7 ns/npe,
   smooth 210 ns saturation, time-slewing, gain spread) + a **canonical
   anode-waveform generator** on a causal physical SPE (fast 3.5 ns rise, slow
   decay, 1-pe ToT anchored to 25.08 ns) + summary-stat extraction
   (`waveform_summary`: charge, peak, leading-edge, ToT).
   **Key reconciliation:** the waveform and the official ToT are bridged by the
   **CHARGE** (waveform integral → Jpp model), **NOT by thresholding** the
   waveform — the KM3NeT ToT is charge-driven (the front-end clips and widens
   with charge), not the bare anode pulse width. The waveform additionally keeps
   the true, unsaturated charge that ToT loses above ~tens of PE.

## Where things live
- **Code:** `src/tpu_trigger/backgrounds/` — `response.py`, `optical_tables.py`,
  `closure_k40.py`, `digitize.py`, plus existing `geometry.py`, `k40.py`,
  `dark.py`, `biolum.py`, `compose.py`. Event-display module: `src/tpu_trigger/display/`.
- **External KM3NeT clones (gitignored, local only, NOT committed):**
  `external/inputs4qefit/` (OMGsim Geant4 radioactive-bg sim + our biolum
  `GenType 3` edits) and `external/jpp/` (the Jpp digitization framework = the
  ToT-model source). Re-clone from `git@git.km3net.de:vkulikovskiy/inputs4qefit`
  and `git@git.km3net.de:common/jpp` (SSH key `id_ed25519_km3net` is set up).
- **Sim data (`.evt`, NOT in repo):** OMGsim outputs on FASRC
  `/n/netscratch/arguelles_delgado_lab/Everyone/miaochenjin/omgsim_{k40,biolum}/runs/`;
  local copies were in the session scratchpad (K40 `output_real1M.evt`, biolum
  `scanR2/R5/R10.evt`).
- **Figures + scripts:** `reports/figures/` (event displays, waveform/ToT sanity
  check, and their generating scripts).
- **Docs:** `docs/response_calibration.md`, this file. `WORK_LOG.md` = narrative.

## Infrastructure / how to resume
- **FASRC** (`ssh harvard`): `/usr/bin/singularity` 4.4.2 system-wide. Containers
  at `…/miaochenjin/tools/{omgsim_v2.1.4.sif (2.2 GB), Jpp_v18.0.0.sif (3.9 GB)}`.
  Modified OMGsim source + build at `…/miaochenjin/omgsim_src/` (run
  `$SRC/build/OMGsim`, NOT the container's prebuilt binary). Account
  `arguelles_delgado_lab`; partition `shared` (CPU) or `test` (fast turnaround).
  **The lab allocation is often saturated with the user's own `v7reco` array —
  use `test` for short jobs.**
- **WARD** (`ssh ward`): RTX 5090 box for GPU training (intermittently
  unreachable; FASRC is the reliable host for OMGsim).
- **git.km3net.de**: internal GitLab; SSH key set up on the Mac.

## Gotchas
- **Mac numpy is old** (no `np.trapezoid`): scripts shim `np.trapezoid = np.trapz`.
  FASRC/WARD have numpy ≥ 2.
- **`k40gen`** is WARD/FASRC-only → importing `k40.py`/`compose.py` fails on the
  Mac. The other background modules import fine.
- **OMGsim geometry quirk:** `/KM3/det/setTargetLength` *and* `setWorldLength`
  both set the water-sphere radius (`fTargetLength`, default 5 m); the biolum
  emitter-radius command is **shadowed** (same UI path registered by two
  messengers). K40 ran genuinely at 10 m. For large-R biolum a **decoupled
  emitter-radius command** is needed (`KM3DetectorConstruction.cc:118`,
  `KM3DetectorMessenger.cc:185/193`).
- **QE2 factor:** the OMGsim K40 macro uses "QE2" = 2× real QE → halve raw counts.
- **Internal-data boundary:** `external/` is gitignored — never commit/push the
  KM3NeT clones or their data to the public GitHub remote.
- The digitization **SPE pulse shape is generic** (no measured KM3NeT waveform
  exists — KM3NeT is ToT-native). Only the **ToT model + measured TTS** are
  sourced from Jpp; per-PMT calibration lives in the KM3NeT DB.

## Next steps (prioritised)
1. **Wire `digitize.py` onto the OMGsim/`response.py` PE streams** so K40/biolum/
   signal all flow PE → waveform → {ToT, charge, leading-edge, …}, then build the
   `(31, T)` training tensors with the real ToT (replacing the k40gen Gaussian-ToT
   placeholder in `geometry.py`).
2. **Waveform-vs-ToT trigger study** — train a trigger on ToT vs on the
   waveform / PE-time-resolved representation, and the in-situ "save the full
   waveform vs save the ToT summary" decision. Decide the FADC sampling
   assumption (this is a forward-looking waveform-capable-DOM study).
3. **Muon/neutrino signal source** — PROPOSAL (WARD's gcc 13 should unblock the
   build FASRC's gcc 8.5 couldn't) + analytic Cherenkov through the same response.
   This is the point of the project.
4. **Calibrate the K40 photon yield** — extract from OMGsim output, or a larger-
   world OMGsim run to de-truncate the singles, to close the 0.34 → ~1 gap.
5. **Biolum production** — cap M at saturation, radial importance sampling, and
   the decoupled emitter-radius command for the right injection range.
6. Optional: source the real R12199 SPE shape + per-PMT `JPMTParameters` (DB) to
   replace the generic SPE template.
