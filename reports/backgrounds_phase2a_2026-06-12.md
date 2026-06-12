# Phase 2a complete: KM3NeT background generation — 2026-06-12

All six phases of `docs/backgrounds_plan.md` executed and gated.

## A — k40gen (gitlab.nikhef.nl/roelaaij/k40gen) in the venv

pip-from-clone build (cmake 3.25 module, no conda). Two upstream bugs patched
in our clone — see `tools/K40GEN_PATCHES.md`:
1. build: `sysconfig.get_config_var('SO')` removed in py3.11+ → module
   installed under a broken filename;
2. **physics: pre/post-increment mismatch in `fill_coincidences` shifted every
   coincidence's PMT list by one slot against its hit times** — time
   clustering and total rates looked right, but pair identities were
   scrambled (flat angular distribution, impossible same-PMT pairs). Both
   worth reporting upstream.

## B — K40 validation (after the fix): PASS

- singles 7106 Hz/PMT vs 7052 expected; ToT 26.94 vs 26.936 (floor-corrected)
- fold rates match k40gen's multiplicity model (documented in
  K40GEN_PATCHES.md — `rates[1:]` is not "rate per fold")
- pair-angle log-correlation vs the cross_prob parameterization: **0.992**
- plots: `reports/k40_validation/`

## C/D — dark noise + bioluminescence: PASS

- dark: 1017 Hz/PMT vs 1020 expected; close-pair rate at accidental-only
  level (the anti-signature of K40)
- biolum: per-PMT counts track the cos^1.5 acceptance model (corr 1.000),
  light confined to the source-facing hemisphere

## E — composed dataset

`bg_v1.h5` (netscratch data/): 100k windows (31, 256) @ 8 ns, classes
0 = steady sea (K40+dark, 0.512 hits/window vs 0.52 expected),
1 = biolum burst (directional, log-uniform peak intensity 20 kHz–1 MHz,
stored per window for per-intensity analysis).

## F — 31-channel Edge TPU pipeline on real backgrounds (job 22007228, 5m47s)

| variant | params | torch AUC | int8 AUC | CPU ops | on-chip mem |
|---|---|---|---|---|---|
| plain | 83,554 | 0.8275 | 0.8270 | 0 | 108.50 KiB |
| dilated | 257,282 | 0.8278 | 0.8272 | 0 | 315.75 KiB |
| depthwise | 30,050 | 0.8266 | 0.8264 | 0 | 64.75 KiB |

**100% Edge TPU mapping survives the 16→31 channel change**; int8 AUC equals
float AUC to the third decimal for all variants.

Notes:
- AUC ~0.83 is the honest ceiling for this placeholder task: low-intensity
  bursts add <1 hit per 2 µs window and are genuinely indistinguishable from
  steady sea in a single window. A per-intensity AUC breakdown (intensity is
  stored in the dataset) is the right next analysis if this task matters
  beyond pipeline validation.
- int8 *accuracy* dips for plain/depthwise while AUC is unchanged — the fixed
  scores>0 threshold shifts slightly under quantization; operating points
  must be picked on the int8 score distribution (as planned).
- The three variants are statistically identical here — the task is
  occupancy/pattern-limited, not capacity-limited. Architecture comparisons
  should wait for signal-vs-background (phase 2b).

## Next (phase 2b)

Prometheus muon/neutrino signal with the 31-PMT splitting layer (photon
arrival directions are stored for water geometries), overlaid on these
background streams via the common (t, pmt, tot) hit-stream interface.
