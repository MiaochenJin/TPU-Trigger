# Mock-data pipeline sweep — 2026-06-11

Full end-to-end run of `python -m tpu_trigger.pipeline` (train → litert-torch
convert → ai-edge-quantizer full-int8 → edgetpu_compiler v16 → fidelity eval)
for all three TriggerNet variants, as SLURM job 21599428 on
`arguelles_delgado_gpu_a100` (A100 MIG, total wall time 4m03s).

Task: binary signal-vs-noise classification on mock data — 16 correlated time
series (T=256), signal = coincident transient pulse on ≥5 channels over
correlated noise + single-channel dark pulses (snr=2.0). 20k train / 4k val /
4k test events, 5 epochs, Adam 1e-3.

## Results

| variant | params | torch acc | int8 acc | torch AUC | int8 AUC | logit NRMSE | conv err | CPU ops | on-chip mem |
|---|---|---|---|---|---|---|---|---|---|
| plain | 80,194 | 0.9273 | 0.9280 | 0.9762 | 0.9766 | 0.0670 | 1.9e-06 | 0 | 97.75 KiB |
| dilated | 252,482 | 0.9520 | 0.9525 | 0.9884 | 0.9887 | 0.0832 | 1.9e-06 | 0 | 307.00 KiB |
| depthwise | 26,690 | 0.9153 | 0.9187 | 0.9663 | 0.9664 | 0.0899 | 2.4e-06 | 0 | 54.00 KiB |

All variants: **100% of ops mapped to the Edge TPU, single subgraph, zero CPU
ops** (not even boundary quantize ops). int8 accuracy/AUC match float to the
third decimal — quantization is free at this fidelity level.

## Op mapping (compiled)

- plain: CONV_2D ×4, PAD ×4, AVERAGE_POOL_2D, FULLY_CONNECTED, TRANSPOSE
- dilated: CONV_2D ×8, ADD ×5 (residuals), PAD ×3, AVERAGE_POOL_2D, FULLY_CONNECTED, TRANSPOSE
- depthwise: DEPTHWISE_CONV_2D ×3, CONV_2D ×4, PAD ×4, AVERAGE_POOL_2D, FULLY_CONNECTED, TRANSPOSE

Settles the open questions empirically: **dilated convolutions and residual
ADDs compile and map fully** on edgetpu_compiler v16 via the litert-torch
path, as do depthwise convs.

## Trigger-threshold resolution

Logit quantization scales (threshold granularity at deployment): plain 0.0172,
dilated 0.0163, depthwise 0.0227 — ~1/256 of each logit range. ROC evaluation
in the pipeline is done on dequantized int8 logits, i.e. at deployed
resolution.

## Takeaways

- dilated wins on separation (AUC 0.989) at 3× the params and on-chip memory
  (still only 307 KiB of 7.6 MB — far from the cache limit).
- depthwise is 10× smaller than dilated with AUC 0.966 — the latency-critical
  candidate.
- All artifacts: `/n/holylfs05/.../results/TPU-trigger/mock_pipeline/`
  (compiled `_edgetpu.tflite` per variant + per-variant report.json).
- Mock task is a placeholder: real per-variant comparisons should wait for
  phase-2 data (real multi-PMT waveforms); the deliverable here is the
  validated pipeline, not the AUC numbers.
