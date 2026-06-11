# Milestone 0 results — 2026-06-11 — **GO**

The modern toolchain (PyTorch 2.12.0 → litert-torch 0.9.1 → ai-edge-quantizer
0.7.0 `static_wi8_ai8` → edgetpu_compiler 16.0.384591198) works end-to-end on
FASRC (Rocky 8, python/3.12.8-fasrc01 venv). No fallbacks needed.

## Compiler op table (run 2, with native AvgPool2d)

```
Operator                       Count      Status
FULLY_CONNECTED                1          Mapped to Edge TPU
CONV_2D                        2          Mapped to Edge TPU
AVERAGE_POOL_2D                1          Mapped to Edge TPU
TRANSPOSE                      1          Mapped to Edge TPU
```

100% mapped, single Edge TPU subgraph, 0 CPU ops (not even boundary quantize
ops — the recipe produces a fully-integer graph). Model params fit entirely in
on-chip memory (74 KiB used / 7.6 MiB available).

## Parity (32 fixed-seed inputs)

| check | value | criterion | result |
|---|---|---|---|
| float .tflite vs PyTorch, max abs err | 7.1e-08 | ≤ 1e-4 | PASS |
| int8 vs PyTorch, NRMSE | 0.0136 | ≤ 0.05 | PASS |
| int8 vs PyTorch, Pearson r | 0.99892 | > 0.99 | PASS |
| int8 vs PyTorch, max abs err | 0.0111 (~27 output LSB) | informational | — |

## Notes

- Run 1 used `AdaptiveAvgPool2d(1)`, which decomposed to TRANSPOSE+SUM — still
  mapped 100% to TPU, but prefer fixed-kernel `AvgPool2d` for real models.
- The lone TRANSPOSE comes from NCHW (torch) → NHWC (tflite) layout lowering.
- The int8 max error (~27 LSB of the output scale) is accumulated quantization
  noise across 3 quantized layers of a random-weight net; trained models with
  regularized activation ranges are expected to do better. Watch NRMSE per
  model, not absolute LSBs.
- Remaining off-cluster step: run `out/smoke_int8_edgetpu.tflite` once on the
  physical Coral device (locally) to validate the runtime half of the chain.
