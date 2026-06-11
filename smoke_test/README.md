# Milestone 0 smoke test — toolchain go/no-go gate

Tests the decisive risk of the modern stack: `edgetpu_compiler` is frozen at
v16 (2021), while `litert-converter` emits current TFLite flatbuffers. Nothing
else in this project proceeds until this passes.

Pipeline: `TinyTriggerNet` (PyTorch, input `(1, 60, 12, 12)`, conv/relu/pool/fc
only) → `litert_torch.convert` → float `.tflite` → `ai_edge_quantizer` with the
`static_wi8_ai8()` recipe (int8 weights + activations, 128 calibration samples)
→ `edgetpu_compiler -s` → parity check via the `ai_edge_litert` CPU interpreter.

## Run

```bash
module load python/3.12.8-fasrc01
source /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/envs/tpu-trigger/bin/activate
export PATH=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/tools/edgetpu_compiler/usr/bin:$PATH
python smoke_test/convert_and_compile.py
```

## Pass criteria

1. `edgetpu_compiler` exits 0 and produces `out/smoke_int8_edgetpu.tflite`.
2. **100% of compute ops "Mapped to Edge TPU"** — at most 2 boundary
   `QUANTIZE`/`DEQUANTIZE` ops on CPU.
3. Parity over 32 fixed-seed inputs: max |int8 output − PyTorch float output|
   ≤ 3× the output tensor's quantization scale, Pearson r > 0.99.

## If it fails (in order — don't build fallbacks preemptively)

1. Customize the quantizer recipe: per-tensor (not per-channel) weights,
   symmetric activations.
2. Try the PT2E in-converter quantization route (known-bad per ai-edge-torch
   issue #450, but worth one shot if the recipe route fails differently).
3. Last resort: a second small env pinned to old TF, exporting via
   ONNX → onnx2tf → TFLiteConverter full-int8.

Record every outcome (pass or fail, with the compiler's op table) in
`RESULTS.md`.
