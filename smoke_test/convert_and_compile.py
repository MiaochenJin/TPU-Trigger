"""Milestone 0 smoke test: PyTorch -> litert-torch -> full-int8 TFLite ->
edgetpu_compiler -> parity check.

Run on a FASRC login node (CPU-only) with the tpu-trigger venv active and
edgetpu_compiler on PATH:

    python smoke_test/convert_and_compile.py

Pass criteria (see smoke_test/README.md):
  1. compiler exits 0 and writes smoke_int8_edgetpu.tflite
  2. all compute ops "Mapped to Edge TPU" (<=2 boundary QUANTIZE ops on CPU)
  3. int8-interpreter output vs PyTorch float: max |err| <= 3x output quant
     scale and Pearson r > 0.99 over 32 fixed-seed inputs
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import litert_torch
from ai_edge_quantizer import quantizer, recipe
from ai_edge_litert.interpreter import Interpreter

sys.path.insert(0, str(Path(__file__).parent))
from tiny_net import INPUT_SHAPE, make_model

OUT = Path(__file__).parent / "out"
N_REF = 32
N_CALIB = 128

# Boundary (de)quantize ops are allowed on CPU; everything else must map.
BOUNDARY_OPS = {"QUANTIZE", "DEQUANTIZE"}
MAX_CPU_BOUNDARY_OPS = 2


def get_signature(tflite_path):
    """Return (signature_key, [input_names]) of a converted model."""
    interp = Interpreter(model_path=str(tflite_path))
    sigs = interp.get_signature_list()
    key = next(iter(sigs))
    return key, list(sigs[key]["inputs"])


def run_tflite(tflite_path, inputs):
    """Run a (possibly quantized) tflite model on float NCHW inputs."""
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    results = []
    for x in inputs:
        x_t = x.astype(np.float32)
        if inp["dtype"] in (np.int8, np.uint8):
            scale, zp = inp["quantization"]
            x_t = np.clip(np.round(x_t / scale + zp),
                          np.iinfo(inp["dtype"]).min,
                          np.iinfo(inp["dtype"]).max).astype(inp["dtype"])
        interp.set_tensor(inp["index"], x_t.reshape(inp["shape"]))
        interp.invoke()
        y = interp.get_tensor(out["index"]).astype(np.float32)
        if out["dtype"] in (np.int8, np.uint8):
            scale, zp = out["quantization"]
            y = (y - zp) * scale
        results.append(y.reshape(-1))
    out_scale = out["quantization"][0] if out["dtype"] in (np.int8, np.uint8) else None
    return np.stack(results), out_scale


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    # 1. reference model + float outputs
    model = make_model(seed=0)
    ref_inputs = [rng.standard_normal(INPUT_SHAPE).astype(np.float32) for _ in range(N_REF)]
    with torch.no_grad():
        ref_outputs = np.stack(
            [model(torch.from_numpy(x)).numpy().reshape(-1) for x in ref_inputs]
        )

    # 2. convert to float tflite
    print("== converting with litert_torch ==")
    edge_model = litert_torch.convert(model, (torch.zeros(*INPUT_SHAPE),))
    float_path = OUT / "smoke_float.tflite"
    edge_model.export(str(float_path))
    print(f"wrote {float_path}")

    # 3. full-int8 PTQ via ai-edge-quantizer
    print("== quantizing (static_wi8_ai8) ==")
    sig_key, input_names = get_signature(float_path)
    print(f"signature '{sig_key}', inputs {input_names}")
    calib_data = {
        sig_key: [
            {input_names[0]: rng.standard_normal(INPUT_SHAPE).astype(np.float32)}
            for _ in range(N_CALIB)
        ]
    }
    qt = quantizer.Quantizer(str(float_path))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    calib_result = qt.calibrate(calib_data)
    int8_path = OUT / "smoke_int8.tflite"
    qt.quantize(calib_result).export_model(str(int8_path))
    print(f"wrote {int8_path}")

    # 4. compile for Edge TPU and parse the op-mapping table
    print("== compiling with edgetpu_compiler ==")
    if shutil.which("edgetpu_compiler") is None:
        sys.exit("FAIL: edgetpu_compiler not on PATH")
    proc = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", str(OUT), str(int8_path)],
        capture_output=True, text=True,
    )
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        sys.exit(f"FAIL: edgetpu_compiler exit code {proc.returncode}")

    tpu_ops, cpu_compute, cpu_boundary = {}, {}, {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].isdigit():
            op, count, status = parts[0], int(parts[1]), " ".join(parts[2:])
            if "Mapped to Edge TPU" in status:
                tpu_ops[op] = count
            elif op in BOUNDARY_OPS:
                cpu_boundary[op] = count
            else:
                cpu_compute[op] = count
    print(f"TPU ops: {tpu_ops}")
    print(f"CPU boundary ops: {cpu_boundary}")
    print(f"CPU compute ops: {cpu_compute}")

    compiled_path = OUT / "smoke_int8_edgetpu.tflite"
    ok_mapping = (
        compiled_path.exists()
        and not cpu_compute
        and sum(cpu_boundary.values()) <= MAX_CPU_BOUNDARY_OPS
        and tpu_ops
    )

    # 5. parity: int8 tflite (CPU interpreter) vs PyTorch float
    print("== parity check ==")
    q_outputs, out_scale = run_tflite(int8_path, ref_inputs)
    max_err = float(np.max(np.abs(q_outputs - ref_outputs)))
    r = float(np.corrcoef(q_outputs.reshape(-1), ref_outputs.reshape(-1))[0, 1])
    tol = 3 * out_scale if out_scale else 0.1
    print(f"max |err| = {max_err:.5f} (tol {tol:.5f}), pearson r = {r:.5f}")
    ok_parity = max_err <= tol and r > 0.99

    print()
    print(f"op mapping: {'PASS' if ok_mapping else 'FAIL'}")
    print(f"parity:     {'PASS' if ok_parity else 'FAIL'}")
    if not (ok_mapping and ok_parity):
        sys.exit("SMOKE TEST FAILED")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
