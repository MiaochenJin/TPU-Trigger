"""Milestone 0 smoke test: PyTorch -> litert-torch -> full-int8 TFLite ->
edgetpu_compiler -> parity check.

Run on a FASRC login node (CPU-only) with the tpu-trigger venv active and
edgetpu_compiler on PATH:

    python smoke_test/convert_and_compile.py

Pass criteria (see smoke_test/README.md):
  1. compiler exits 0 and writes smoke_int8_edgetpu.tflite
  2. all compute ops "Mapped to Edge TPU" (<=2 boundary QUANTIZE ops on CPU)
  3. conversion parity: float .tflite vs PyTorch, max |err| <= 1e-4
  4. quantization parity: int8 .tflite vs PyTorch over 32 fixed-seed inputs,
     NRMSE (rmse / output range) <= 5% and Pearson r > 0.99
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

    # conversion-only parity: float tflite must match torch almost exactly
    f_outputs, _ = run_tflite(float_path, ref_inputs)
    conv_err = float(np.max(np.abs(f_outputs - ref_outputs)))
    print(f"float-tflite vs torch max |err| = {conv_err:.2e}")

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
    qt.quantize(calib_result, serialize_to_path=str(int8_path))
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

    # 5. quantization parity: int8 tflite (CPU interpreter) vs PyTorch float.
    # Judged on normalized RMSE and correlation — absolute LSB-level bounds
    # are not meaningful after error accumulation across quantized layers,
    # especially for a random-weight net with untrained activation ranges.
    print("== parity check ==")
    q_outputs, out_scale = run_tflite(int8_path, ref_inputs)
    out_range = float(ref_outputs.max() - ref_outputs.min())
    max_err = float(np.max(np.abs(q_outputs - ref_outputs)))
    rmse = float(np.sqrt(np.mean((q_outputs - ref_outputs) ** 2)))
    nrmse = rmse / out_range
    r = float(np.corrcoef(q_outputs.reshape(-1), ref_outputs.reshape(-1))[0, 1])
    print(f"output range {out_range:.4f}, quant scale {out_scale}")
    print(f"max |err| = {max_err:.5f}, rmse = {rmse:.5f}, "
          f"nrmse = {nrmse:.4f} (tol 0.05), pearson r = {r:.5f} (>0.99)")
    ok_convert = conv_err <= 1e-4
    ok_parity = nrmse <= 0.05 and r > 0.99

    print()
    print(f"op mapping:        {'PASS' if ok_mapping else 'FAIL'}")
    print(f"conversion parity: {'PASS' if ok_convert else 'FAIL'}")
    print(f"int8 parity:       {'PASS' if ok_parity else 'FAIL'}")
    if not (ok_mapping and ok_convert and ok_parity):
        sys.exit("SMOKE TEST FAILED")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
