"""PyTorch -> float tflite -> full-int8 tflite -> edgetpu_compiler -> eval.

Same validated chain as smoke_test/convert_and_compile.py, generalized:
litert_torch.convert, ai_edge_quantizer static_wi8_ai8 with real calibration
data, edgetpu_compiler v16, and fidelity metrics computed at int8 resolution.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

import litert_torch
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer, recipe

BOUNDARY_OPS = {"QUANTIZE", "DEQUANTIZE"}


def convert_float(model, T, float_path):
    model.eval()
    sample = (torch.zeros(1, 16, 1, T),)
    edge_model = litert_torch.convert(model, sample)
    edge_model.export(str(float_path))


def quantize_int8(float_path, calib_x, int8_path):
    """calib_x: (N, 16, T) float32 representative samples."""
    interp = Interpreter(model_path=str(float_path))
    sigs = interp.get_signature_list()
    sig_key = next(iter(sigs))
    input_name = sigs[sig_key]["inputs"][0]
    calib = {
        sig_key: [
            {input_name: x[None, :, None, :].astype(np.float32)}
            for x in calib_x
        ]
    }
    qt = quantizer.Quantizer(str(float_path))
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    qt.quantize(qt.calibrate(calib), serialize_to_path=str(int8_path))


def compile_edgetpu(int8_path, outdir):
    """Run edgetpu_compiler; return op-mapping report dict."""
    if shutil.which("edgetpu_compiler") is None:
        raise RuntimeError("edgetpu_compiler not on PATH")
    proc = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", str(outdir), str(int8_path)],
        capture_output=True, text=True,
    )
    tpu_ops, cpu_boundary, cpu_compute, mem = {}, {}, {}, None
    for line in proc.stdout.splitlines():
        if line.startswith("On-chip memory used"):
            mem = line.split(":")[1].strip()
        parts = line.split()
        if len(parts) >= 3 and parts[1].isdigit():
            op, count, status = parts[0], int(parts[1]), " ".join(parts[2:])
            if "Mapped to Edge TPU" in status:
                tpu_ops[op] = count
            elif op in BOUNDARY_OPS:
                cpu_boundary[op] = count
            else:
                cpu_compute[op] = count
    compiled = Path(outdir) / (Path(int8_path).stem + "_edgetpu.tflite")
    ok = (proc.returncode == 0 and compiled.exists() and not cpu_compute
          and sum(cpu_boundary.values()) <= 2 and bool(tpu_ops))
    return {"ok": ok, "returncode": proc.returncode, "tpu_ops": tpu_ops,
            "cpu_boundary": cpu_boundary, "cpu_compute": cpu_compute,
            "onchip_mem": mem, "compiled": str(compiled),
            "stdout": proc.stdout}


def tflite_logits(tflite_path, x):
    """Run a (possibly int8) tflite model on (N, 16, T) float data.

    Returns (logits (N, 2) float, output quant scale or None).
    """
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    logits = []
    for xi in x:
        xt = xi[None, :, None, :].astype(np.float32)
        if inp["dtype"] in (np.int8, np.uint8):
            scale, zp = inp["quantization"]
            info = np.iinfo(inp["dtype"])
            xt = np.clip(np.round(xt / scale + zp),
                         info.min, info.max).astype(inp["dtype"])
        interp.set_tensor(inp["index"], xt.reshape(inp["shape"]))
        interp.invoke()
        y = interp.get_tensor(out["index"]).astype(np.float32)
        if out["dtype"] in (np.int8, np.uint8):
            scale, zp = out["quantization"]
            y = (y - zp) * scale
        logits.append(y.reshape(-1))
    out_scale = (out["quantization"][0]
                 if out["dtype"] in (np.int8, np.uint8) else None)
    return np.stack(logits), out_scale
