"""End-to-end pipeline: train -> convert -> int8 quantize -> edgetpu compile
-> fidelity evaluation, per model variant.

Usage (venv active, edgetpu_compiler on PATH, from anywhere):
    python -m tpu_trigger.pipeline --variants plain dilated depthwise \
        --epochs 5 --n-train 20000 --outdir runs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .export import compile_edgetpu, convert_float, quantize_int8, tflite_logits
from .mockdata import make_dataset
from .models import VARIANTS, make_model
from .train import auc_score, evaluate, train


def run_variant(variant, args):
    out = Path(args.outdir) / variant
    out.mkdir(parents=True, exist_ok=True)

    # 1. train
    tr = train(variant, T=args.T, n_train=args.n_train, n_val=args.n_val,
               epochs=args.epochs, snr=args.snr, seed=args.seed,
               outdir=args.outdir)
    model = make_model(variant, T=args.T)
    model.load_state_dict(torch.load(tr["ckpt"], map_location="cpu",
                                     weights_only=True))
    model.eval()

    # 2. test data (held out) + float reference
    x_te, y_te = make_dataset(args.n_test, T=args.T, snr=args.snr,
                              seed=args.seed + 2)
    torch_acc, torch_auc, torch_logit = evaluate(model, x_te, y_te, "cpu")

    # 3. convert + parity of the float tflite
    float_path = out / f"{variant}_float.tflite"
    print(f"[{variant}] converting -> {float_path}")
    convert_float(model, args.T, float_path)
    f_logits, _ = tflite_logits(float_path, x_te[:64])
    conv_err = float(np.max(np.abs(f_logits - torch_logit[:64])))

    # 4. quantize with real calibration data (training distribution)
    x_cal, _ = make_dataset(args.n_calib, T=args.T, snr=args.snr,
                            seed=args.seed + 3)
    int8_path = out / f"{variant}_int8.tflite"
    print(f"[{variant}] quantizing -> {int8_path}")
    quantize_int8(float_path, x_cal, int8_path)

    # 5. compile
    print(f"[{variant}] compiling for Edge TPU")
    comp = compile_edgetpu(int8_path, out)
    (out / "compile_stdout.txt").write_text(comp.pop("stdout"))

    # 6. int8 fidelity at deployed resolution
    q_logits, out_scale = tflite_logits(int8_path, x_te)
    scores = q_logits[:, 1] - q_logits[:, 0]
    int8_acc = float(((scores > 0).astype(np.int64) == y_te).mean())
    int8_auc = auc_score(scores, y_te)
    rng_out = float(torch_logit.max() - torch_logit.min())
    nrmse = float(np.sqrt(np.mean((q_logits - torch_logit) ** 2)) / rng_out)

    row = {
        "variant": variant, "params": tr["params"],
        "torch_acc": torch_acc, "torch_auc": torch_auc,
        "int8_acc": int8_acc, "int8_auc": int8_auc,
        "conv_err": conv_err, "logit_nrmse": nrmse,
        "logit_quant_scale": out_scale,
        "tpu_ops": comp["tpu_ops"], "cpu_compute": comp["cpu_compute"],
        "cpu_boundary": comp["cpu_boundary"], "onchip_mem": comp["onchip_mem"],
        "mapping_ok": comp["ok"], "compiled": comp["compiled"],
    }
    (out / "report.json").write_text(json.dumps(row, indent=2))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS),
                   choices=VARIANTS)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--n-calib", type=int, default=256)
    p.add_argument("--snr", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="runs")
    args = p.parse_args()

    rows = [run_variant(v, args) for v in args.variants]

    lines = [
        "| variant | params | torch acc | int8 acc | torch AUC | int8 AUC | "
        "logit NRMSE | conv err | CPU ops | on-chip mem |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n_cpu = sum(r["cpu_compute"].values()) + sum(r["cpu_boundary"].values())
        lines.append(
            f"| {r['variant']} | {r['params']:,} | {r['torch_acc']:.4f} | "
            f"{r['int8_acc']:.4f} | {r['torch_auc']:.4f} | {r['int8_auc']:.4f} | "
            f"{r['logit_nrmse']:.4f} | {r['conv_err']:.1e} | {n_cpu} | "
            f"{r['onchip_mem']} |")
    summary = "\n".join(lines)
    print("\n" + summary)
    Path(args.outdir, "summary.md").write_text(summary + "\n")

    if not all(r["mapping_ok"] for r in rows):
        bad = [r["variant"] for r in rows if not r["mapping_ok"]]
        raise SystemExit(f"FAIL: incomplete Edge TPU mapping for {bad}")
    print("\nALL VARIANTS: trained, quantized, compiled, 100% TPU-mapped.")


if __name__ == "__main__":
    main()
