"""Phase C+D validation gates for dark noise and bioluminescence.

Dark:  rate matches config; close-pair rate is accidental-only (the
       anti-signature of K40's genuine coincidences).
Biolum: per-PMT counts track the acceptance model toward the source
       direction; facing hemisphere dominates.

Run: python -m tpu_trigger.backgrounds.validate_noise
"""

import json
from pathlib import Path

import numpy as np

from .biolum import generate_biolum, sample_burst
from .dark import DARK_RATE_HZ, generate_dark
from .geometry import N_PMT, PMT_DIRS


def validate_dark(span_ns=int(5e9), rate=DARK_RATE_HZ):
    t, pmt, tot = generate_dark(span_ns, rate_hz=rate, seed=11)
    span_s = span_ns * 1e-9
    measured = len(t) / (N_PMT * span_s)
    exp = rate * 1.02  # afterpulses
    dt = np.diff(t)
    close = (dt <= 10).sum() / span_s
    r_dom = measured * N_PMT
    acc = r_dom * r_dom * 10e-9  # accidental-pair rate prediction
    print(f"[dark] rate {measured:.0f} Hz/PMT (expected {exp:.0f})")
    print(f"[dark] close-pair rate {close:.2f} Hz vs accidental-only "
          f"{acc:.2f} Hz (genuine coincidences would exceed this)")
    return {
        "rate": bool(abs(measured - exp) / exp < 0.03),
        "uncorrelated": bool(abs(close - acc) / acc < 0.25),
    }


def validate_biolum(n_bursts=200):
    rng = np.random.default_rng(5)
    corrs, frac_facing = [], []
    for k in range(n_bursts):
        v, rates = sample_burst(rng)
        if rates.max() * 2e-6 < 5:  # need enough hits to measure pattern
            scale = 5 / (rates.max() * 2e-6)
        else:
            scale = 1.0
        t, pmt, tot = generate_biolum(int(2e6), rates * scale, seed=1000 + k)
        counts = np.bincount(pmt, minlength=N_PMT)
        corrs.append(np.corrcoef(counts, rates)[0, 1])
        facing = PMT_DIRS @ v > 0
        frac_facing.append(counts[facing].sum() / max(1, counts.sum()))
    corr, ff = float(np.mean(corrs)), float(np.mean(frac_facing))
    print(f"[biolum] mean corr(per-PMT counts, acceptance model): {corr:.3f}")
    print(f"[biolum] fraction of light in facing hemisphere: {ff:.3f}")
    return {"pattern": bool(corr > 0.9), "hemisphere": bool(ff > 0.99)}


def main():
    checks = {**{f"dark_{k}": v for k, v in validate_dark().items()},
              **{f"biolum_{k}": v for k, v in validate_biolum().items()}}
    out = Path("reports/noise_validation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(checks, indent=2))
    print("checks:", checks)
    print("NOISE VALIDATION:", "PASS" if all(checks.values()) else "FAIL")
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
