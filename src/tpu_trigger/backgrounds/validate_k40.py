"""Phase-B validation gate: reproduce the standard K40 calibration analysis.

Two runs:
  A (nominal rates): per-PMT singles rate, ToT distribution, dt spectrum
    (coincidence peak over accidental floor) — validates totals.
  B (low singles, same coincidence rates): genuine accidentals drop x100,
    so 2-fold/3-fold group rates and the pair angular correlation are
    measured cleanly — validates the coincidence physics.

Notes: k40gen emits integer-floored times/ToT, so the ToT mean is shifted
by -0.5 ns relative to the underlying Gaussian.

Run: python -m tpu_trigger.backgrounds.validate_k40 --outdir reports/k40_validation
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import TOT_MEAN, TOT_SIGMA, cross_prob, pair_cos_angles
from .k40 import RATES_DEFAULT, iter_dom_streams

GAP_NS = 10  # hits closer than this are clustered into one coincidence group


def cluster_pairs(t, pmt):
    """Split a time-sorted stream into groups (dt <= GAP_NS chains).

    Returns (group_sizes, pair_pmts) where pair_pmts is the (a, b) PMT ids
    of all 2-fold groups.
    """
    if len(t) < 2:
        return np.array([], dtype=int), (np.array([]), np.array([]))
    splits = np.where(np.diff(t) > GAP_NS)[0] + 1
    bounds = np.concatenate([[0], splits, [len(t)]])
    sizes = np.diff(bounds)
    starts = bounds[:-1][sizes == 2]
    return sizes, (pmt[starts], pmt[starts + 1])


def collect(span_ns, rates, seeds):
    """Accumulate singles, group-size counts, 2-fold pair matrix, ToT, dts."""
    stats = {"n_hits": 0, "n_doms": 0, "groups": np.zeros(8),
             "pairs": np.zeros((31, 31)), "tot": [], "dt": []}
    for dom_id, (t, pmt, tot) in iter_dom_streams(span_ns, tuple(rates),
                                                  seeds=seeds):
        stats["n_doms"] += 1
        stats["n_hits"] += len(t)
        sizes, (a, b) = cluster_pairs(t, pmt)
        for s in range(2, 8):
            stats["groups"][s] += (sizes == s).sum()
        np.add.at(stats["pairs"], (np.minimum(a, b), np.maximum(a, b)), 1)
        stats["tot"].append(tot[:50])
        dt = np.diff(t)
        stats["dt"].append(dt[dt < 500])
    stats["tot"] = np.concatenate(stats["tot"]).astype(float)
    stats["dt"] = np.concatenate(stats["dt"])
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-a", type=float, default=2e7)
    ap.add_argument("--span-b", type=float, default=1e8)
    ap.add_argument("--outdir", default="reports/k40_validation")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    r_nom = RATES_DEFAULT                      # (7000, 700, 70, 0)
    r_low = (700.0,) + RATES_DEFAULT[1:]       # low-singles control

    # ---- run A: nominal rates ----
    a = collect(int(args.span_a), r_nom, seeds=(21341, 1245))
    span_s = args.span_a * 1e-9
    singles = a["n_hits"] / (a["n_doms"] * 31 * span_s)
    # hits belonging to genuine coincidences add to the per-PMT singles rate
    exp_singles = r_nom[0] + (2 * r_nom[1] + 3 * r_nom[2] + 4 * r_nom[3]) / 31
    tot_mean = a["tot"].mean() + 0.5  # undo integer floor
    print(f"[A] singles {singles:.0f} Hz/PMT (expected {exp_singles:.0f})")
    print(f"[A] ToT mean {tot_mean:.2f} (exp {TOT_MEAN}), "
          f"sigma {a['tot'].std():.2f} (exp {TOT_SIGMA})")

    # ---- run B: low-singles control ----
    b = collect(int(args.span_b), r_low, seeds=(77, 78))
    span_s_b = args.span_b * 1e-9
    dom_s = b["n_doms"] * span_s_b
    rate2 = b["groups"][2] / dom_s
    rate3 = b["groups"][3] / dom_s
    # accidental contamination of the 2-fold count (Poisson singles pairs)
    r_dom = r_low[0] * 31
    acc2 = r_dom * r_dom * GAP_NS * 1e-9
    print(f"[B] 2-fold groups {rate2:.0f} Hz/DOM "
          f"(expected {r_low[1]:.0f} genuine + {acc2:.0f} accidental)")
    print(f"[B] 3-fold groups {rate3:.1f} Hz/DOM (expected {r_low[2]:.0f})")

    # angular correlation from 2-fold pairs, per PMT pair (465 types)
    ct = pair_cos_angles()
    iu = np.triu_indices(31, k=1)
    counts = b["pairs"][iu]
    model = cross_prob(ct[iu])
    scale = counts.sum() / model.sum()
    sel = scale * model >= 5  # populated pair types only
    corr = np.corrcoef(np.log(np.maximum(counts[sel], 0.5)),
                       np.log(scale * model[sel]))[0, 1]
    print(f"[B] angular correlation (log, {sel.sum()}/465 pair types): "
          f"{corr:.4f}")

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(a["dt"], bins=100, range=(0, 500), histtype="step")
    axes[0].set(xlabel="dt consecutive hits [ns]", ylabel="pairs",
                title="run A: coincidence peak + accidental floor",
                yscale="log")
    axes[1].plot(scale * model[sel], counts[sel], "k.", ms=4)
    lim = [scale * model[sel].min() * 0.5, scale * model[sel].max() * 2]
    axes[1].plot(lim, lim, "r-", lw=1)
    axes[1].set(xlabel="expected pairs (cross_prob, scaled)",
                ylabel="measured pairs",
                title=f"run B: per-pair-type counts (log-corr {corr:.3f})",
                xscale="log", yscale="log")
    axes[2].hist(a["tot"], bins=np.arange(10, 45), histtype="step",
                 density=True, label="generated")
    x = np.linspace(10, 45, 200)
    axes[2].plot(x, np.exp(-0.5 * ((x - TOT_MEAN + 0.5) / TOT_SIGMA) ** 2)
                 / (TOT_SIGMA * np.sqrt(2 * np.pi)), "r-", lw=1,
                 label="Gaussian (floored)")
    axes[2].set(xlabel="ToT [ns]", title="ToT distribution")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "k40_validation.png", dpi=120)
    print(f"wrote {out}/k40_validation.png")

    checks = {
        "singles": abs(singles - exp_singles) / exp_singles < 0.02,
        "twofold": abs(rate2 - (r_low[1] + acc2)) / (r_low[1] + acc2) < 0.10,
        "threefold": abs(rate3 - r_low[2]) / r_low[2] < 0.20,
        "angular": corr > 0.95,
        "tot": abs(tot_mean - TOT_MEAN) < 0.3,
    }
    results = {"singles_rate_hz": float(singles),
               "twofold_rate_hz_per_dom": float(rate2),
               "threefold_rate_hz_per_dom": float(rate3),
               "angular_log_corr": float(corr),
               "tot_mean_corrected": float(tot_mean),
               "checks": checks}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("checks:", checks)
    print("K40 VALIDATION:", "PASS" if all(checks.values()) else "FAIL")
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
