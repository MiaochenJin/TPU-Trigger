"""Phase-B validation gate: reproduce the standard K40 calibration analysis.

Checks, against configured rates and the k40gen parameterization:
  1. per-PMT singles rate (configured: rates[0])
  2. genuine coincidence rate per DOM from the dt peak over the accidental
     floor (configured: rates[1] for 2-fold)
  3. coincidence pair rate vs cos(opening angle) against cross_prob()
  4. ToT distribution (Gaussian 26.94 +- 2.44 ns)

Run: python -m tpu_trigger.backgrounds.validate_k40 --span-ns 2e7 \
         --outdir reports/k40_validation
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import TOT_MEAN, TOT_SIGMA, cross_prob, pair_cos_angles
from .k40 import N_DOMS_PER_CALL, RATES_DEFAULT, iter_dom_streams

COINC_WINDOW_NS = 25  # KM3NeT L1 window
PEAK_NS = 10          # genuine-coincidence peak region |dt| <= PEAK_NS
SIDE_NS = (50, 200)   # sideband for accidental-floor estimate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-ns", type=float, default=2e7)
    ap.add_argument("--rates", type=float, nargs=4, default=list(RATES_DEFAULT))
    ap.add_argument("--outdir", default="reports/k40_validation")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    span_ns = int(args.span_ns)

    n_hits = 0
    tots = []
    dts = []           # consecutive-pair dt on each DOM (any PMT pair)
    pair_counts = np.zeros((31, 31))   # close pairs (|dt|<=PEAK_NS)
    side_counts = np.zeros((31, 31))   # sideband pairs for accidental subtr.
    n_doms = 0

    for dom_id, (t, pmt, tot) in iter_dom_streams(span_ns, tuple(args.rates)):
        n_doms += 1
        n_hits += len(t)
        tots.append(tot[:50])
        dt = np.diff(t)
        dts.append(dt[dt < 500])
        # pair statistics from consecutive hits (genuine pairs are ~ns apart)
        close = dt <= PEAK_NS
        side = (dt >= SIDE_NS[0]) & (dt < SIDE_NS[1])
        for mask, counts in ((close, pair_counts), (side, side_counts)):
            a, b = pmt[:-1][mask], pmt[1:][mask]
            np.add.at(counts, (np.minimum(a, b), np.maximum(a, b)), 1)

    span_s = span_ns * 1e-9
    total_pmt_s = n_doms * 31 * span_s
    singles_rate = n_hits / total_pmt_s
    print(f"DOMs: {n_doms}, hits: {n_hits}")
    print(f"singles rate: {singles_rate:.0f} Hz/PMT "
          f"(configured {args.rates[0]:.0f} + coincidence excess)")

    # genuine 2-fold rate: close pairs minus accidental floor scaled to window
    n_close = pair_counts.sum()
    acc_per_ns = side_counts.sum() / (SIDE_NS[1] - SIDE_NS[0])
    n_genuine = n_close - acc_per_ns * PEAK_NS
    genuine_rate = n_genuine / (n_doms * span_s)
    print(f"genuine coincidence rate: {genuine_rate:.0f} Hz/DOM "
          f"(configured 2-fold {args.rates[1]:.0f} + higher folds)")

    # angular correlation: genuine pair count vs cos(opening angle)
    ct = pair_cos_angles()
    iu = np.triu_indices(31, k=1)
    acc_pairs = side_counts[iu] / (SIDE_NS[1] - SIDE_NS[0]) * PEAK_NS
    genuine_pairs = pair_counts[iu] - acc_pairs
    cts = ct[iu]
    bins = np.linspace(-1, 1, 21)
    idx = np.digitize(cts, bins) - 1
    meas = np.array([genuine_pairs[idx == b].sum() for b in range(20)])
    npairs = np.array([(idx == b).sum() for b in range(20)])
    centers = 0.5 * (bins[:-1] + bins[1:])
    model = cross_prob(centers)
    valid = (npairs > 0) & (meas > 0)
    meas_n = meas / npairs
    # compare shapes in log space where both defined
    scale = meas_n[valid].sum() / model[valid].sum()
    corr = np.corrcoef(np.log(meas_n[valid]), np.log(scale * model[valid]))[0, 1]
    print(f"angular correlation log-shape corr vs cross_prob: {corr:.4f}")

    tots = np.concatenate(tots).astype(float)
    print(f"ToT: mean {tots.mean():.2f} (exp {TOT_MEAN}), "
          f"sigma {tots.std():.2f} (exp {TOT_SIGMA})")

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    dts = np.concatenate(dts)
    axes[0].hist(dts, bins=100, range=(0, 500), histtype="step")
    axes[0].axvline(PEAK_NS, color="r", ls="--", lw=0.8)
    axes[0].set(xlabel="dt consecutive hits [ns]", ylabel="pairs",
                title="coincidence peak + accidental floor", yscale="log")
    axes[1].plot(centers[valid], meas_n[valid] / scale, "ko", ms=4,
                 label="measured / scale")
    axes[1].plot(centers, model, "r-", lw=1, label="cross_prob(ct)")
    axes[1].set(xlabel="cos(opening angle)", ylabel="genuine pairs (rel.)",
                title=f"angular correlation (log-corr {corr:.3f})",
                yscale="log")
    axes[1].legend()
    axes[2].hist(tots, bins=np.arange(10, 45), histtype="step", density=True)
    x = np.linspace(10, 45, 200)
    axes[2].plot(x, np.exp(-0.5 * ((x - TOT_MEAN) / TOT_SIGMA) ** 2)
                 / (TOT_SIGMA * np.sqrt(2 * np.pi)), "r-", lw=1)
    axes[2].set(xlabel="ToT [ns]", title="ToT vs Gaussian(26.94, 2.44)")
    fig.tight_layout()
    fig.savefig(out / "k40_validation.png", dpi=120)
    print(f"wrote {out}/k40_validation.png")

    results = {
        "span_ns": span_ns, "n_doms": n_doms,
        "singles_rate_hz": float(singles_rate),
        "genuine_coinc_rate_hz_per_dom": float(genuine_rate),
        "angular_log_corr": float(corr),
        "tot_mean": float(tots.mean()), "tot_sigma": float(tots.std()),
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))

    ok = (abs(singles_rate - args.rates[0]) / args.rates[0] < 0.05
          and abs(genuine_rate - sum(args.rates[1:])) / sum(args.rates[1:]) < 0.15
          and corr > 0.9
          and abs(tots.mean() - TOT_MEAN) < 0.5)
    print("K40 VALIDATION:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
