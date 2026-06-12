"""Event displays for single-DOM (31, T) trigger windows.

Two views per event:
  - raster: the literal network input, PMT (ordered by ring, top to bottom)
    vs time bin — coincidences appear as vertically aligned hits among
    geometric neighbors;
  - DOM map: PMT directions unrolled in (azimuth, cos zenith), marker area
    proportional to hits — biolum bursts light up one hemisphere, K40
    coincidences light up small clusters of adjacent PMTs.

Usage:
  python -m tpu_trigger.display --h5 <dataset.h5> --outdir reports/event_displays
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .backgrounds.geometry import N_PMT, PMT_DIRS

# ring-ordered PMT permutation: top of the DOM first (descending z, then phi)
_phi = np.arctan2(PMT_DIRS[:, 1], PMT_DIRS[:, 0])
RING_ORDER = np.lexsort((_phi, -PMT_DIRS[:, 2]))
_ring_z = PMT_DIRS[RING_ORDER, 2]
RING_BOUNDS = np.where(np.abs(np.diff(_ring_z)) > 1e-3)[0] + 0.5


def plot_window(x, title="", dt_ns=8, ax_pair=None):
    """x: (31, T) counts. Draws raster + DOM map onto a (ax1, ax2) pair."""
    if ax_pair is None:
        _, ax_pair = plt.subplots(1, 2, figsize=(11, 3.2),
                                  gridspec_kw={"width_ratios": [2, 1]})
    ax1, ax2 = ax_pair
    T = x.shape[1]

    xr = x[RING_ORDER]
    im = ax1.imshow(xr, aspect="auto", interpolation="nearest",
                    cmap="inferno", origin="upper",
                    extent=[0, T * dt_ns / 1e3, N_PMT - 0.5, -0.5],
                    vmin=0, vmax=max(1, xr.max()))
    for b in RING_BOUNDS:
        ax1.axhline(b, color="gray", lw=0.4, alpha=0.6)
    ax1.set(xlabel="time [µs]", ylabel="PMT (ring-ordered, top→bottom)",
            title=title)
    plt.colorbar(im, ax=ax1, label="hits/bin", fraction=0.04)

    counts = x.sum(1)
    sizes = 30 + 600 * counts / max(1, counts.max())
    sc = ax2.scatter(np.degrees(_phi), PMT_DIRS[:, 2], s=sizes, c=counts,
                     cmap="inferno", vmin=0, vmax=max(1, counts.max()),
                     edgecolors="gray", linewidths=0.5)
    for i in np.where(counts > 0)[0]:
        ax2.annotate(str(int(counts[i])),
                     (np.degrees(_phi[i]), PMT_DIRS[i, 2]),
                     fontsize=7, ha="center", va="center", color="white")
    ax2.set(xlabel="PMT azimuth [deg]", ylabel="PMT cos(zenith)",
            title=f"DOM map ({int(counts.sum())} hits)",
            xlim=(-200, 200), ylim=(-1.15, 0.75))
    plt.colorbar(sc, ax=ax2, label="hits", fraction=0.05)
    return ax_pair


def pick_examples(x, y, intensity, n_hits):
    """Choose displays: quiet + coincidence steady windows, weak/mid/bright
    bursts. Returns list of (index, title)."""
    ex = []
    steady = np.where(y == 0)[0]
    quiet = steady[n_hits[steady] == 1]
    if len(quiet):
        ex.append((quiet[0], "steady sea (single K40/dark hit)"))
    # a genuine K40 coincidence = several hits within ~25 ns (3 bins),
    # not just a high total count: score by rolling 3-bin DOM-summed hits
    prof = x[steady].sum(1)  # (n_steady, T) hits per bin
    roll3 = prof[:, :-2] + prof[:, 1:-1] + prof[:, 2:]
    score = roll3.max(1)
    if score.max() >= 3:
        i = steady[int(np.argmax(score))]
        ex.append((i, f"steady sea (K40 coincidence: "
                      f"{int(score.max())} hits in 24 ns)"))
    bursts = np.where(y == 1)[0]
    if len(bursts):
        for q, name in ((0.25, "weak"), (0.7, "medium"), (0.98, "bright")):
            tgt = np.quantile(intensity[bursts], q)
            i = bursts[np.argmin(np.abs(intensity[bursts] - tgt))]
            ex.append((i, f"biolum burst ({name}, "
                          f"peak {intensity[i]/1e3:.0f} kHz)"))
    return ex


def main():
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--outdir", default="reports/event_displays")
    ap.add_argument("--n-random", type=int, default=0,
                    help="additionally dump N random windows")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5) as f:
        x, y = f["x"][:], f["y"][:]
        intensity = f["intensity"][:] if "intensity" in f else np.zeros(len(x))
        dt_ns = int(f.attrs.get("dt_ns", 8))
    n_hits = x.reshape(len(x), -1).sum(1)

    examples = pick_examples(x, y, intensity, n_hits)
    fig, axes = plt.subplots(len(examples), 2, figsize=(11, 3.3 * len(examples)),
                             gridspec_kw={"width_ratios": [2, 1]})
    for (i, title), ax_pair in zip(examples, np.atleast_2d(axes)):
        plot_window(x[i], f"[{i}] {title}", dt_ns, ax_pair)
    fig.tight_layout()
    p = out / "examples.png"
    fig.savefig(p, dpi=130)
    print(f"wrote {p}")

    rng = np.random.default_rng(0)
    for k in range(args.n_random):
        i = int(rng.integers(len(x)))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.3),
                                 gridspec_kw={"width_ratios": [2, 1]})
        cls = "burst" if y[i] else "steady"
        plot_window(x[i], f"[{i}] {cls}", dt_ns, axes)
        fig.tight_layout()
        fig.savefig(out / f"window_{i}_{cls}.png", dpi=130)
        plt.close(fig)
    if args.n_random:
        print(f"wrote {args.n_random} random windows to {out}")


if __name__ == "__main__":
    main()
