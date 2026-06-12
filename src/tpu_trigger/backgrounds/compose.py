"""Compose background hit streams into labeled (31, T) trigger windows.

Classes:
  0 steady-state sea: K40 (k40gen) + dark noise
  1 bioluminescence burst: steady + directional elevated rate

At nominal rates a 2 us window holds ~0.4 K40 hits on the whole DOM — most
steady windows are near-empty, which is the true operating condition of a
trigger. Burst separability depends on the sampled intensity (weak bursts are
genuinely indistinguishable in a single window); class-1 windows therefore
also carry the sampled intensity for later per-intensity analysis.
"""

import numpy as np

from .biolum import sample_burst
from .dark import AFTERPULSE_PROB, DARK_RATE_HZ
from .geometry import N_PMT
from .k40 import RATES_DEFAULT, iter_dom_streams

T_DEFAULT = 256
DT_NS_DEFAULT = 8


def bin_stream(t, pmt, n_windows, T, dt_ns):
    """Bin a single-DOM stream into (n_windows, 31, T) hit counts."""
    window_ns = T * dt_ns
    w = t // window_ns
    keep = w < n_windows
    w, t, pmt = w[keep], t[keep], pmt[keep]
    b = (t % window_ns) // dt_ns
    out = np.zeros((n_windows, N_PMT, T), dtype=np.uint8)
    np.add.at(out, (w.astype(int), pmt.astype(int), b.astype(int)), 1)
    return out


def make_arrays(n_windows, T=T_DEFAULT, dt_ns=DT_NS_DEFAULT, p_burst=0.5,
                rates=RATES_DEFAULT, dark_rate=DARK_RATE_HZ, seed=0):
    """Returns x (n, 31, T) uint8, y (n,) int64, intensity (n,) float32."""
    rng = np.random.default_rng(seed)
    window_ns = T * dt_ns

    # K40 base: carve consecutive windows out of independent DOM streams.
    # One k40gen call yields 2070 DOM-streams of span_ns each. k40gen's
    # buffer preallocation misbehaves for very short spans (heap overrun in
    # fill_coincidences), so never request less than 10 ms per call.
    per_dom = max(1, int(np.ceil(n_windows / 2070)))
    span_ns = max(per_dom * window_ns, int(1e7))
    per_dom_avail = span_ns // window_ns
    chunks = []
    n_left = n_windows
    for dom_id, (t, pmt, tot) in iter_dom_streams(
            span_ns, rates, seeds=(int(rng.integers(2**30)),
                                   int(rng.integers(2**30)))):
        take = min(per_dom_avail, n_left)
        if take <= 0:
            break
        chunks.append(bin_stream(t, pmt, take, T, dt_ns))
        n_left -= take
    x = np.concatenate(chunks)[:n_windows]

    # dark noise: uniform Poisson per bin (afterpulse correlation negligible
    # at these occupancies: ~0.06 hits / window / DOM)
    lam = dark_rate * (1 + AFTERPULSE_PROB) * dt_ns * 1e-9
    x = x + rng.poisson(lam, x.shape).astype(np.uint8)

    # biolum bursts on a random subset
    y = (rng.random(n_windows) < p_burst).astype(np.int64)
    intensity = np.zeros(n_windows, dtype=np.float32)
    for i in np.where(y == 1)[0]:
        _, pmt_rates = sample_burst(rng)
        intensity[i] = pmt_rates.max()
        lam_pmt = pmt_rates[:, None] * dt_ns * 1e-9 * np.ones((1, T))
        x[i] = np.minimum(x[i].astype(int) + rng.poisson(lam_pmt), 255)
    return x, y, intensity


def write_h5(path, x, y, intensity, meta):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x, compression="gzip")
        f.create_dataset("y", data=y)
        f.create_dataset("intensity", data=intensity)
        for k, v in meta.items():
            f.attrs[k] = v


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    ap.add_argument("--dt-ns", type=int, default=DT_NS_DEFAULT)
    ap.add_argument("--p-burst", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x, y, inten = make_arrays(args.n, T=args.T, dt_ns=args.dt_ns,
                              p_burst=args.p_burst, seed=args.seed)
    write_h5(args.out, x, y, inten,
             {"T": args.T, "dt_ns": args.dt_ns, "p_burst": args.p_burst,
              "seed": args.seed, "classes": "0=steady(k40+dark) 1=biolum"})
    print(f"wrote {args.out}: x{x.shape} mean-occupancy "
          f"steady={x[y == 0].mean():.4f} burst={x[y == 1].mean():.4f}")


if __name__ == "__main__":
    main()
