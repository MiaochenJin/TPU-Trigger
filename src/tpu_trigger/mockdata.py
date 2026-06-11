"""Synthetic multi-PMT trigger data: 16 correlated time series per event.

Noise events: band-limited Gaussian noise with a common-mode (correlated)
component, plus Poisson "dark count" pulses on individual channels.
Signal events: noise + a coincident transient pulse on a random subset of
channels, with per-channel amplitude spread and small time jitter.

The discriminating structure is multi-channel coincidence + pulse shape;
isolated dark pulses can be as large as signal pulses, so amplitude alone is
insufficient — the network has to learn coincidence.
"""

import numpy as np

from .models import N_CH


def _pulse(T, t0, amp, tau):
    t = np.arange(T, dtype=np.float32) - t0
    out = np.zeros(T, dtype=np.float32)
    m = t > 0
    out[m] = amp * (t[m] / tau) * np.exp(1.0 - t[m] / tau)
    return out


def _smooth(x, w=3):
    k = np.ones(w, dtype=np.float32) / w
    return np.stack([np.convolve(row, k, mode="same") for row in x])


def make_noise(T, rng):
    common = 0.6 * rng.standard_normal((1, T)).astype(np.float32)
    indep = rng.standard_normal((N_CH, T)).astype(np.float32)
    x = _smooth(common + indep)
    for _ in range(rng.poisson(2.0)):  # dark counts
        ch = rng.integers(N_CH)
        x[ch] += _pulse(T, rng.uniform(0, T),
                        rng.uniform(2.0, 6.0), rng.uniform(3, 8))
    return x


def add_signal(x, T, rng, snr):
    n_hit = rng.integers(5, N_CH + 1)
    chans = rng.choice(N_CH, size=n_hit, replace=False)
    t0 = rng.uniform(0.1 * T, 0.8 * T)
    tau = rng.uniform(4, 10)
    base = rng.uniform(0.7, 1.3) * snr
    for ch in chans:
        amp = base * (0.5 + rng.exponential(0.5))
        jitter = rng.uniform(-2, 2)
        x[ch] += _pulse(T, t0 + jitter, amp, tau)


def make_dataset(n, T=256, snr=2.0, p_signal=0.5, seed=0):
    """Returns x of shape (n, N_CH, T) float32 and labels (n,) int64."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < p_signal).astype(np.int64)
    x = np.empty((n, N_CH, T), dtype=np.float32)
    for i in range(n):
        x[i] = make_noise(T, rng)
        if y[i]:
            add_signal(x[i], T, rng, snr)
    return x, y
