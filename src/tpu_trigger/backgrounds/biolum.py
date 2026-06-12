"""Empirical bioluminescence burst model for a 31-PMT DOM.

Bioluminescence bursts last ms-to-seconds (ANTARES, arXiv:2107.08063), far
longer than a ~2 us trigger window: within one window a burst is a
quasi-stationary elevated Poisson rate. So instead of simulating organisms,
each burst-window draws:
  - a source direction (uniform on the sphere; organisms stream past the DOM),
  - a burst intensity I (per-PMT peak rate, log-uniform across the published
    dynamic range: tens of kHz up to ~MHz on facing PMTs),
and the per-PMT rate is  base + I * acceptance(angle between PMT and source).

Acceptance model: relative PMT response ~ max(0, cos(theta))**ACC_POWER with
a field-of-view cutoff — a standard cosine-law approximation of the 3" PMT
angular response; the directional *correlation pattern* across PMTs (bright
hemisphere facing the source) is the feature that distinguishes biolum from
K40 coincidences, and is robust to the exact exponent.
"""

import numpy as np

from .geometry import N_PMT, PMT_DIRS, TOT_MEAN, TOT_SIGMA

ACC_POWER = 1.5          # acceptance ~ cos^1.5, FOV cutoff at cos > 0
BURST_RATE_RANGE = (2e4, 1e6)   # peak per-PMT rate on a facing PMT [Hz]


def sample_burst(rng):
    """Draw (direction, per-PMT rates in Hz) for one burst window."""
    # isotropic direction
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    lo, hi = BURST_RATE_RANGE
    intensity = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    cos_t = PMT_DIRS @ v
    acc = np.where(cos_t > 0, cos_t ** ACC_POWER, 0.0)
    return v, intensity * acc


def generate_biolum(span_ns, pmt_rates_hz, seed=0):
    """One DOM window of burst light. Returns (t_ns, pmt, tot), sorted."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.asarray(pmt_rates_hz) * span_ns * 1e-9)
    t = np.concatenate([rng.uniform(0, span_ns, c) for c in counts])
    pmt = np.repeat(np.arange(N_PMT, dtype=np.int8), counts)
    order = np.argsort(t)
    t, pmt = t[order].astype(np.int64), pmt[order]
    tot = np.floor(rng.normal(TOT_MEAN, TOT_SIGMA, len(t))).astype(np.int16)
    return t, pmt, tot
