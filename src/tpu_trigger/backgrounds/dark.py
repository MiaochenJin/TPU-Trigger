"""Parametric PMT dark noise for a 31-PMT DOM.

Thermal dark counts: independent Poisson per PMT (uncorrelated across PMTs —
the signature that distinguishes dark noise from K40 coincidences). Default
rate is representative of the KM3NeT 3" PMTs (O(1 kHz) thermal at the 0.3 SPE
threshold; in sea water this is subdominant to the ~7 kHz K40-dominated
singles). Afterpulses: each primary hit spawns an afterpulse with small
probability, delayed by a broad ~µs-scale distribution (ion feedback).

ToT values share the SPE Gaussian of geometry.TOT_MEAN/SIGMA (floored to int,
matching k40gen's convention).
"""

import numpy as np

from .geometry import N_PMT, TOT_MEAN, TOT_SIGMA

DARK_RATE_HZ = 1000.0       # thermal rate per PMT
AFTERPULSE_PROB = 0.02      # afterpulses per primary hit
AP_DELAY_LOGNORM = (np.log(2000.0), 0.8)   # ns; median 2 us, broad


def _tot(rng, n):
    return np.floor(rng.normal(TOT_MEAN, TOT_SIGMA, n)).astype(np.int16)


def generate_dark(span_ns, rate_hz=DARK_RATE_HZ, ap_prob=AFTERPULSE_PROB,
                  seed=0):
    """One DOM of dark noise. Returns (t_ns, pmt, tot), time-sorted."""
    rng = np.random.default_rng(seed)
    n_exp = rate_hz * span_ns * 1e-9
    counts = rng.poisson(n_exp, N_PMT)
    t = np.concatenate([rng.uniform(0, span_ns, c) for c in counts])
    pmt = np.repeat(np.arange(N_PMT, dtype=np.int8), counts)
    # afterpulses
    is_ap = rng.random(len(t)) < ap_prob
    t_ap = t[is_ap] + rng.lognormal(*AP_DELAY_LOGNORM, is_ap.sum())
    pmt_ap = pmt[is_ap]
    t = np.concatenate([t, t_ap])
    pmt = np.concatenate([pmt, pmt_ap])
    keep = t < span_ns
    t, pmt = t[keep], pmt[keep]
    order = np.argsort(t)
    t, pmt = t[order].astype(np.int64), pmt[order]
    return t, pmt, _tot(rng, len(t))
