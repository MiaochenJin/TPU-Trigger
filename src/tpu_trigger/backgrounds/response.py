"""Unified photon -> photoelectron response for a KM3NeT multi-PMT DOM.

Shared by every light-based source (bioluminescence, muon Cherenkov, and the
phase-2b signal) so a classifier learns physics, not per-source simulation
artifacts. Photons are delivered at the DOM as (arrival_time, travel_direction,
wavelength, fluence); this module converts them to photoelectron hits
(t_ns, pmt, tot) via two curves -- the per-PMT effective area A_eff(theta) and
the (transmission-folded) quantum efficiency QE_eff(lambda) -- plus transit-time
spread. K40 (k40gen) and dark noise BYPASS this module: they are data-driven and
inject PE-level hits directly.

Conventions
-----------
A photon travels along unit vector d_hat. PMT j with axis n_hat_j sees incidence
angle theta_j, cos(theta_j) = -d_hat . n_hat_j  (a photon moving along -n_hat
strikes the PMT facing +n_hat head-on, theta=0). For a photon sample carrying
fluence w [photons/cm^2] at wavelength lambda, the expected PEs on PMT j are

    mu_j = w * A_eff(theta_j) * QE_eff(lambda)

A_eff is an ABSOLUTE area [cm^2] (photocathode projection x angular acceptance x
collection efficiency, wavelength-independent); QE_eff is dimensionless
(photocathode QE folded with the glass+gel transmission, carrying ALL the
wavelength dependence). This split avoids double-counting the glass/gel
transmission that the design note attached to both curves. mu_j is Poisson-
sampled, arrival times are smeared by the TTS, and ToT is drawn from the SPE
Gaussian (geometry.TOT_MEAN/SIGMA) to match k40gen's hit convention.

Optical constants below are first-cut literature values (see
docs/response_calibration.md for sources); the K40 closure test
(closure_k40.py) checks their absolute scale against k40gen's singles rate.
"""

import numpy as np

from . import optical_tables as _ot
from .geometry import N_PMT, PMT_DIRS, TOT_MEAN, TOT_SIGMA

# --- quantum efficiency (3" Hamamatsu R12199-02 bialkali) --------------------
# CALIB: (lambda_nm, QE) sample points; peak ~0.28 near 390 nm. Pending the
# research pull + closure test (docs/response_calibration.md).
_QE_LAMBDA = np.array([300, 350, 380, 400, 440, 500, 550, 600, 650], dtype=float)
_QE_VALUE = np.array([0.10, 0.24, 0.28, 0.28, 0.23, 0.13, 0.06, 0.02, 0.005])

# CALIB: optical gel + Vitrovex glass transmission vs wavelength (UV cutoff
# ~300-350 nm, visible plateau ~0.92).
_TGG_LAMBDA = np.array([290, 300, 320, 350, 400, 500, 600, 700], dtype=float)
_TGG_VALUE = np.array([0.00, 0.10, 0.55, 0.85, 0.92, 0.93, 0.93, 0.92])

# Prefer the measured KM3NeT curves (OMGsim tables under external/, gitignored)
# when present; otherwise fall back to the placeholder arrays above. See
# optical_tables.py and docs/response_calibration.md.
try:
    _CURVES = _ot.load_curves()
    MEASURED_OPTICS = True
except (FileNotFoundError, KeyError, OSError):
    _CURVES = None
    MEASURED_OPTICS = False

# --- angular acceptance & absolute area --------------------------------------
# CALIB: relative angular response vs incidence angle (0 deg head-on). First cut
# = projected cos(theta) with a hard FOV cut; will be replaced by the measured
# KM3NeT 3" PMT angular-acceptance curve.
_ANG_THETA_DEG = np.array([0, 15, 30, 45, 60, 75, 90, 100], dtype=float)
_ANG_VALUE = np.cos(np.radians(np.clip(_ANG_THETA_DEG, 0, 90)))
_ANG_VALUE[_ANG_THETA_DEG > 90] = 0.0

# CALIB: peak effective area at normal incidence [cm^2] = photocathode projected
# area x collection efficiency (QE and transmission live in QE_eff). 3" PMT
# photocathode ~ 32-40 cm^2; collection eff ~0.9.
A_EFF_PEAK_CM2 = 34.0

# CALIB: single-PE transit-time spread, FWHM [ns].
TTS_FWHM_NS = 2.0
_TTS_SIGMA_NS = TTS_FWHM_NS / 2.35482


def qe_eff(lam_nm):
    """Transmission-folded quantum efficiency QE_eff(lambda), dimensionless.

    With the measured KM3NeT tables available (MEASURED_OPTICS), this is the
    real R12199-02 QE(lambda) folded with Beer-Lambert glass + gel transmission
    (14 mm glass, 2 mm gel). Otherwise it falls back to the placeholder QE x
    glass/gel plateau curves above.
    """
    lam = np.asarray(lam_nm, dtype=float)
    if MEASURED_OPTICS:
        qe_x, qe_v = _CURVES["qe"]
        qe = np.interp(lam, qe_x, qe_v, left=0.0, right=0.0)
        gx, gv = _CURVES["glass_abs_mm"]
        lx, lv = _CURVES["gel_abs_mm"]
        t_glass = _ot.transmission(lam, gx, gv, _CURVES["glass_mm"])
        t_gel = _ot.transmission(lam, lx, lv, _CURVES["gel_mm"])
        return qe * t_glass * t_gel
    qe = np.interp(lam, _QE_LAMBDA, _QE_VALUE, left=0.0, right=0.0)
    tgg = np.interp(lam, _TGG_LAMBDA, _TGG_VALUE, left=0.0, right=_TGG_VALUE[-1])
    return qe * tgg


def a_eff(theta_deg):
    """Absolute per-PMT effective area [cm^2] vs incidence angle (deg)."""
    th = np.asarray(theta_deg, dtype=float)
    ang = np.interp(th, _ANG_THETA_DEG, _ANG_VALUE, left=_ANG_VALUE[0], right=0.0)
    return A_EFF_PEAK_CM2 * ang


def angular_area_integral(n=2001):
    """int_FOV A_eff(theta) dOmega  [cm^2 . sr], over the forward hemisphere.

    The angle factor for detection from an isotropic radiance field: a photon
    field of radiance L gives single rate  R = (this integral) x int L QE_eff dl.
    """
    th = np.linspace(0.0, np.pi, n)
    a = a_eff(np.degrees(th))
    return 2.0 * np.pi * np.trapezoid(a * np.sin(th), th)


class DOMResponse:
    """Photon batch -> photoelectron hit stream (t_ns, pmt, tot)."""

    def __init__(self, tts_sigma_ns=_TTS_SIGMA_NS):
        self.tts_sigma_ns = tts_sigma_ns

    def detect(self, t_ns, d_hat, lam_nm, fluence, rng):
        """Convert a photon batch to PE hits.

        t_ns      (M,)   arrival times at the DOM
        d_hat     (M,3)  unit travel directions
        lam_nm    (M,)   wavelengths
        fluence   (M,)   photons/cm^2 carried by each sample
        Returns (t_ns, pmt, tot) int arrays, time-sorted.
        """
        t_ns = np.asarray(t_ns, dtype=float)
        d_hat = np.asarray(d_hat, dtype=float).reshape(-1, 3)
        lam_nm = np.asarray(lam_nm, dtype=float)
        fluence = np.asarray(fluence, dtype=float)

        cos_inc = np.clip(-(d_hat @ PMT_DIRS.T), -1.0, 1.0)     # (M, 31)
        theta = np.degrees(np.arccos(cos_inc))
        mu = (fluence[:, None] * a_eff(theta)) * qe_eff(lam_nm)[:, None]
        counts = rng.poisson(mu)                                # (M, 31)

        ph, pmt = np.nonzero(counts)
        if ph.size == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z.astype(np.int8), z.astype(np.int16)
        n_each = counts[ph, pmt]
        ph_pe = np.repeat(ph, n_each)
        pmt_pe = np.repeat(pmt, n_each).astype(np.int8)
        t_pe = t_ns[ph_pe] + rng.normal(0.0, self.tts_sigma_ns, ph_pe.size)
        tot = np.floor(rng.normal(TOT_MEAN, TOT_SIGMA, ph_pe.size)).astype(np.int16)

        order = np.argsort(t_pe)
        return t_pe[order].astype(np.int64), pmt_pe[order], tot[order]


assert PMT_DIRS.shape == (N_PMT, 3)
