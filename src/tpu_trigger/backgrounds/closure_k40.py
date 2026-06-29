"""K40 closure test: predict the single-PMT rate from first principles and
compare it to k40gen's data-driven rate.

This is the absolute-scale check on the whole response chain. K40 decays in
seawater emit Cherenkov light isotropically; folding that photon field through
A_eff(theta) and QE_eff(lambda) predicts a single-PMT counting rate that must
match the ~7 kHz that k40gen injects (and that KM3NeT measures). If it does,
the absolute normalization of A_eff x QE is trustworthy -- otherwise the hardest
thing in the chain to believe.

Chain (infinite uniform isotropic emitter):
    j(lambda)   = a * dNg/dlambda          [photons / cm^3 / s / nm]
    phi(lambda) = j(lambda) * L_abs(lambda) [photons / cm^2 / s / nm]   (scalar flux)
    L(lambda)   = phi(lambda) / (4 pi)      [photons / cm^2 / s / sr / nm] (radiance)
    R           = (int_FOV A_eff dOmega) * int L(lambda) QE_eff(lambda) dlambda

All K40/water constants are first-cut literature values (CALIB); see
docs/response_calibration.md.
"""

import numpy as np

from . import response
from .geometry import N_PMT

# --- K40 source & seawater optics (CALIB) ------------------------------------
K40_ACTIVITY_BQ_PER_L = 13.5      # beta-active K40 in seawater [decays/s/L]
N_CHERENKOV_PER_DECAY = 30.0      # Cherenkov photons/decay in [LAMBDA_LO,HI]
LAMBDA_LO, LAMBDA_HI = 300.0, 600.0   # nm, Cherenkov detection band

# water absorption length vs wavelength (CALIB: deep-sea site, peaks ~blue)
_ABS_LAMBDA = np.array([300, 350, 400, 440, 470, 500, 550, 600], dtype=float)
_ABS_LENGTH_M = np.array([8.0, 20.0, 45.0, 60.0, 55.0, 40.0, 20.0, 8.0])

K40GEN_SINGLES_HZ = 7000.0        # k40gen RATES_DEFAULT[0], the comparison target


def cherenkov_pdf(lam_nm):
    """Frank-Tamm photon spectrum shape dN/dlambda ~ 1/lambda^2, normalized to
    1 over [LAMBDA_LO, LAMBDA_HI]."""
    lam = np.asarray(lam_nm, dtype=float)
    norm = 1.0 / LAMBDA_LO - 1.0 / LAMBDA_HI            # int 1/l^2 dl
    pdf = (1.0 / lam ** 2) / norm
    return np.where((lam >= LAMBDA_LO) & (lam <= LAMBDA_HI), pdf, 0.0)


def abs_length_cm(lam_nm):
    """Seawater absorption length [cm] vs wavelength. Uses the measured KM3NeT
    NEMOWater curve when available (response.MEASURED_OPTICS), else placeholder."""
    if response.MEASURED_OPTICS:
        wx, wv = response._CURVES["water_abs_mm"]
        return np.interp(lam_nm, wx, wv, left=wv[0], right=wv[-1]) / 10.0   # mm -> cm
    return np.interp(lam_nm, _ABS_LAMBDA, _ABS_LENGTH_M, left=_ABS_LENGTH_M[0],
                     right=_ABS_LENGTH_M[-1]) * 100.0


def predict_single_rate(n=3001):
    """Predicted single-PMT rate [Hz] from the K40 Cherenkov field."""
    a_cm3 = K40_ACTIVITY_BQ_PER_L / 1000.0             # decays/s/cm^3
    lam = np.linspace(LAMBDA_LO, LAMBDA_HI, n)
    j = a_cm3 * N_CHERENKOV_PER_DECAY * cherenkov_pdf(lam)   # /cm^3/s/nm
    phi = j * abs_length_cm(lam)                            # /cm^2/s/nm
    radiance = phi / (4.0 * np.pi)                         # /cm^2/s/sr/nm
    spectral_fold = np.trapezoid(radiance * response.qe_eff(lam), lam)
    return response.angular_area_integral() * spectral_fold


def empirical_k40gen_rate(span_ns=int(1e7)):
    """Measured total hit rate per PMT from one k40gen call [Hz] (WARD only)."""
    from .k40 import generate_full_detector, N_DOMS_PER_CALL
    t, dom, pmt, tot = generate_full_detector(span_ns)
    return len(t) / (N_DOMS_PER_CALL * N_PMT) / (span_ns * 1e-9)


def main():
    pred = predict_single_rate()
    print(f"optics mode            = {'MEASURED (OMGsim tables)' if response.MEASURED_OPTICS else 'placeholder'}")
    print(f"angular area integral  = {response.angular_area_integral():.2f} cm^2.sr")
    print(f"predicted single rate  = {pred:,.0f} Hz/PMT")
    print(f"k40gen nominal singles = {K40GEN_SINGLES_HZ:,.0f} Hz/PMT")
    try:
        emp = empirical_k40gen_rate()
        print(f"k40gen measured rate   = {emp:,.0f} Hz/PMT")
    except Exception as e:                 # k40gen absent (e.g. on the Mac)
        emp = K40GEN_SINGLES_HZ
        print(f"k40gen measured rate   = (skipped: {type(e).__name__})")
    ratio = pred / emp
    print(f"ratio pred/measured    = {ratio:.2f}")
    print("CLOSURE", "PASS" if 0.5 <= ratio <= 2.0 else "NEEDS CALIBRATION",
          f"(factor {ratio:.2f}; target within 2x for first cut)")


if __name__ == "__main__":
    main()
