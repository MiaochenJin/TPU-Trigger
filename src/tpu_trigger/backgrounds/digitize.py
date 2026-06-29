"""KM3NeT PMT digitization: photoelectrons -> time-over-threshold (ToT) hits, and
the analog waveform they are derived from.

Faithful port of Jpp's analog front-end model (`JDETECTOR::JPMTAnalogueSignalProcessor`
+ `JPMTParameters`, KM3NeT `common/jpp`), which is the model the official chain
(`JEventTimesliceWriter`) uses to turn MC photoelectrons into DAQ ToT hits.

The analog pulse is modelled as a Gaussian rising edge (sigma = riseTime_ns)
joined to an exponential decay tail; its amplitude is proportional to the
collected charge. A discriminator at `threshold` [npe] produces, per pulse, a
leading-edge time (with charge-dependent time slewing) and a time-over-threshold.
ToT(npe) is the above-threshold pulse width, which becomes linear (slope ns/npe)
at high charge and then saturates smoothly toward `saturation` ns. Jpp forms NO
sampled waveform -- ToT is its native product. We add a waveform generator
(`make_waveform`) built on a separate, PHYSICAL SPE pulse (causal: fast rise +
slow decay). The waveform and the ToT are bridged by the CHARGE (waveform
integral -> Jpp charge->ToT model), NOT by thresholding the waveform: the KM3NeT
ToT is charge-driven (the front-end clips and widens with charge), not the bare
anode pulse width. The waveform keeps the true, unsaturated charge ToT discards.

Constants verified against the Jpp source: TIME_OVER_THRESHOLD_NS = 25.08 ns
(JCalibration.hh), getTH0() = 0.1, two-PE resolution Tmin = 1 ns. By
construction ToT(1 pe) = 25.08 ns before saturation.

Provenance: the ToT model + parameters are Jpp's (nominal defaults; per-PMT
calibration lives in the KM3NeT DB). TTS is a Gaussian approximation of Jpp's
MEASURED transit-time distribution (JPMTTransitTimeProbability, sigma ~2 ns). The
SPE pulse SHAPE is GENERIC -- no measured KM3NeT waveform exists (ToT-native) --
fast rise ~ R12199 datasheet class, decay solved so the 1-pe ToT matches the
nominal. QE is applied upstream (OMGsim / response.py), so the input here is the
already-accepted PE arrival times.
"""

import numpy as np

TIME_OVER_THRESHOLD_NS = 25.08   # JCalibration.hh -- ToT of a nominal 1 pe pulse
TH0 = 0.1                        # JPMTAnalogueSignalProcessor::getTH0()
TMIN_NS = 1.0                    # two-PE resolution (JPMTSignalProcessorInterface::getTmin)


class PMTToTModel:
    """Jpp JPMTAnalogueSignalProcessor: charge npe <-> ToT, and the pulse shape.

    Defaults are JPMTParameters() defaults. All "npe" are in photoelectron units.
    """

    def __init__(self, riseTime_ns=7.24, threshold=0.24, thresholdBand=0.12,
                 gain=1.0, gainSpread=0.4, PunderAmplified=0.05,
                 mean_ns=4.5, sigma_ns=1.5, slope=7.0, saturation=210.0,
                 slewing=True, tts_sigma_ns=1.9,
                 spe_rise_ns=3.5, spe_decay_ns=None):
        self.riseTime_ns = riseTime_ns
        self.threshold = threshold
        self.thresholdBand = thresholdBand
        self.gain = gain
        self.gainSpread = gainSpread
        self.PunderAmplified = PunderAmplified
        self.mean_ns = mean_ns          # threshold-band ToT mean
        self.sigma_ns = sigma_ns        # threshold-band ToT sigma
        self.slope = slope              # ns/npe in the linear regime
        self.saturation = saturation    # ns, smooth ToT saturation
        self.slewing = slewing
        # TTS: Gaussian approx of the measured Jpp transit-time distribution
        # (JPMTTransitTimeProbability, main peak ~4-5 ns FWHM -> sigma ~2 ns).
        self.tts_sigma_ns = tts_sigma_ns
        # waveform SPE pulse: causal, fast rise (R12199 datasheet-class) + slow
        # decay; decay solved so the 1-pe pulse's ToT = the nominal value (below).
        # Distinct from the ToT-model riseTime_ns above.
        self.spe_rise_ns = spe_rise_ns
        self.spe_decay_ns = spe_decay_ns
        self._configure()
        self._configure_spe()

    # --- configure(): match Gaussian/exponential and find linearisation point ---
    def _configure(self):
        rt, th, tot0 = self.riseTime_ns, self.threshold, TIME_OVER_THRESHOLD_NS
        y = -np.log(th)
        a, b, c = y, rt * np.sqrt(2.0 * y) - tot0, 0.5 * rt * rt
        Q = b * b - 4.0 * a * c
        self.decayTime_ns = ((-b + np.sqrt(Q)) / (2.0 * a)) if Q > 0.0 else (-b / (2.0 * a))
        x = rt / self.decayTime_ns
        self.t1 = rt * x
        self.y1 = np.exp(-0.5 * x * x)
        # start of linearisation x1: where d(ToT_unsat)/dnpe == slope (Jpp bisects
        # getDerivative(x)*slope == 1, the same continuous-derivative condition).
        def dtot_dnpe(npe):
            return rt / (npe * np.sqrt(2.0 * np.log(npe / th))) + self.decayTime_ns / npe
        lo, hi = 1.0, 1.0e4
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if dtot_dnpe(mid) > self.slope:   # derivative decreases with npe
                lo = mid
            else:
                hi = mid
        self.x1 = 0.5 * (lo + hi)

    def _configure_spe(self):
        """Configure the waveform SPE pulse (causal bi-exponential). If
        spe_decay_ns is None, solve the decay so a single 1-pe pulse is above
        threshold for the nominal 1-pe ToT (TIME_OVER_THRESHOLD_NS)."""
        tr = self.spe_rise_ns
        def _norm_peak(td):
            tp = (tr * td / (td - tr)) * np.log(td / tr)        # peak time
            return np.exp(-tp / td) - np.exp(-tp / tr), tp
        def _one_pe_tot(td):
            self.spe_decay_ns = td
            self._spe_norm, self._spe_peak_ns = _norm_peak(td)
            t = np.arange(0.0, 500.0, 0.02)
            a = t[self.pulse_shape(t) > self.threshold]
            return (a.max() - a.min()) if a.size else 0.0
        if self.spe_decay_ns is None:
            lo, hi = tr * 1.01, 80.0
            for _ in range(60):
                td = 0.5 * (lo + hi)
                if _one_pe_tot(td) < TIME_OVER_THRESHOLD_NS:
                    lo = td
                else:
                    hi = td
            self.spe_decay_ns = 0.5 * (lo + hi)
        self._spe_norm, self._spe_peak_ns = _norm_peak(self.spe_decay_ns)

    # --- pulse geometry (single pulse of amplitude `npe`) ---
    def rise_time(self, npe, th):
        """Time from threshold `th` to the peak (Gaussian leading edge) [ns]."""
        return self.riseTime_ns * np.sqrt(2.0 * np.log(npe / th))

    def decay_time(self, npe, th):
        """Time from peak back down to threshold `th` (exp tail, else Gaussian) [ns]."""
        if npe * self.y1 > th:
            return self.decayTime_ns * (np.log(npe / th) - np.log(self.y1))
        return self.riseTime_ns * np.sqrt(2.0 * np.log(npe / th))

    def threshold_domain(self, npe):
        if npe > self.threshold:
            return 2          # ABOVE_THRESHOLD
        if npe > self.threshold - self.thresholdBand:
            return 1          # THRESHOLDBAND
        return 0              # BELOW_THRESHOLD

    def _tot_unsat(self, npe):
        """Above-threshold pulse width before saturation [ns]."""
        th = self.threshold
        if npe * self.y1 <= th:
            return 2.0 * self.rise_time(npe, th)                # Gaussian + Gaussian
        if npe <= self.x1:
            return self.rise_time(npe, th) + self.decay_time(npe, th)  # Gaussian + exp
        tot1 = self.rise_time(self.x1, th) + self.decay_time(self.x1, th)
        return tot1 + self.slope * (npe - self.x1)              # linear

    def apply_saturation(self, tot):
        s = self.saturation
        return s / np.sqrt(tot * tot + s * s) * tot

    def remove_saturation(self, tot):
        s = self.saturation
        return np.where(tot < s, s / np.sqrt(np.maximum(s * s - tot * tot, 1e-30)) * tot, np.inf)

    def tot(self, npe, rng=None):
        """Time-over-threshold [ns] for a pulse of charge `npe`. Scalar."""
        dom = self.threshold_domain(npe)
        if dom == 0:
            return 0.0
        if dom == 1:                                            # threshold band: stochastic
            r = rng if rng is not None else np.random
            return float(r.normal(self.mean_ns, self.sigma_ns))
        return float(self.apply_saturation(self._tot_unsat(npe)))

    def npe_from_tot(self, tot_ns):
        """Inverse: number of photo-electrons from ToT (linear-regime inversion,
        Jpp getNPE). Approximate for the curved low-charge part."""
        return 1.0 + (self.remove_saturation(np.asarray(tot_ns, float)) - TIME_OVER_THRESHOLD_NS) / self.slope

    def leading_edge_offset(self, npe):
        """Charge-dependent rise to threshold = time slewing [ns]. Zero at 1 pe."""
        if not self.slewing:
            return 0.0
        th, tb = self.threshold, self.thresholdBand
        if self.threshold_domain(npe) == 1:
            ref = th - tb
            return ((self.rise_time(npe, TH0) - self.rise_time(npe, ref)) -
                    (self.rise_time(1.0, TH0) - self.rise_time(1.0, ref))) + self.mean_ns
        return ((self.rise_time(npe, TH0) - self.rise_time(npe, th)) -
                (self.rise_time(1.0, TH0) - self.rise_time(1.0, th)))

    def gain_spread(self, NPE):
        return np.sqrt(NPE * self.gain) * self.gainSpread

    def random_charge(self, NPE, rng):
        """Collected charge [npe] for NPE photo-electrons (gain spread +
        under-amplification mixture), resampled non-negative."""
        if self.PunderAmplified <= 0.0:
            mu, sigma = NPE * self.gain, self.gain_spread(NPE)
            q = rng.normal(mu, sigma)
            while q < 0.0:
                q = rng.normal(mu, sigma)
            return q
        # binomial mixture over k under-amplified PEs (Jpp inverse-transform sampling)
        while True:
            X, s, w, k = rng.random(), 0.0, (1.0 - self.PunderAmplified) ** NPE, 0
            for k in range(NPE + 1):
                s += w
                if s > X:
                    break
                w *= ((NPE - k) / (k + 1.0)) * self.PunderAmplified / (1.0 - self.PunderAmplified)
            fs = self.gainSpread * self.gainSpread
            mu = (NPE - k) * self.gain + k * fs * self.gain
            sigma = np.sqrt(mu) * self.gain_spread(1)
            q = rng.normal(mu, sigma)
            if q >= 0.0:
                return q

    def pulse_shape(self, dt_ns):
        """CAUSAL single-PE analog pulse, unit peak, as a function of time since
        the PE arrival [ns]: zero before the hit, fast bi-exponential rise
        (spe_rise_ns), slow decay (spe_decay_ns). This is the physical waveform
        pulse (NOT the ToT-model riseTime_ns/decayTime_ns)."""
        dt = np.asarray(dt_ns, float)
        tr, td = self.spe_rise_ns, self.spe_decay_ns
        y = (np.exp(-dt / td) - np.exp(-dt / tr)) / self._spe_norm
        return np.where(dt >= 0.0, np.clip(y, 0.0, None), 0.0)


def digitize_pmt(pe_times_ns, model, rng):
    """One PMT: PE arrival times -> list of (leading_edge_t_ns, tot_ns) ToT hits.

    Mirrors JPMTSignalProcessorInterface::operator(): TTS-smear each PE, time-sort,
    merge PEs within one rise-time into a combined signal, sample its charge,
    apply the discriminator threshold, then merge overlapping ToT pulses.
    """
    t = np.asarray(pe_times_ns, float)
    if t.size == 0:
        return []
    if model.tts_sigma_ns > 0.0:
        t = t + rng.normal(0.0, model.tts_sigma_ns, t.size)
    t.sort()

    pulses = []
    i = 0
    while i < t.size:                                  # cluster PEs within riseTime_ns
        j = i + 1
        while j < t.size and t[j] < t[i] + model.riseTime_ns:
            j += 1
        N = j - i
        q = model.random_charge(N, rng)
        if model.threshold_domain(q) > 0:
            pulses.append((t[i] + model.leading_edge_offset(q), model.tot(q, rng)))
        i = j

    # merge overlapping ToT pulses (leading edge of first, trailing of last)
    pulses.sort()
    merged = []
    for lead, tot in pulses:
        if merged and lead < merged[-1][0] + merged[-1][1] + TMIN_NS:
            l0, t0 = merged[-1]
            merged[-1] = (l0, max(t0, lead + tot - l0))
        else:
            merged.append((lead, tot))
    return merged


def make_waveform(pe_times_ns, model, rng, t0_ns, dt_ns, n_samples, noise_pe=0.0):
    """The CANONICAL object: the sampled anode waveform [npe units] for one PMT,
    V(t) = sum_i gain_i * SPE(t - t_i) (+ optional baseline noise), on a grid of
    n_samples at spacing dt_ns from t0_ns. This is the richest representation -- an
    ideal high-dynamic-range FADC trace -- from which all summary statistics
    (charge, time, ToT, ...) are extracted. It is linear (no front-end clipping),
    so its INTEGRAL gives the true, unsaturated charge that the official ToT loses.
    """
    grid = t0_ns + dt_ns * np.arange(n_samples)
    wf = rng.normal(0.0, noise_pe, n_samples) if noise_pe > 0.0 else np.zeros(n_samples)
    t = np.asarray(pe_times_ns, float)
    if t.size == 0:
        return grid, wf
    if model.tts_sigma_ns > 0.0:
        t = t + rng.normal(0.0, model.tts_sigma_ns, t.size)
    for ti in t:
        gi = rng.normal(model.gain, model.gainSpread)          # per-PE gain
        wf += max(gi, 0.0) * model.pulse_shape(grid - ti)
    return grid, wf


def spe_pulse_area_ns(model, dt_ns=0.02, span_ns=600.0):
    """Integral of the normalised 1-pe pulse shape [ns], for charge calibration."""
    t = np.arange(-0.5 * span_ns, 0.5 * span_ns, dt_ns)
    y = model.pulse_shape(t)
    return float(np.trapezoid(y, t)) if hasattr(np, "trapezoid") else float(np.trapz(y, t))


def waveform_summary(grid, wf, model):
    """Summary statistics extracted from an anode waveform -- ToT among them.

    charge_pe         : integral / SPE-area -> TRUE, unsaturated charge [npe]
    peak_pe           : peak amplitude [npe]
    leading_edge_ns   : first up-crossing of the discriminator threshold [ns]
    tot_threshold_ns  : bare above-threshold width of the anode pulse [ns]
                        (NOT the KM3NeT ToT -- see note)
    tot_KM3NeT_ns     : the official front-end ToT = Jpp charge->ToT model applied
                        to charge_pe (linear 7 ns/npe, saturating at 210 ns)

    Note: tot_threshold_ns != tot_KM3NeT_ns. The KM3NeT ToT is charge-driven (the
    front-end makes ToT grow with collected charge, then clip), not the bare anode
    pulse width. The two are bridged by the CHARGE (the waveform integral).
    """
    area = float(np.trapezoid(wf, grid)) if hasattr(np, "trapezoid") else float(np.trapz(wf, grid))
    charge = area / spe_pulse_area_ns(model)
    above = wf > model.threshold
    if above.any():
        idx = np.where(above)[0]
        leading = float(grid[idx[0]])
        tot_threshold = float(grid[idx[-1]] - grid[idx[0]])
    else:
        leading, tot_threshold = float("nan"), 0.0
    tot_km3net = float(model.apply_saturation(model._tot_unsat(charge))) if charge > model.threshold else 0.0
    return dict(charge_pe=charge, peak_pe=float(wf.max()) if wf.size else 0.0,
                leading_edge_ns=leading, tot_threshold_ns=tot_threshold,
                tot_KM3NeT_ns=tot_km3net)


def _selftest():
    """ToT(npe) curve + a waveform/ToT consistency check."""
    m = PMTToTModel()
    print(f"decayTime={m.decayTime_ns:.3f} ns  y1={m.y1:.3f}  t1={m.t1:.3f} ns  "
          f"start-of-linearisation x1={m.x1:.3f} pe")
    print("\nnpe   ToT_unsat   ToT(sat)   slewing   ToT->npe")
    for npe in [1, 2, 5, 10, 20, 30, 50, 100, 300]:
        tu = m._tot_unsat(npe)
        ts = m.apply_saturation(tu)
        print(f"{npe:4d}  {tu:8.1f}   {ts:7.1f}    {m.leading_edge_offset(npe):6.2f}   "
              f"{float(m.npe_from_tot(ts)):7.1f}")

    rng = np.random.default_rng(0)
    print(f"\nSPE pulse area = {spe_pulse_area_ns(m):.2f} ns  (charge calibration)")

    # --- RECONCILIATION of the 47-vs-102 ns discrepancy ---
    # Build the canonical waveform from 12 ~coincident PE, extract summary stats.
    grid, wf = make_waveform(np.full(12, 100.0), m, rng, t0_ns=50.0, dt_ns=0.1,
                             n_samples=4000, noise_pe=0.0)
    s = waveform_summary(grid, wf, m)
    print("\n--- reconciliation (12 coincident PE) ---")
    print(f"waveform charge (integral / SPE area)  = {s['charge_pe']:5.1f} pe   <- TRUE charge, unsaturated")
    print(f"waveform peak                          = {s['peak_pe']:5.1f} pe")
    print(f"bare anode threshold-width             = {s['tot_threshold_ns']:5.1f} ns   <- anode pulse width, NOT the ToT")
    print(f"KM3NeT ToT = Jpp_model(charge={s['charge_pe']:.1f})       = {s['tot_KM3NeT_ns']:5.1f} ns   <- official charge-driven ToT")
    print(f"  [cross-check Jpp_model(12 pe) saturated = {m.apply_saturation(m._tot_unsat(12)):.1f} ns]")
    print("=> RECONCILED: both numbers are correct but different quantities. The KM3NeT ToT")
    print("   is CHARGE-driven (front-end clips & widens ∝ charge), not the anode threshold")
    print("   width. Bridge = charge = integral of the waveform -> Jpp model -> official ToT.")
    print("   The waveform additionally keeps the true 12 pe charge that the saturating ToT loses.")


if __name__ == "__main__":
    _selftest()
