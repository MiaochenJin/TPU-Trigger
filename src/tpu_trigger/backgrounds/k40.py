"""K40 hit-stream generation for single KM3NeT DOMs via k40gen.

k40gen generates a full reference detector (115 strings x 18 modules x 31
PMTs) per call; every DOM is statistically independent, so one short-span
call yields 2070 independent single-DOM streams. Memory scales with
span_ns x total PMT rate: 1e7 ns (10 ms) of full detector is ~4.6M hits
(~150 MB) — keep spans at this scale and iterate.
"""

import numpy as np

import k40gen as _k40gen

# (singles, genuine 2-fold, 3-fold, 4-fold) rates in Hz — k40gen reference
# values, consistent with published KM3NeT K40 calibration numbers.
RATES_DEFAULT = (7000.0, 700.0, 70.0, 0.0)

N_DOMS_PER_CALL = 115 * 18


def generate_full_detector(span_ns, rates=RATES_DEFAULT, seeds=(21341, 1245)):
    """One k40gen call. Returns (t_ns, dom_id, pmt, tot) flat arrays."""
    gens = _k40gen.Generators(int(seeds[0]), int(seeds[1]), list(rates))
    arr = np.asarray(_k40gen.generate_k40(0, int(span_ns), gens,
                                          "reference", False))
    t, dom, pmt, tot = arr
    return (t.astype(np.int64), dom.astype(np.int32),
            pmt.astype(np.int8), tot.astype(np.int16))


def iter_dom_streams(span_ns, rates=RATES_DEFAULT, seeds=(21341, 1245)):
    """Yield (dom_id, (t_ns, pmt, tot)) per DOM, each time-sorted.

    One full-detector call split by DOM: 2070 independent streams of
    span_ns each.
    """
    t, dom, pmt, tot = generate_full_detector(span_ns, rates, seeds)
    order = np.lexsort((t, dom))
    t, dom, pmt, tot = t[order], dom[order], pmt[order], tot[order]
    dom_ids, starts = np.unique(dom, return_index=True)
    bounds = np.append(starts, len(dom))
    for i, dom_id in enumerate(dom_ids):
        s, e = bounds[i], bounds[i + 1]
        yield int(dom_id), (t[s:e], pmt[s:e], tot[s:e])
