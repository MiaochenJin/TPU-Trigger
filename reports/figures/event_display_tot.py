"""DOM event display: the 31-PMT DOM unrolled (azimuth vs cos-zenith), each lit
PMT coloured and sized by its ToT (the Jpp charge->ToT for the PEs it collected).

Input: OMGsim `.evt` files (PE-level hits; NOT committed -- they live on FASRC
under .../omgsim_{k40,biolum}/runs/ or wherever the OMGsim run wrote output.evt).
Run from the repo root so `src/` is importable:

    python reports/figures/event_display_tot.py \
        --k40 path/to/k40/output.evt --biolum path/to/biolum/output.evt \
        --out reports/figures/event_display_k40_biolum.png
"""
import argparse, sys, os
import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tpu_trigger.backgrounds.geometry import PMT_DIRS
from tpu_trigger.backgrounds.digitize import PMTToTModel

M = PMTToTModel()
AZ = np.degrees(np.arctan2(PMT_DIRS[:, 1], PMT_DIRS[:, 0]))   # azimuth [deg]
CZ = PMT_DIRS[:, 2]                                           # cos(zenith)


def parse_events(path):
    cur = None
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "start_event:":
            cur = {}
        elif t[0] == "hit:" and cur is not None:
            cur.setdefault(int(t[2]), []).append(float(t[4]))   # {pmt_code(101-131): [times]}
        elif t[0] == "end_event:":
            if cur:
                yield cur
            cur = None


def pick_event(path, target_mult=None, min_mult=2):
    """If target_mult: event with multiplicity closest to it. Else: brightest
    (most total PE) event with >= min_mult lit PMTs."""
    best, bkey = None, None
    for ev in parse_events(path):
        mult = len(ev)
        if mult < min_mult:
            continue
        key = (abs(mult - target_mult), -mult) if target_mult else (-sum(len(v) for v in ev.values()),)
        if bkey is None or key < bkey:
            best, bkey = ev, key
    return best


def draw(ax, ev, title):
    ax.scatter(AZ, CZ, s=120, facecolors="none", edgecolors="#cfcfcf", linewidths=1.0, zorder=1)
    az, cz, tot, npe = [], [], [], []
    for code, times in ev.items():
        i = code - 101                       # detx id 101..131 -> PMT_DIRS index 0..30
        n = len(times)
        az.append(AZ[i]); cz.append(CZ[i]); npe.append(n)
        tot.append(M.tot(n) if n > M.threshold else 0.0)
    sc = ax.scatter(az, cz, s=[90 + 4.0 * x for x in tot], c=tot, cmap="plasma",
                    vmin=20, vmax=max(60, max(tot)), edgecolors="black", linewidths=0.6, zorder=3)
    for x, y, n in zip(az, cz, npe):
        ax.annotate(str(n), (x, y), ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", zorder=4)
    ax.set_xlim(-185, 185); ax.set_ylim(-1.12, 1.12); ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("azimuth  φ  [deg]"); ax.set_ylabel("cos(zenith)   (+1 up / −1 down)")
    ax.set_title(title, fontsize=10); ax.grid(ls=":", alpha=0.4, zorder=0)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k40", required=True)
    ap.add_argument("--biolum", required=True)
    ap.add_argument("--out", default="event_display_k40_biolum.png")
    a = ap.parse_args()

    k40 = pick_event(a.k40, target_mult=3)              # K40 3-fold coincidence
    bio = pick_event(a.biolum, min_mult=10)             # brightest >=10-PMT flash
    fig, (x1, x2) = plt.subplots(1, 2, figsize=(13, 4.8))
    s1 = draw(x1, k40, f"K40 decay — {len(k40)}-fold coincidence  (number = PE/PMT)")
    s2 = draw(x2, bio, f"Bioluminescence flash — {len(bio)} PMTs lit")
    for ax, sc in [(x1, s1), (x2, s2)]:
        fig.colorbar(sc, ax=ax, pad=0.02).set_label("ToT  [ns]")
    fig.suptitle("OMGsim event displays — single 31-PMT DOM, lit PMTs by ToT", fontsize=11)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
