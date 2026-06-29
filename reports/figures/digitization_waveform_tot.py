"""Remade sanity-check plot with a GENERIC, physical SPE pulse:
- causal (zero before the PE time),
- fast rising edge + slow exponential decay (bi-exponential), unit peak = 1 pe,
- time-scale set so a single 1-pe pulse is above the 0.24 pe threshold for the
  nominal 1-pe ToT (25.08 ns).
ToT here = the discriminator output = contiguous threshold crossings of the
waveform, so overlapping pulses naturally MERGE (a realistic dense event -> one
big ToT). An isolated single PE is included to show the 1-pe ToT anchor.
"""
import sys, numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tpu_trigger.backgrounds.digitize import PMTToTModel, TIME_OVER_THRESHOLD_NS

THRESH = PMTToTModel().threshold          # 0.24 pe
TARGET = TIME_OVER_THRESHOLD_NS           # 25.08 ns  (nominal 1-pe ToT)

# --- generic causal SPE: bi-exponential, fast rise / slow decay, unit peak ---
def make_spe(tr, td):
    tp = (tr * td / (td - tr)) * np.log(td / tr)        # time of peak
    pk = np.exp(-tp / td) - np.exp(-tp / tr)
    def spe(dt):
        dt = np.asarray(dt, float)
        y = (np.exp(-dt / td) - np.exp(-dt / tr)) / pk
        return np.where(dt >= 0.0, np.clip(y, 0.0, None), 0.0)   # causal
    return spe

def one_pe_tot(spe):
    t = np.arange(0, 400, 0.01); a = t[spe(t) > THRESH]
    return (a.max() - a.min()) if a.size else 0.0

# fix rise = 3 ns; solve decay so the 1-pe ToT hits the nominal 25.08 ns
tr, lo, hi = 3.0, 3.1, 80.0
for _ in range(60):
    td = 0.5 * (lo + hi)
    if one_pe_tot(make_spe(tr, td)) < TARGET:
        lo = td
    else:
        hi = td
td = 0.5 * (lo + hi)
spe = make_spe(tr, td)
print(f"generic SPE: rise={tr:.1f} ns, decay={td:.1f} ns  ->  1-pe ToT = {one_pe_tot(spe):.2f} ns "
      f"(target {TARGET})")

# --- PE hit times on one PMT [ns]: isolated single + a dense burst + a late single ---
pe_times = np.array([
    40.0,
    110, 116, 120, 123, 128, 132, 137, 141, 146,    # 9-PE burst over ~36 ns -> merges
    240.0,
])

t0, dt, n = 0.0, 0.05, 6200                          # 0..310 ns @ 0.05 ns
grid = t0 + dt * np.arange(n)
wf = np.zeros(n)
for ti in pe_times:
    wf += spe(grid - ti)

# ToT = contiguous threshold crossings of the waveform (discriminator output)
above = wf > THRESH
chg = np.diff(above.astype(int))
starts = list(grid[1:][chg == 1]);  ends = list(grid[1:][chg == -1])
if above[0]:  starts = [grid[0]] + starts
if above[-1]: ends = ends + [grid[-1]]
hits = [(a, b) for a, b in zip(starts, ends) if b - a > 1.0]   # drop sub-ns flicker

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[3, 1], hspace=0.07))
ax1.fill_between(grid, THRESH, wf, where=wf > THRESH, color="#9ec9ff", alpha=0.5)
ax1.plot(grid, wf, color="#1f4e8c", lw=1.3, label="anode waveform  V(t)")
ax1.axhline(THRESH, color="gray", ls="--", lw=1.0, label=f"threshold = {THRESH} pe")
ax1.vlines(pe_times, -0.55, -0.12, color="#c0392b", lw=1.3)
ax1.plot([], [], color="#c0392b", lw=1.3, label=f"PE hits (sim), N={len(pe_times)}")
ax1.plot([], [], color="#9ec9ff", lw=6, alpha=0.5, label="V > threshold")
ax1.set_ylabel("amplitude  [pe]"); ax1.set_ylim(-0.7, max(wf.max() * 1.15, 1.3))
ax1.legend(loc="upper right", fontsize=8, frameon=False)
ax1.set_title(f"Generic SPE waveform (rise {tr:.0f} ns, decay {td:.0f} ns; 1-pe ToT = {TARGET} ns)  →  ToT")

ax2.fill_between(grid, 0, above.astype(float), step="mid", color="#2e8b57", alpha=0.55)
ax2.step(grid, above.astype(float), where="mid", color="#1e6b3a", lw=1.3)
ax2.set_ylim(-0.2, 1.55); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["", "hit"])
ax2.set_ylabel("ToT"); ax2.set_xlabel("time  [ns]")
labels = ["1 PE", "9-PE burst (merged)", "1 PE"]
for (a, b), lab in zip(hits, labels):
    ax2.annotate(f"{lab}\nToT={b-a:.0f} ns", xy=((a + b) / 2, 1.0), xytext=((a + b) / 2, 1.18),
                 ha="center", va="bottom", fontsize=8, color="#1e6b3a")

fig.savefig("waveform_tot_realistic.png", dpi=150, bbox_inches="tight")
print("\nToT hits (threshold-crossing intervals):")
for a, b in hits:
    print(f"  t=[{a:6.1f}, {b:6.1f}] ns   ToT = {b - a:5.1f} ns")
print("wrote waveform_tot_realistic.png")
