# Patches applied to our k40gen clone

Clone: `gitlab.nikhef.nl/roelaaij/k40gen`, at the lab `tools/k40gen` on FASRC
and in-project `tools/k40gen` on WARD. Re-apply these if the clone is recreated
(the WARD setup script `env/setup_env_ward.sh` does so automatically). Patches
1-2 are platform-agnostic and upstream-reportable; **4-5 are only needed on
WARD** (gcc 13 / glibc 2.39 — newer than FASRC's gcc 8.5, which compiled the
unpatched code).

## 1. Python ≥3.11 module suffix (build bug)

`cmake/FindPythonLibsNew.cmake:82` queries `sysconfig.get_config_var('SO')`,
removed in Python 3.11+ → returns `None` → the extension installs as a file
literally named `k40genNone` and cannot be imported.

Fix: `s/get_config_var('SO')/get_config_var('EXT_SUFFIX')/`.

## 2. Coincidence PMT/time misalignment (physics bug, scalar path)

`src/generate/generate.cpp` (`fill_coincidences`): member hit times are
written with pre-increment `times[++idx]` while the caller fills the matching
PMT values over `[idx, idx + n_times)` (post-increment convention everywhere
else). Every coincidence's PMT list is shifted one slot against its times:
time clustering and total rates look correct, but pair identities are
scrambled — flat pair-angle distribution (measured top-50 pair fraction 0.136
vs 0.468 expected) and impossible same-PMT "coincidences".

Fix: `s/times\[++idx\]/times[idx++]/g` (3 occurrences).

After the fix: pair-angle log-correlation vs the cross_prob parameterization
0.993, same-PMT pairs at the accidental-only level, fold rates matching the
multiplicity model.

## Note on k40gen's rate semantics (not a bug, but non-obvious)

`Generators(s1, s2, [r0, r1, r2, r3])`: r0 = per-PMT singles rate. Total
coincidence-event rate = r1 + r2 + r3 (per DOM). Each coincidence has 2 + M
members with M sampled with weights = the whole rates array
(P(M=m) ∝ r[m]) — so the (2+m)-fold rate is (r1+r2+r3) · r[m] / Σr.
With the nominal hierarchy (7000, 700, 70, 0) this lands close to the naive
reading (2-fold ≈ 694 Hz, 3-fold ≈ 69 Hz, 4-fold ≈ 7 Hz), but the naive
reading breaks if you change the ratios.

## 3. AVX2 flag silently ignored

Our build compiles without AVX2 (`USE_AVX2` undefined), so
`generate_k40(..., use_avx2=True)` silently runs the scalar path. Not a
problem for us (scalar is fast enough), but don't assume the flag did
anything.

## 4. [WARD] gcc 13 — missing `<stdexcept>` (build break)

`lib/generate/generate_common.h` uses `std::domain_error` (lines ~97, ~126)
without including `<stdexcept>`. gcc 8.5's libstdc++ pulled it in transitively
via another header; gcc 13's does not → `'domain_error' is not a member of
'std'`. Fix: add `#include <stdexcept>` to the include block.

## 5. [WARD] glibc 2.39 — bundled Catch2 unit tests don't compile

The vendored Catch2 v2 (`build/.../catch2/catch.hpp`) declares
`constexpr ... sigStackSize = ... MINSIGSTKSZ`, but on glibc ≥2.34 `MINSIGSTKSZ`
is a runtime `sysconf()` call, not a compile-time constant → "call to
non-'constexpr' function 'sysconf'". This only affects k40gen's C++ unit-test
target, which we don't need. Fix: `option(ENABLE_TESTS ... TRUE)` → `FALSE` in
`CMakeLists.txt` (and `rm -rf build` so the cached CMake config is regenerated).
The Python module builds fine; the project's own physics validation
(`validate_k40`) is the real test gate.
