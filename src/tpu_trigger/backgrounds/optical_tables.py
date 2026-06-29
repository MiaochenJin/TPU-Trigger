"""Parser + loader for the KM3NeT OMGsim optical-property tables.

The DOM response model needs measured curves -- PMT quantum efficiency QE(lambda),
glass and optical-gel transmission, seawater absorption -- which the KM3NeT
collaboration maintains as Geant4 material `PROPERTY` tables in the
*internal* `inputs4qefit/omgsim` repository. We clone that repo under `external/`
(gitignored) and read the tables at runtime here.

IMPORTANT: this module only *reads* those tables if the external clone is
present; it embeds no KM3NeT-internal numbers. `response.py` falls back to its
documented placeholder curves when the tables are absent, so the public repo
stays free of internal data and still runs everywhere.

File format (Geant4 / GenericLAND material data)::

    CREATE <material>
    DENSITY ...
    COMPONENTS  <element frac> ...  COMPONENTS
    CREATE
    PROPERTY <NAME>            # e.g. EFFICIENCY, ABSLENGTH, RINDEX
    OPTION wavelength|eV       # x-axis unit of the rows below
    <x> <value>
    ...

`parse_dat` returns ``{material: {property: {"unit", "x_nm", "v"}}}`` with the
x-axis always converted to nanometres and sorted ascending. Geant4's default
length unit is the millimetre, so ABSLENGTH values are returned in mm.
"""

import os

import numpy as np

# h*c in eV*nm:  lambda_nm = HC_EV_NM / E_eV
HC_EV_NM = 1239.841984

_HERE = os.path.dirname(os.path.abspath(__file__))
# <repo>/external/inputs4qefit/omgsim/common/data  (this file is src/tpu_trigger/backgrounds/)
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..",
    "external", "inputs4qefit", "omgsim", "common", "data"))

# DOM layer thicknesses [mm] from inputs4qefit/.../KM3OMDOMQE2.dat
GLASS_MM = 14.0     # GlassThickness
GEL_MM = 2.0        # DistSpherePMT (glass inner surface -> PMT front gel gap)

# material / property names as they appear in the OMGsim .dat files
_QE_FILE, _QE_MAT, _QE_PROP = "KM3MatPMT3inchesQE2.dat", "photocathode3inchesQE2", "EFFICIENCY"
_GLASS_FILE, _GLASS_MAT = "KM3MatGlassOM.dat", "AntaresGlass"      # Erlangen-measured KM3NeT glass
_GEL_FILE, _GEL_MAT = "KM3MatOpticalGel.dat", "WackerSilGel612_A100B67"  # KM3NeT gel mix
_WATER_FILE, _WATER_MAT = "KM3MatWater.dat", "NEMOWater"           # the macro's setTargetMaterial

# The QE table is stored as "QE2" = real QE x 2 (a Geant4 oversampling trick;
# JEventTimesliceWriter later applies QE=QE*0.5). Undo it to recover real QE.
QE2_SCALE = 0.5


def parse_dat(path):
    """Parse an OMGsim material .dat file into nested dicts (see module doc)."""
    materials = {}
    cur_mat = cur_prop = cur_unit = None
    rows = []

    def flush():
        nonlocal rows
        if cur_mat is not None and cur_prop is not None and rows:
            x = np.array([r[0] for r in rows], dtype=float)
            v = np.array([r[1] for r in rows], dtype=float)
            if cur_unit == "eV":
                x = HC_EV_NM / x
            order = np.argsort(x)
            materials[cur_mat][cur_prop] = {"unit": cur_unit, "x_nm": x[order], "v": v[order]}
        rows = []

    with open(path) as fh:
        for raw in fh:
            line = raw.split("#")[0].split("//")[0].strip()
            if not line:
                continue
            toks = line.split()
            key = toks[0].upper()
            if key == "CREATE":
                flush()
                cur_prop = cur_unit = None
                if len(toks) > 1:                       # named CREATE starts a material
                    cur_mat = toks[1]
                    materials.setdefault(cur_mat, {})
                continue
            if key == "PROPERTY":
                flush()
                cur_prop = toks[1] if len(toks) > 1 else None
                cur_unit = None
                continue
            if key == "OPTION":
                cur_unit = toks[1] if len(toks) > 1 else None
                continue
            if key in ("DENSITY", "COMPONENTS", "NAME"):
                continue
            if cur_prop is not None and len(toks) >= 2:  # a "<x> <value>" data row
                try:
                    rows.append((float(toks[0]), float(toks[1])))
                except ValueError:                       # e.g. a "G4_K 40" component line
                    pass
    flush()
    return materials


def _table(data_dir, fname, material, prop):
    tbl = parse_dat(os.path.join(data_dir, fname))
    return tbl[material][prop]


def load_curves(data_dir=None, glass_mm=GLASS_MM, gel_mm=GEL_MM):
    """Load the measured optical curves needed by the response/closure models.

    Returns a dict with x in nm and absorption lengths in mm::

        qe            (lam_nm, real_QE)
        glass_abs_mm  (lam_nm, L_abs_glass_mm)
        gel_abs_mm    (lam_nm, L_abs_gel_mm)
        water_abs_mm  (lam_nm, L_abs_water_mm)
        glass_mm, gel_mm   layer thicknesses [mm]

    Raises FileNotFoundError if the external OMGsim clone is absent.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"OMGsim optical tables not found at {data_dir}")

    eff = _table(data_dir, _QE_FILE, _QE_MAT, _QE_PROP)
    glass = _table(data_dir, _GLASS_FILE, _GLASS_MAT, "ABSLENGTH")
    gel = _table(data_dir, _GEL_FILE, _GEL_MAT, "ABSLENGTH")
    water = _table(data_dir, _WATER_FILE, _WATER_MAT, "ABSLENGTH")

    return {
        "qe": (eff["x_nm"], eff["v"] * QE2_SCALE),
        "glass_abs_mm": (glass["x_nm"], glass["v"]),
        "gel_abs_mm": (gel["x_nm"], gel["v"]),
        "water_abs_mm": (water["x_nm"], water["v"]),
        "glass_mm": glass_mm,
        "gel_mm": gel_mm,
        "source": data_dir,
    }


def transmission(lam_nm, abs_x_nm, abs_v_mm, thickness_mm):
    """Beer-Lambert transmission exp(-d / L_abs(lambda)) for a uniform layer."""
    lam = np.asarray(lam_nm, dtype=float)
    labs = np.interp(lam, abs_x_nm, abs_v_mm, left=abs_v_mm[0], right=abs_v_mm[-1])
    with np.errstate(divide="ignore"):
        return np.exp(-thickness_mm / np.where(labs > 0, labs, np.inf))
