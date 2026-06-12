"""KM3NeT reference DOM geometry and K40 coincidence parameterization.

PMT direction table and constants are taken verbatim from the k40gen source
(lib/generate/generate_scalar.h, generate_common.h) so that our analysis uses
exactly the geometry the generator uses. PMT ids 0..30; +z is up; PMTs 0..11
point into the upper hemisphere, 12..30 into the lower.
"""

import numpy as np

N_PMT = 31

# unit direction vectors of the 31 PMTs (from k40gen generate_scalar.h)
PMT_DIRS = np.array([
    [ 0.000, -0.832,  0.555],
    [-0.955,  0.000,  0.295],
    [-0.478, -0.827,  0.295],
    [ 0.478, -0.827,  0.295],
    [ 0.720, -0.416,  0.555],
    [-0.720, -0.416,  0.555],
    [ 0.955,  0.000,  0.295],
    [-0.720,  0.416,  0.555],
    [ 0.720,  0.416,  0.555],
    [ 0.000,  0.832,  0.555],
    [ 0.478,  0.827,  0.295],
    [-0.478,  0.827,  0.295],
    [ 0.000,  0.955, -0.295],
    [ 0.416,  0.720, -0.555],
    [ 0.000,  0.527, -0.850],
    [ 0.827,  0.478, -0.295],
    [-0.827,  0.478, -0.295],
    [-0.416,  0.720, -0.555],
    [-0.456,  0.263, -0.850],
    [ 0.456,  0.263, -0.850],
    [-0.832,  0.000, -0.555],
    [ 0.832,  0.000, -0.555],
    [ 0.000,  0.000, -1.000],
    [ 0.827, -0.478, -0.295],
    [ 0.000, -0.527, -0.850],
    [ 0.456, -0.263, -0.850],
    [-0.456, -0.263, -0.850],
    [-0.827, -0.478, -0.295],
    [-0.416, -0.720, -0.555],
    [ 0.416, -0.720, -0.555],
    [ 0.000, -0.955, -0.295],
], dtype=np.float64)

# K40 coincidence probability vs cos(opening angle) between two PMTs,
# p(ct) = exp(ct * (p2 + ct * (p3 + ct * p4)))  (k40gen generate_common.h)
P2, P3, P4 = 2.4347, -0.68884, 1.3911

# ToT pulse distribution (Gaussian fit, ns)
TOT_MEAN, TOT_SIGMA = 26.936, 2.44078


def cross_prob(ct):
    """Relative genuine-coincidence probability vs cos(opening angle)."""
    ct = np.asarray(ct, dtype=np.float64)
    return np.exp(ct * (P2 + ct * (P3 + ct * P4)))


def pair_cos_angles():
    """(31, 31) matrix of cos(opening angle) between PMT pairs."""
    return PMT_DIRS @ PMT_DIRS.T
