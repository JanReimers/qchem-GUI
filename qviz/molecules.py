"""
qviz.molecules -- a few molecular geometries (atomic numbers + positions in BOHR)
for the desktop app's molecule picker. The qchem binding takes arbitrary
geometries, so adding one is just a dict entry.

Each preset is (numbers, positions_3N, basis, suggested_grid_n). Heavier
molecules get a smaller default grid so sampling stays snappy.
"""
from __future__ import annotations
import math

_A = 1.8897259886   # angstrom -> bohr


def _flat(coords):
    return [c for xyz in coords for c in xyz]


# --- water (experimental, already bohr; matches M_HF_U.C) --------------------
_WATER = ([8, 1, 1],
          [0.0, 0.0, 0.0,  0.0, 1.431, 1.107,  0.0, -1.431, 1.107])

# --- methane: C + 4 H at tetrahedral corners --------------------------------
_a = 1.087 * _A / math.sqrt(3)
_METHANE = ([6, 1, 1, 1, 1],
            _flat([(0, 0, 0), (_a, _a, _a), (_a, -_a, -_a), (-_a, _a, -_a), (-_a, -_a, _a)]))

# --- ammonia (pyramidal) ----------------------------------------------------
_AMMONIA = ([7, 1, 1, 1],
            _flat([(x*_A, y*_A, z*_A) for (x, y, z) in [
                (0, 0, 0.1173), (0, 0.9377, -0.2737),
                (0.8121, -0.4689, -0.2737), (-0.8121, -0.4689, -0.2737)]]))

# --- ethene C2H4 (planar) ---------------------------------------------------
_ETHENE = ([6, 6, 1, 1, 1, 1],
           _flat([(x*_A, y*_A, z*_A) for (x, y, z) in [
               (0, 0, 0.6695), (0, 0, -0.6695),
               (0, 0.9289, 1.2321), (0, -0.9289, 1.2321),
               (0, 0.9289, -1.2321), (0, -0.9289, -1.2321)]]))

# NOTE: benzene (C6H6) is deliberately NOT here. At dzvp it is ~114 basis
# functions; the 2-electron integral cache scales ~N^4 (~500x water) and OOM'd a
# 14 GB box. Re-add it only with a memory budget / a smaller basis, or on a rig
# with more RAM. Everything below is <=16 electrons -- comparable to water.

# name -> (numbers, positions, basis, grid_n).
MOLECULES = {
    "Water (H2O)":   (_WATER[0],   _WATER[1],   "dzvp", 80),
    "Methane (CH4)": (_METHANE[0], _METHANE[1], "dzvp", 72),
    "Ammonia (NH3)": (_AMMONIA[0], _AMMONIA[1], "dzvp", 72),
    "Ethene (C2H4)": (_ETHENE[0],  _ETHENE[1],  "dzvp", 72),
}
