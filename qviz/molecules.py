"""
qviz.molecules -- a few molecular geometries (atomic numbers + positions in BOHR)
for the desktop app's molecule picker. The qchem binding takes arbitrary
geometries, so adding one is just a dict entry.

Each preset is (numbers, positions_3N, basis, suggested_grid_n). Heavier
molecules get a smaller default grid so sampling stays snappy.
"""
from __future__ import annotations
import math, pathlib

_A = 1.8897259886   # angstrom -> bohr

# symbol -> Z (through Kr; extend as needed)
_ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn "
             "Fe Co Ni Cu Zn Ga Ge As Se Br Kr").split()
_Z = {sym: i + 1 for i, sym in enumerate(_ELEMENTS)}


def from_xyz(path: str):
    """Parse a standard .xyz file (Å) -> (numbers, positions_bohr_flat, label)."""
    lines = pathlib.Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    numbers, positions = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        z = _Z.get(p[0].capitalize())
        if z is None:
            raise ValueError(f"unknown element {p[0]!r} in {path}")
        numbers.append(z)
        positions += [float(p[1]) * _A, float(p[2]) * _A, float(p[3]) * _A]
    return numbers, positions, pathlib.Path(path).stem


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

# NOTE: benzene (C6H6) is not a preset. At dzvp it is ~114 basis functions; under
# HF the 2-electron integral cache scales ~N^4 (~500x water) and OOM'd a 14 GB box.
# DFT is fine for RAM (user, 2026-11-01) -- the memory risk is HF's exact exchange
# only; examples/benzene.xyz is there to load for a DFT run.
# Everything below is <=16 electrons -- comparable to water.

# O2, experimental bond 1.208 Å = 2.283 bohr; ground state is a TRIPLET (paramagnetic)
_O2 = ([8, 8], [0.0, 0.0, -1.1415, 0.0, 0.0, 1.1415])

# name -> (numbers, positions, basis, grid_n, multiplicity).  multiplicity 0 = minimal spin
# (closed-shell singlet for even Ne); >1 forces the open-shell / spin-polarized path.
MOLECULES = {
    "Oxygen (O2, triplet)": (_O2[0],      _O2[1],      "dzvp", 72, 3),   # open-shell default
    # F2 at 1.412 Å: closed shell, and every atom is in the valence (GTH) bases -- the cross-code pair
    "Fluorine (F2)":        ([9, 9],      [0.0, 0.0, -1.3343, 0.0, 0.0, 1.3343], "valence_lowq_vb", 72, 0),
    "Water (H2O)":          (_WATER[0],   _WATER[1],   "dzvp", 80, 0),
    "Methane (CH4)":        (_METHANE[0], _METHANE[1], "dzvp", 72, 0),
    "Ammonia (NH3)":        (_AMMONIA[0], _AMMONIA[1], "dzvp", 72, 0),
    "Ethene (C2H4)":        (_ETHENE[0],  _ETHENE[1],  "dzvp", 72, 0),
}


def preset_spec(name: str):
    """The spec a preset starts as -- ONE rule for every view that offers presets (the Setup panel, the agent): the preset's
    basis and multiplicity; a valence (GTH) basis is LDA + pseudopotential (the only model both codes' GTH routes share),
    an all-electron one HF; point-group blocking on (the orbital table then carries irrep labels)."""
    from .spec import Accuracy, molecule
    Z, pos, basis, _n, mult = MOLECULES[name]
    pp = basis.startswith(("valence", "sipp"))
    return molecule(name.split(" (")[0], Z, pos, theory="DFT" if pp else "HF", functional="LDA" if pp else None,
                    multiplicity=mult or None, core="pseudopotential" if pp else "all-electron",
                    accuracy=Accuracy(representation={"basis": basis}, scf={"impose_symmetry": True}))


def preset_of(spec) -> str | None:
    """The preset whose structure `spec` has (label and geometry), else None."""
    for name, (Z, pos, *_r) in MOLECULES.items():
        sites = spec.structure.sites
        if (spec.structure.kind == "molecule" and name.split(" (")[0] == spec.label and len(sites) == len(Z)
                and all(s.Z == z and all(abs(a - b) < 1e-9 for a, b in zip(s.xyz, pos[3*i:3*i+3]))
                        for i, (s, z) in enumerate(zip(sites, Z)))):
            return name
    return None
