"""
qviz.geometry -- structural readout (bond lengths, angles) from a Structure.

Pure geometry on the atom positions; feeds the Structure info panel (the GaussView
"structural parameters" readout). Positions are bohr; lengths reported in Å.
"""
from __future__ import annotations
import numpy as np

from .data import Structure

BOHR = 0.52917721            # bohr -> Å

# covalent radii (Å) for bond detection (Cordero 2008, common elements)
_COV = {"H": 0.31, "Li": 1.28, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
        "F": 0.57, "Na": 1.66, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02,
        "Mn": 1.39, "Co": 1.26, "Ni": 1.24}


def bonds(s: Structure, scale: float = 1.3) -> list[tuple[int, int, float]]:
    """(i, j, length_Å) for atom pairs closer than scale·(covalent radii sum)."""
    P = np.asarray(s.positions, float)
    out = []
    for i in range(len(s.symbols)):
        for j in range(i + 1, len(s.symbols)):
            d = float(np.linalg.norm(P[i] - P[j])) * BOHR
            rcut = scale * (_COV.get(s.symbols[i], 0.75) + _COV.get(s.symbols[j], 0.75))
            if d < rcut:
                out.append((i, j, d))
    return out


def angles(s: Structure, bond_list) -> list[tuple[int, int, int, float]]:
    """(i, center, j, degrees) for every pair of bonds sharing a central atom."""
    P = np.asarray(s.positions, float)
    adj: dict[int, list[int]] = {}
    for i, j, _ in bond_list:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    out = []
    for c, nb in adj.items():
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                i, j = nb[a], nb[b]
                v1, v2 = P[i] - P[c], P[j] - P[c]
                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                out.append((i, c, j, np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out


def report(s: Structure) -> str:
    """A compact text readout: atoms (Å), bonds (Å), angles (°)."""
    P = np.asarray(s.positions, float) * BOHR
    L = ["Atoms (Å):"]
    for i, sym in enumerate(s.symbols):
        L.append(f"  {i:>2} {sym:<2} {P[i,0]:8.3f} {P[i,1]:8.3f} {P[i,2]:8.3f}")
    bl = bonds(s)
    L.append("\nBonds (Å):")
    for i, j, d in bl:
        L.append(f"  {s.symbols[i]}{i}–{s.symbols[j]}{j}  {d:6.3f}")
    al = angles(s, bl)
    if al:
        L.append("\nAngles (°):")
        for i, c, j, a in al:
            L.append(f"  {s.symbols[i]}{i}–{s.symbols[c]}{c}–{s.symbols[j]}{j}  {a:6.1f}")
    return "\n".join(L)
