"""
qviz.backends.gth -- ONE GTH pseudopotential, written in another code's format (doc/RunSpecPlan.md §3.4, S7).

The source is qchem's GTH database (BasisSetData/GTH_POTENTIALS.txt, CP2K's format -- the file qchem's own JSON is
transcoded from), selected by the gth-lda-valence set's q.  A code that reads GTH/HGH pseudopotentials natively gets
THAT entry re-wrapped, never a look-alike from its own library, so a cross-code comparison compares codes.

ABINIT: pspcod 10 (HGH with the full h matrix; m_psp_hgh.F90 psp10in).  The two layouts are the same physics with the
same triangular h_ij per angular momentum (Hartree), so the conversion is a re-wrap: the header, then
`rloc nn C1..Cnn`, `nnonloc`, and per l `r nprl h11..h1n` / the rest of the triangle / (l > 0) a zero k triangle
(no spin-orbit).  ABINIT needs nnonloc = lmax+1, so an entry with no projectors (H) gets one empty l = 0 channel.
"""
from __future__ import annotations

import functools
import pathlib
from dataclasses import dataclass

from .qe import pp_valence, qchem_src

_LDA_FAMILY = "GTH-PADE"


@dataclass(frozen=True)
class GTHEntry:
    element: str
    zion: int
    rloc: float
    C: tuple[float, ...]
    projectors: tuple[tuple[float, int, tuple[tuple[float, ...], ...]], ...]   # per l: (r_l, nprj, upper-triangle rows of h)


def database() -> pathlib.Path:
    return qchem_src() / "BasisSetData/GTH_POTENTIALS.txt"


@functools.lru_cache(maxsize=None)
def read_entry(element: str, q: int, family: str = _LDA_FAMILY) -> GTHEntry:
    """The `<El> <family>-q<q>` entry of the database.  Raises KeyError when there is none."""
    lines = database().read_text().splitlines()
    name = f"{family}-q{q}"
    i = next((n for n, l in enumerate(lines) if (t := l.split()) and t[0] == element and name in t[1:]), None)
    if i is None:
        raise KeyError(f"{element} {name} not in {database()}")
    body = [l.split() for l in lines[i + 1:]]
    zion = sum(int(x) for x in body[0])
    rloc, nn = float(body[1][0]), int(body[1][1])
    C = tuple(float(x) for x in body[1][2:2 + nn])
    nnonloc = int(body[2][0])
    k, projs = 3, []
    for _l in range(nnonloc):
        r, nprj = float(body[k][0]), int(body[k][1])
        rows = [tuple(float(x) for x in body[k][2:2 + nprj])]
        for m in range(1, nprj):
            rows.append(tuple(float(x) for x in body[k + m][:nprj - m]))
        projs.append((r, nprj, tuple(rows)))
        k += max(nprj, 1)
    return GTHEntry(element, zion, rloc, C, tuple(projs))


def gth_entry(element: str) -> GTHEntry:
    """The gth-lda-valence set's entry for `element` (q from qchem's pseudopotentials.json)."""
    return read_entry(element, pp_valence(element))


def abinit_name(element: str) -> str:
    return f"{element}.gth-q{pp_valence(element)}.psp10"


def to_abinit_psp10(element: str, Z: int, ixc: int) -> str:
    """The pspcod-10 file for `element`'s gth-lda-valence entry; `ixc` is written as pspxc (the run's own functional)."""
    e = gth_entry(element)
    projs = e.projectors or ((0.0, 0, ()),)
    lmax = len(projs) - 1
    out = [f"{element} {_LDA_FAMILY}-q{e.zion}, qchem {database().name} via qviz (HGH full-h, pspcod 10)",
           f"{Z} {e.zion} 261031          zatom, zion, pspdat",
           f"10 {ixc} {lmax} 0 2001 0     pspcod, pspxc, lmax, lloc, mmax, r2well",
           f"{e.rloc:.8f} {len(e.C)} " + " ".join(f"{c:.8f}" for c in e.C) + "   rloc, nn, C1..Cnn",
           f"{len(projs)}                 nnonloc"]
    for l, (r, nprj, rows) in enumerate(projs):
        out.append(f"{r:.8f} {nprj} " + " ".join(f"{h:.8f}" for h in (rows[0] if rows else ())))
        out += ["  " + " ".join(f"{h:.8f}" for h in row) for row in rows[1:]]
        if l > 0:
            out += ["  " + " ".join("0.0" for _ in range(nprj - m)) for m in range(nprj)]   # k_ij: no spin-orbit
    return "\n".join(out) + "\n"
