"""
qviz.symmetry -- the symmetry a structure HAS, as the library detects it (qchem_py.point_group / space_group).

DETECTED, never imposed: what the positions (and, for a crystal, the site moments) allow.  Whether the SCF is then
blocked by it is a separate, stated choice in the spec (accuracy.scf.impose_symmetry); a user may impose a subgroup of
what is detected, never more.

A crystal's operations come back as integer matrices W in lattice coordinates; their crystal CLASS (one of the 32) is
named here from the rotation types, which (det W, tr W) fixes in any basis.  A magnetic decoration is seen by giving
each site a species label that carries its moment's sign: the result is the SPIN-PRESERVING subgroup (the unitary half
of the Shubnikov group; the moment-flipping half is not named here).
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from .spec import RunSpec

# rotation type from (det, trace): 2/3/4/6 proper, "-1" inversion, "m" mirror, "-3"/"-4"/"-6" improper
_TYPE = {(1, 3): "1", (1, -1): "2", (1, 0): "3", (1, 1): "4", (1, 2): "6",
         (-1, -3): "-1", (-1, 1): "m", (-1, 0): "-3", (-1, -1): "-4", (-1, -2): "-6"}
_ORDER = ("2", "3", "4", "6", "-1", "m", "-3", "-4", "-6")
#: the 32 crystal classes: (Hermann-Mauguin, Schoenflies) keyed by the counts of each non-identity rotation type
_CLASSES = {
    (0, 0, 0, 0, 0, 0, 0, 0, 0): ("1", "C1"),         (0, 0, 0, 0, 1, 0, 0, 0, 0): ("-1", "Ci"),
    (1, 0, 0, 0, 0, 0, 0, 0, 0): ("2", "C2"),         (0, 0, 0, 0, 0, 1, 0, 0, 0): ("m", "Cs"),
    (1, 0, 0, 0, 1, 1, 0, 0, 0): ("2/m", "C2h"),      (3, 0, 0, 0, 0, 0, 0, 0, 0): ("222", "D2"),
    (1, 0, 0, 0, 0, 2, 0, 0, 0): ("mm2", "C2v"),      (3, 0, 0, 0, 1, 3, 0, 0, 0): ("mmm", "D2h"),
    (1, 0, 2, 0, 0, 0, 0, 0, 0): ("4", "C4"),         (1, 0, 0, 0, 0, 0, 0, 2, 0): ("-4", "S4"),
    (1, 0, 2, 0, 1, 1, 0, 2, 0): ("4/m", "C4h"),      (5, 0, 2, 0, 0, 0, 0, 0, 0): ("422", "D4"),
    (1, 0, 2, 0, 0, 4, 0, 0, 0): ("4mm", "C4v"),      (3, 0, 0, 0, 0, 2, 0, 2, 0): ("-42m", "D2d"),
    (5, 0, 2, 0, 1, 5, 0, 2, 0): ("4/mmm", "D4h"),    (0, 2, 0, 0, 0, 0, 0, 0, 0): ("3", "C3"),
    (0, 2, 0, 0, 1, 0, 2, 0, 0): ("-3", "S6"),        (3, 2, 0, 0, 0, 0, 0, 0, 0): ("32", "D3"),
    (0, 2, 0, 0, 0, 3, 0, 0, 0): ("3m", "C3v"),       (3, 2, 0, 0, 1, 3, 2, 0, 0): ("-3m", "D3d"),
    (1, 2, 0, 2, 0, 0, 0, 0, 0): ("6", "C6"),         (0, 2, 0, 0, 0, 1, 0, 0, 2): ("-6", "C3h"),
    (1, 2, 0, 2, 1, 1, 2, 0, 2): ("6/m", "C6h"),      (7, 2, 0, 2, 0, 0, 0, 0, 0): ("622", "D6"),
    (1, 2, 0, 2, 0, 6, 0, 0, 0): ("6mm", "C6v"),      (3, 2, 0, 0, 0, 4, 0, 0, 2): ("-6m2", "D3h"),
    (7, 2, 0, 2, 1, 7, 2, 0, 2): ("6/mmm", "D6h"),    (3, 8, 0, 0, 0, 0, 0, 0, 0): ("23", "T"),
    (3, 8, 0, 0, 1, 3, 8, 0, 0): ("m-3", "Th"),       (9, 8, 6, 0, 0, 0, 0, 0, 0): ("432", "O"),
    (3, 8, 0, 0, 0, 6, 0, 6, 0): ("-43m", "Td"),      (9, 8, 6, 0, 1, 9, 8, 6, 0): ("m-3m", "Oh"),
}


def crystal_class(W: list[float]) -> Optional[tuple[str, str]]:
    """(Hermann-Mauguin, Schoenflies) of the point group whose linear parts are `W` (flat row-major 3x3 each), or None
    when they are not one of the 32 classes (a non-primitive cell can repeat a W)."""
    mats = {tuple(round(x) for x in W[k:k + 9]) for k in range(0, len(W), 9)}
    types = Counter()
    for m in mats:
        det = (m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]))
        types[_TYPE.get((det, m[0] + m[4] + m[8]), "?")] += 1
    if types["?"] or types["1"] != 1:
        return None
    return _CLASSES.get(tuple(types[t] for t in _ORDER))


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def detected(spec: RunSpec, qc) -> list[str]:
    """What the library detects for `spec`'s structure, as lines for the Structure tab; `qc` is the qchem_py module (None:
    one line saying detection needs it)."""
    if qc is None:
        return ["detection needs the qchem_py extension (QCHEM_BUILD)"]
    st = spec.structure
    try:
        if st.kind == "molecule":
            if len(st.sites) == 1:
                return ["O(3) -- a spherical atom (the SCF blocks by D2h)"]
            g = qc.point_group([s.Z for s in st.sites], [x for s in st.sites for x in s.xyz])
            return [f"{g['symbol']}  point group" + (f", order {g['order']}" if g["order"] else "")
                    + f"  (the SCF blocks by {g['abelian']})"]
        L = [x for row in st.lattice_bohr for x in row]
        frac = [x for s in st.sites for x in s.frac]
        out = []

        def line(what: str, species: list[int]) -> str:
            g = qc.space_group(L, species, frac)
            cls = crystal_class(g["W"])
            name = f"class {cls[0]} ({cls[1]})" if cls else "class not named (non-primitive cell?)"
            return f"{what}: {g['order']} operations, {name}, {'symmorphic' if g['symmorphic'] else 'non-symmorphic'}"
        out.append(line("space group", [s.Z for s in st.sites]))
        m = spec.physics.spin.site_moments
        if spec.physics.spin.polarized and m and any(m):
            out.append(line("with the site moments (spin-preserving)", [4 * s.Z + 1 + _sign(x) for s, x in zip(st.sites, m)]))
        return out
    except Exception as e:                            # noqa: BLE001 -- a readout, never a crash
        return [f"detection failed: {e}"]
