"""
qviz.basisinfo -- what a spec's orbital basis IS, before any SCF: the shells on each element (read from qchem's
BasisSetData file) and, for a molecule, the size and conditioning the facade's own basis builder reports
(qchem_py.basis_info: the overlap S's extreme eigenvalues).

A small S eigenvalue is a near-linear dependence: diffuse functions on neighbouring atoms spanning almost the same
space.  The condition number s_max/s_min says how much the SCF's orthogonalization amplifies round-off.
"""
from __future__ import annotations

import functools
from collections import Counter
from typing import Optional

from . import gridinfo
from .backends import gaussian_basis
from .backends.solid2 import basis_for
from .spec import RunSpec

_L = "spdfgh"
_CART = (1, 3, 6, 10, 15, 21)                     # Cartesian components per l (the facade's molecular basis is Cartesian)
_NEAR_DEPENDENT = 1e-6                            # an S eigenvalue below this is flagged as a near-linear dependence


def basis_name(spec: RunSpec) -> str:
    """The basis SOLID² will use: the stated one, else its ε rule's."""
    return (spec.accuracy.representation or {}).get("basis") or basis_for(spec.accuracy.epsilon_Ha_per_atom)


def element_rows(spec: RunSpec) -> list[tuple[str, int, str, int]]:
    """(element, atoms of it, contracted shells like "3s 2p 1d", Cartesian functions per atom) per element in the
    structure; shells "no block in <basis>" (and 0 functions) for an element the file lacks."""
    name = basis_name(spec)
    try:
        data = gaussian_basis.read_bsd(name)
    except (OSError, ValueError):
        data = {}
    out = []
    for el, n in Counter(s.element for s in spec.structure.sites).items():
        shells = data.get(el.upper())
        if not shells:
            out.append((el, n, f"no block in {name}", 0))
            continue
        per_l = Counter(l for l, _ in shells)
        out.append((el, n, " ".join(f"{per_l[l]}{_L[l]}" for l in sorted(per_l)), sum(_CART[l] for l, _ in shells)))
    return out


@functools.lru_cache(maxsize=32)
def _info(qc, numbers: tuple, positions: tuple, basis: str) -> dict:
    return qc.basis_info(list(numbers), list(positions), basis)


def fits(spec: RunSpec, record: Optional[dict]) -> list[str]:
    """The Basis tab's other bases -- what the run fits ρ (Hartree) and Vxc in -- as the record files them
    (grids.hamiltonian.fits: [{role, name, nFunctions}]; the Hamiltonian builds them).  Hartree-Fock has none: its Coulomb
    and exchange are analytic."""
    ph = spec.physics
    if spec.structure.kind != "molecule":
        return []
    if not (ph.theory == "DFT" or ph.core == "pseudopotential"):
        return ["fit bases: none -- Hartree-Fock's Coulomb and exchange are analytic four-centre integrals"]
    got = gridinfo.fit_bases(record)
    if not got:
        return ["fit bases (Hartree ρ fit, Vxc fit): " + ("not in the library's record yet" if record else
                                                         "Build (or Run) fills this from the library's record")]
    return [f"{f.get('role', 'fit')}: {f.get('name', '?')}" + (f", {f['nFunctions']} functions" if "nFunctions" in f else "")
            for f in got]


def summary(spec: RunSpec, qc) -> list[str]:
    """Lines for the Basis tab: the function count and S's conditioning for a molecule (None-safe on `qc`)."""
    name = basis_name(spec)
    if spec.structure.kind != "molecule":
        return [f"{name}: a crystal's overlap is per k -- its conditioning is the run's [basis trim] report"]
    if qc is None:
        return ["the overlap needs the qchem_py extension (QCHEM_BUILD)"]
    try:
        d = _info(qc, tuple(s.Z for s in spec.structure.sites), tuple(x for s in spec.structure.sites for x in s.xyz), name)
    except Exception as e:                            # noqa: BLE001 -- a readout, never a crash
        return [f"{name}: {e}"]
    cond = d["s_max"] / d["s_min"] if d["s_min"] > 0 else float("inf")
    out = [f"{d['n_functions']} functions ({name}, Cartesian)",
           f"overlap S: min eigenvalue {d['s_min']:.3e}, max {d['s_max']:.3g}, condition number {cond:.3g}"]
    if d["s_min"] < _NEAR_DEPENDENT:
        out.append(f"⚠ near-linear dependence: S's smallest eigenvalue is below {_NEAR_DEPENDENT:g}")
    return out

