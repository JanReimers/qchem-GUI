"""
qviz.basisinfo -- what a spec's orbital basis IS, before any SCF: the shells on each element (read from qchem's
BasisSetData file); its size and the overlap's conditioning are the library's own record (a Build's or a run's basis section,
gridinfo.conditioning), never recomputed here.

A small S eigenvalue is a near-linear dependence: diffuse functions on neighbouring atoms spanning almost the same
space.  The condition number s_max/s_min says how much the SCF's orthogonalization amplifies round-off.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from . import gridinfo
from .backends import gaussian_basis
from .backends.solid2 import basis_for
from .spec import RunSpec

_L = "spdfgh"
_CART = (1, 3, 6, 10, 15, 21)                     # Cartesian components per l (the facade's molecular basis is Cartesian)


def basis_name(spec: RunSpec) -> str:
    """The basis SOLID² will use: the stated one, else its ε rule's."""
    return (spec.accuracy.representation or {}).get("basis") or basis_for(spec.accuracy.epsilon_Ha_per_atom)


def basis_file(spec: RunSpec) -> Optional[str]:
    """The downloaded .bsd a spec runs on (representation['solid2']['basis_file'], qviz.bse), None for a shipped basis."""
    return ((spec.accuracy.representation or {}).get("solid2") or {}).get("basis_file")


def element_rows(spec: RunSpec) -> list[tuple[str, int, str, int]]:
    """(element, atoms of it, contracted shells like "3s 2p 1d", Cartesian functions per atom) per element in the
    structure; shells "no block in <basis>" (and 0 functions) for an element the file lacks."""
    name = basis_name(spec)
    file = basis_file(spec)
    try:
        data = gaussian_basis.read_bsd_file(file) if file else gaussian_basis.read_bsd(name)
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


def fits(spec: RunSpec, record: Optional[dict]) -> list[str]:
    """The Basis tab's other bases -- what the run fits ρ (Hartree) and Vxc in -- as the record files them
    (hamiltonian.<term>.fitbasis: [{role, name, nFunctions}]).  Hartree-Fock has none: its Coulomb and exchange are analytic."""
    ph = spec.physics
    if spec.structure.kind == "molecule" and not (ph.theory == "DFT" or ph.core == "pseudopotential"):
        return ["fit bases: none -- Hartree-Fock's Coulomb and exchange are analytic four-centre integrals"]
    got = gridinfo.fit_bases(record)
    if not got:
        return ["fit bases (Hartree ρ fit, Vxc fit): Build (or Run) fills this from the library's record"]
    return [f"{f.get('role', 'fit')}: {f.get('name', '?')}" + (f", {f['nFunctions']} functions" if "nFunctions" in f else "")
            for f in got]


def summary(spec: RunSpec) -> list[str]:
    """The Basis tab's lines before the library has built this spec: its conditioning is the build's own record
    (basis.perIrrep, gridinfo.conditioning), so Build (or Run) fills them."""
    name = basis_name(spec)
    if spec.structure.kind != "molecule":
        return [f"{name}: a crystal's overlap is per k -- Build (or Run) shows its conditioning (the [basis trim] report)"]
    return [f"{name}: Build (or Run) shows the function count and the overlap's conditioning per irrep"]
