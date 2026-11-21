"""
qviz.basisinfo -- what a spec's orbital basis IS, before any SCF.  The shells on each element, its size and the overlap's
conditioning are the library's own record (a Build's or a run's basis section: basis.elements, gridinfo.conditioning), never
recomputed here.  The one exception is a CRYSTAL's shells per element: the library files no basis.elements for a solid yet
(its Gaussian basis is built outside a "basis" scope -- OpenWork, lane S), so those are still read from the .bsd file.

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
_CART = (1, 3, 6, 10, 15, 21)                     # Cartesian components per l (a crystal's .bsd read only)


def basis_name(spec: RunSpec) -> str:
    """The basis SOLID² will use: the stated one, else its ε rule's (an atom's: its radial family, solid2_atom)."""
    if spec.structure.kind == "atom":
        from .backends.solid2_atom import radial_type
        return radial_type(spec)
    return (spec.accuracy.representation or {}).get("basis") or basis_for(spec.accuracy.epsilon_Ha_per_atom)


def basis_file(spec: RunSpec) -> Optional[str]:
    """The downloaded .bsd a spec runs on (representation['solid2']['basis_file'], qviz.bse), None for a shipped basis."""
    return ((spec.accuracy.representation or {}).get("solid2") or {}).get("basis_file")


def element_rows(spec: RunSpec, record: Optional[dict] = None) -> list[tuple[str, int, str, int]]:
    """(element, atoms of it, contracted shells like "3s 2p 1d", functions per atom) per element: the library's own rows
    (basis.elements) from \a record (a Build's or a run's).  A molecule with no record yet has none (summary() says Build fills
    them).  A crystal, whose build files none yet, reads its .bsd: shells "no block in <basis>" (0 functions) for an element the
    file lacks, functions counted Cartesian."""
    lib = gridinfo.basis_elements(record)
    if lib:
        return [(r["element"], r["atoms"], r["shells"], r["nFunctions"]) for r in lib]
    if spec.structure.kind != "crystal":
        return []
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
    if spec.structure.kind != "crystal" and not (ph.theory == "DFT" or ph.core == "pseudopotential"):
        return ["fit bases: none -- Hartree-Fock's Coulomb and exchange are analytic four-centre integrals"]
    got = gridinfo.fit_bases(record)
    if not got:
        return ["fit bases (Hartree ρ fit, Vxc fit): Build (or Run) fills this from the library's record"]
    return [f"{f.get('role', 'fit')}: {f.get('name', '?')}" + (f", {f['nFunctions']} functions" if "nFunctions" in f else "")
            for f in got] + gridinfo.coulomb_fit_details(record)


def summary(spec: RunSpec) -> list[str]:
    """The Basis tab's lines before the library has built this spec: its conditioning is the build's own record
    (basis.perIrrep, gridinfo.conditioning), so Build (or Run) fills them."""
    name = basis_name(spec)
    if spec.structure.kind == "atom":
        return [f"{name} radial basis (the radial atom: blocked by l, m analytic): Build (or Run) shows its functions per l "
                "and the overlap's conditioning"]
    if spec.structure.kind != "molecule":
        return [f"{name}: a crystal's overlap is per k -- Build (or Run) shows its conditioning (the [basis trim] report)"]
    return [f"{name}: Build (or Run) shows the shells per element, the function count and the overlap's conditioning per irrep"]
