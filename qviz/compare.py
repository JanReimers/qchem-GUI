"""
qviz.compare -- run-vs-run comparison operations (Pillar 2 payoff).

Δρ (difference density) is only meaningful between the SAME system computed two
ways (HF vs LDA, AE vs PP): same atoms + positions → same bounding box → aligned
grids, so ρ_A − ρ_B is well defined pointwise. `same_geometry` is the guard; the
UI only offers Δρ against runs that pass it.
"""
from __future__ import annotations
import numpy as np

from .data import ScalarField
from .workspace import Run


def same_geometry(a: Run, b: Run, tol: float = 1e-9) -> bool:
    sa, sb = a.spec.structure, b.spec.structure
    if sa.kind != sb.kind or [s.element for s in sa.sites] != [s.element for s in sb.sites]:
        return False
    coords = (lambda st: [s.xyz for s in st.sites]) if sa.kind == "molecule" else (lambda st: [s.frac for s in st.sites])
    if sa.kind == "crystal" and not np.allclose(sa.lattice_bohr, sb.lattice_bohr, atol=tol):
        return False
    return bool(np.allclose(coords(sa), coords(sb), atol=tol))


def method_tag(r: Run) -> str:
    """A short name for HOW a run was computed: model/basis · back end."""
    p = r.spec.physics
    model = p.functional if p.theory == "DFT" else p.theory
    basis = (r.spec.accuracy.representation or {}).get("basis")
    return f"{model}/{basis} · {r.backend_name}" if basis else f"{model} · {r.backend_name}"


def difference_density(a: Run, b: Run, n: int = 80, pad: float = 5.0) -> ScalarField:
    """Δρ = ρ_a − ρ_b as a signed ScalarField. Requires same_geometry(a, b) so the
    two grids coincide (same origin/spacing/dims)."""
    if not same_geometry(a, b):
        raise ValueError("Δρ requires two runs of the same geometry")
    fa = a.density(n, pad)
    fb = b.density(n, pad)
    d = np.asarray(fa.values) - np.asarray(fb.values)
    return ScalarField(name=f"Δρ ({method_tag(a)} − {method_tag(b)})",
                       values=d, origin=fa.origin, spacing=fa.spacing, signed=True)
