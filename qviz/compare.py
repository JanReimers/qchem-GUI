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
    return (a.spec.numbers == b.spec.numbers
            and len(a.spec.positions) == len(b.spec.positions)
            and np.allclose(a.spec.positions, b.spec.positions, atol=tol))


def difference_density(a: Run, b: Run, n: int = 80, pad: float = 5.0) -> ScalarField:
    """Δρ = ρ_a − ρ_b as a signed ScalarField. Requires same_geometry(a, b) so the
    two grids coincide (same origin/spacing/dims)."""
    if not same_geometry(a, b):
        raise ValueError("Δρ requires two runs of the same geometry")
    fa = a.density(n, pad)
    fb = b.density(n, pad)
    d = np.asarray(fa.values) - np.asarray(fb.values)
    return ScalarField(name=f"Δρ ({a.spec.method}−{b.spec.method})",
                       values=d, origin=fa.origin, spacing=fa.spacing, signed=True)
