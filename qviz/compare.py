"""
qviz.compare -- run-vs-run comparison operations (Pillar 2 payoff).

Δρ (difference density) is only meaningful between the SAME system computed two
ways (HF vs LDA, AE vs PP): same atoms + positions → same bounding box → aligned
grids, so ρ_A − ρ_B is well defined pointwise. `same_geometry` is the guard; the
UI only offers Δρ against runs that pass it (and that both offer a density).

Cross-code (S4): `cross_code` puts ONE spec on every back end that can run it, `legs` finds a run's
siblings, and `code_table` lays their energies beside the settings each code resolved.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from .data import ScalarField
from .workspace import Run, Workspace


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
    if not (a.offers("density") and b.offers("density")):
        raise ValueError("Δρ requires two runs that both give a density")
    fa = a.density(n, pad)
    fb = b.density(n, pad)
    d = np.asarray(fa.values) - np.asarray(fb.values)
    return ScalarField(name=f"Δρ ({method_tag(a)} − {method_tag(b)})",
                       values=d, origin=fa.origin, spacing=fa.spacing, signed=True)


# --------------------------------------------------------------------------
# One spec, several codes (doc/RunSpecPlan.md S4)
# --------------------------------------------------------------------------
def cross_code(ws: Workspace, spec, backends: Optional[list[str]] = None) -> list[Run]:
    """Add `spec` once per back end that can run it (or per named back end), PENDING: the caller computes them --
    the GUI on its worker pool, a script with run.compute().  A synthetic back end (Fake) is never a comparison leg."""
    names = backends or [n for n, v in ws.registry.verdicts(spec).items()
                         if v and not ws.registry.get(n).capabilities().synthetic]
    return [ws.add_run(spec, backend=n, compute=False) for n in names]


def legs(ws: Workspace, run: Run) -> list[Run]:
    """Every run in `ws` of the SAME spec as `run` (its cross-code legs), `run` first."""
    return [run] + [r for r in ws.runs if r is not run and r.spec == run.spec]


def code_table(runs: list[Run]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """The side-by-side panel: (column heads, rows).  Energies first -- each leg's ΔE against the FIRST, total and
    per atom -- then the iterations, then the union of every leg's resolved settings, blank where a code has none
    (a cross-code comparison is only as good as the settings it was made at, so they sit beside the numbers)."""
    heads = [r.backend_name for r in runs]
    E = [r.energy if r.is_ready else None for r in runs]
    nat = len(runs[0].spec.structure.sites) if runs else 1
    fmt = lambda x, f: "—" if x is None else format(x, f)
    d = [None if (e is None or E[0] is None) else e - E[0] for e in E]
    rows = [("status", [r.status.value for r in runs]),
            ("E (Ha)", [fmt(e, ".8f") for e in E]),
            (f"ΔE vs {heads[0] if heads else ''} (μHa)", [fmt(None if x is None else 1e6 * x, "+.1f") for x in d]),
            ("ΔE per atom (μHa)", [fmt(None if x is None else 1e6 * x / nat, "+.1f") for x in d]),
            ("SCF iterations", [str(len(r.result.scf)) if r.result else "—" for r in runs])]
    resolved = [(r.result.resolved if r.result else r.job.resolved if r.job else r.backend.translate(r.spec).resolved)
                for r in runs]                     # a finished run's own: what its record states it used
    keys = list(dict.fromkeys(k for res in resolved for k in res))
    rows += [(k, [str(res.get(k, "")) for res in resolved]) for k in keys]
    return heads, rows
