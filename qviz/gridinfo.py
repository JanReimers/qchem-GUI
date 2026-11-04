"""
qviz.gridinfo -- the grids a spec's ε resolves to: what is PLANNED (read from the spec and the back end's ε rules, no
library call) and what was BUILT (read from the library's own run record, report::GlobalReport, as qchem_py.report()
hands it back).

The record is the library speaking for itself: every provider files what it built under the run's "grids" section
(the GPW density ladder, the density / Vxc fit rasters, the grid audits) and the "folds" section (the molecular XC
mesh's point count).  This module only picks rows out of it; it never re-derives a grid.

The record comes from a run, or from SOLID²'s Build (every pre-SCF step, no SCF: Solid2Backend.build).
"""
from __future__ import annotations

from typing import Optional

from .backends import solid2_deck
from .backends.solid2 import code_block, mesh_for
from .spec import RunSpec

LADDER_COLUMNS = ("Level", "FFT grid", "Ecut (Ha)", "G vectors", "Role")


def planned(spec: RunSpec) -> list[str]:
    """What the spec asks for, before anything is built: the molecular XC mesh, or a crystal's k-mesh and density Ecut."""
    st, ph, ac = spec.structure, spec.physics, spec.accuracy
    if st.kind == "molecule":
        if not (ph.theory == "DFT" or ph.core == "pseudopotential"):
            return ["no XC mesh: Hartree-Fock exchange is analytic",
                    "Hartree: analytic four-centre integrals, no mesh"]
        own = code_block(spec, "solid2")
        mr, ma = mesh_for(ac.epsilon_Ha_per_atom)
        mr, ma = int(own.get("mesh_radial", mr)), int(own.get("mesh_angular", ma))
        why = "stated" if own else f"from ε = {ac.epsilon_Ha_per_atom:g} Ha/atom (provisional rule)"
        return [f"XC mesh: {mr} radial × Lebedev {ma} per atom, Becke-partitioned ({why})",
                "Hartree fit mesh: the Coulomb fit basis's own mesh, built with the XC mesh's parameters"]
    deck = solid2_deck.to_deck(spec)
    k = deck.get("kmesh", [1, 1, 1])
    ecut = (deck.get("solid") or {}).get("densityEcut")
    return [f"k-mesh: {k[0]}×{k[1]}×{k[2]}",
            f"density Ecut: {ecut:g} Ha (stated)" if ecut else "density Ecut: the library's floor (cutoff factor × the basis's tightest exponent)"]


def _runs(record: dict, build: bool) -> list[dict]:
    """The record's runs, the build runs ("<name> build@...") or the SCF runs; nested pilot runs are skipped."""
    out = []
    for key, run in (record or {}).items():
        name = key.rsplit("@", 1)[0]
        if name.startswith("gridAudit"):
            continue
        if name.endswith(" build") == build:
            out.append(run)
    return out


def _grids(record: dict) -> Optional[dict]:
    for build in (True, False):
        for run in _runs(record, build):
            if run.get("grids"):
                return run["grids"]
    return None


def fit_bases(record: Optional[dict]) -> list[dict]:
    """The fit bases the Hamiltonian built, as the record files them ([]: none filed)."""
    return ((_grids(record) or {}).get("hamiltonian") or {}).get("fits") or []


#: the meshes a run files under grids, as the Grids tab names them
_MESHES = (("xcQuadrature", "XC mesh"), ("ppQuadrature", "PP mesh"), ("auxiliaryFit", "Hartree fit mesh"))


def _mesh_line(what: str, m: dict) -> str:
    """One mesh as its report states it: atom-centred (radial × directions) or uniform (N)."""
    if "nDirs" in m:
        return (f"{what}: {m.get('kind', 'Becke')}, {m.get('natom', '?')} atoms × {m.get('nRadial', m.get('radial'))} radial"
                f" × {m['nDirs']} directions (Lebedev L={m.get('L', '?')}) = {m['points']:,} points"
                + (f", {m['dropped']:,} dropped" if m.get("dropped") else ""))
    n = m.get("N")
    return f"{what}: {m.get('kind', '')} " + (f"{n[0]}×{n[1]}×{n[2]} " if n else "") + f"= {m['points']:,} points"


def built(record: Optional[dict]) -> tuple[list[str], list[tuple]]:
    """(summary lines, density-ladder rows) from a run record; ([], []) when it holds no grids."""
    if not record:
        return [], []
    lines, rows = [], []
    g = _grids(record)
    q = (g or {}).get("xcQuadrature") or {}
    for key, what in _MESHES:                         # each mesh's own report (it files itself, lane B b2dabdd0)
        m = (g or {}).get(key) or {}
        if "points" in m:
            lines.append(_mesh_line(what, m))
    for build in (True, False):
        for run in _runs(record, build):
            mesh = (run.get("folds") or {}).get("XC mesh")
            if mesh and "points" not in q:
                lines.append(f"XC mesh: {mesh['raw']:,} points"
                             + (f", {mesh['reps']:,} after the {mesh['ops']}-operation fold" if mesh.get("ops") else ""))
    if g:
        if "densityEcut" in g:
            lines.append(f"density Ecut floor: {g['densityEcut']:g} Ha (cutoff factor {g.get('cutoffFactor', 0):g})")
        if q.get("kind") and "points" not in q:
            lines.append(f"XC quadrature: {q['kind']}")
        for key, what in (("densityFit", "density fit"), ("vxcFitGrid", "Vxc fit")):
            f = g.get(key)
            if f and "N" in f:
                n = f["N"]
                lines.append(f"{what}: {f.get('kind', '')} {n[0]}×{n[1]}×{n[2]} = {f.get('points', n[0] * n[1] * n[2]):,} points"
                             + (f", spacing {f['dr'][0]:.3f} bohr" if f.get("dr") else ""))
        audits = [(k, a) for k, a in g.items() if isinstance(a, dict) and "status" in a]
        for k, a in audits:
            lines.append(f"{'⚠ ' if a['status'] != 'OK' else ''}audit {a.get('term', k)}: {a['status']}")
        for L in g.get("ladder") or []:
            n = L["N"]
            rows.append((L["level"], f"{n[0]}×{n[1]}×{n[2]}", f"{L['ecut']:g}", L["nG"], L["role"]))
    return lines, rows


def conditioning(record: Optional[dict]) -> list[str]:
    """Per-irrep overlap conditioning from a record's basis section (a crystal's is per k): the [basis trim] facts."""
    out = []
    for build in (True, False):
        for run in _runs(record, build):
            basis = run.get("basis") or {}
            for b in basis.get("perIrrep") or []:
                if "lambdaMin" in b:
                    out.append(f"{b['irrep'].strip()}: {b['nFunctions']} functions, S min eigenvalue {b['lambdaMin']:.3e}, "
                               f"condition number {b['cond']:.3g}" + ("  ⚠ near-linear dependence" if b["lambdaMin"] < 1e-6 else ""))
            if out and "nFunctions" in basis:
                out.insert(0, f"{basis['nFunctions']} functions" + (f" ({basis['name']})" if basis.get("name") else ""))
            removed = (run.get("basis") or {}).get("removed")
            if removed:
                out.append(f"[basis trim] removed {len(removed)} near-dependent combination(s)")
            if out:
                return out
    return out
