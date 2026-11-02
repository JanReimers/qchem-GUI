"""
qviz.gridinfo -- the grids a spec's ε resolves to: what is PLANNED (read from the spec and the back end's ε rules, no
library call) and what was BUILT (read from the library's own run record, report::GlobalReport, as qchem_py.report()
hands it back).

The record is the library speaking for itself: every provider files what it built under the run's "grids" section
(the GPW density ladder, the density / Vxc fit rasters, the grid audits) and the "folds" section (the molecular XC
mesh's point count).  This module only picks rows out of it; it never re-derives a grid.

A crystal's grids exist once deck::Build has run, before any SCF -- `preview` builds one for the Setup tab.  A
molecule's XC mesh is built inside its SCF, so its point count arrives with the run.
"""
from __future__ import annotations

import json
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
            return ["no XC mesh: Hartree-Fock exchange is analytic"]
        own = code_block(spec, "solid2")
        mr, ma = mesh_for(ac.epsilon_Ha_per_atom)
        mr, ma = int(own.get("mesh_radial", mr)), int(own.get("mesh_angular", ma))
        why = "stated" if own else f"from ε = {ac.epsilon_Ha_per_atom:g} Ha/atom (provisional rule)"
        return [f"XC mesh: {mr} radial × Lebedev {ma} per atom, Becke-partitioned ({why})"]
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


def built(record: Optional[dict]) -> tuple[list[str], list[tuple]]:
    """(summary lines, density-ladder rows) from a run record; ([], []) when it holds no grids."""
    if not record:
        return [], []
    lines, rows = [], []
    for build in (True, False):
        for run in _runs(record, build):
            mesh = (run.get("folds") or {}).get("XC mesh")
            if mesh:
                lines.append(f"XC mesh: {mesh['raw']:,} points"
                             + (f", {mesh['reps']:,} after the {mesh['ops']}-operation fold" if mesh.get("ops") else ""))
    g = _grids(record)
    if g:
        if "densityEcut" in g:
            lines.append(f"density Ecut floor: {g['densityEcut']:g} Ha (cutoff factor {g.get('cutoffFactor', 0):g})")
        q = (g.get("xcQuadrature") or {}).get("kind")
        if q:
            lines.append(f"XC quadrature: {q}")
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


def preview(spec: RunSpec, qc) -> dict:
    """A crystal's run record after deck::Build alone (no SCF): its grids and basis, as the run will build them.  Raises
    with the library's text on a refusal."""
    if spec.structure.kind != "crystal":
        raise ValueError("a molecule's XC mesh is built inside its SCF; its record arrives with the run")
    qc.clear_report()
    qc.SolidRun(json.dumps(solid2_deck.to_deck(spec)))
    return json.loads(qc.report())
