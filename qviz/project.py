"""
qviz.project -- the workspace / save-restore concept, on HDF5.

A *project* is a single self-describing .h5 file (or a directory with one):
  /meta            attrs: title, backend, created
  /structure       symbols, positions, numbers, [lattice]
  /fields/<name>   dataset + attrs(origin, spacing, signed)   <- densities, MOs
  /scf             table: iteration, energy, then one column per cell (attrs titles, scales, units: the code's own)
  /viewstate       attrs: camera, active iso level, slice axis/pos (JSON)

HDF5 is the right container: native readers in C++ (HighFive) and Python (h5py),
self-describing, and it lets the GUI reopen a finished run with *zero* compute.
This mirrors what the real app would persist after an SCF.
"""
from __future__ import annotations

import json
import math
import datetime as _dt
import numpy as np
import h5py

from .data import Structure, ScalarField, SCFCell, SCFRow


def save(path: str, *, title: str, backend: str,
         structure: Structure,
         fields: dict[str, ScalarField] | None = None,
         scf: list[SCFRow] | None = None,
         viewstate: dict | None = None) -> None:
    with h5py.File(path, "w") as h:
        m = h.create_group("meta")
        m.attrs.update(title=title, backend=backend,
                       created=_dt.datetime.now().isoformat(timespec="seconds"),
                       format="qviz-project/1")

        g = h.create_group("structure")
        g.create_dataset("symbols", data=np.array(structure.symbols, dtype="S4"))
        g.create_dataset("positions", data=structure.positions)
        g.create_dataset("numbers", data=structure.numbers)
        if structure.lattice is not None:
            g.create_dataset("lattice", data=structure.lattice)

        ff = h.create_group("fields")
        for name, f in (fields or {}).items():
            d = ff.create_dataset(name, data=f.values, compression="gzip")
            d.attrs.update(origin=f.origin, spacing=f.spacing,
                           signed=f.signed, display_name=f.name)

        if scf:
            titles = list(dict.fromkeys(c.title for s in scf for c in s.cells))   # the union, in first-seen order
            cell = {c.title: c for s in scf for c in s.cells}
            d = h.create_dataset("scf", data=np.array([[s.iteration, s.energy] + [s.value(t) for t in titles] for s in scf]))
            d.attrs.update(titles=json.dumps(titles), scales=json.dumps([cell[t].scale for t in titles]),
                           units=json.dumps([cell[t].unit for t in titles]))

        h.create_group("viewstate").attrs["json"] = json.dumps(viewstate or {})


def load(path: str) -> dict:
    """Reopen a project. Returns structure, fields, scf rows, viewstate."""
    out: dict = {}
    with h5py.File(path, "r") as h:
        out["meta"] = dict(h["meta"].attrs)
        g = h["structure"]
        out["structure"] = Structure(
            symbols=[s.decode() for s in g["symbols"][:]],
            positions=g["positions"][:],
            numbers=g["numbers"][:],
            lattice=g["lattice"][:] if "lattice" in g else None)
        out["fields"] = {}
        for name, d in h.get("fields", {}).items():
            out["fields"][name] = ScalarField(
                name=d.attrs["display_name"], values=d[:],
                origin=tuple(d.attrs["origin"]), spacing=tuple(d.attrs["spacing"]),
                signed=bool(d.attrs["signed"]))
        out["scf"] = _scf_rows(h["scf"]) if "scf" in h else None
        out["viewstate"] = json.loads(h["viewstate"].attrs.get("json", "{}"))
    return out


def _scf_rows(d) -> list[SCFRow]:
    """/scf back into SCFRows (numbers only: the printed text is not stored)."""
    titles, scales, units = (json.loads(d.attrs[k]) for k in ("titles", "scales", "units"))
    return [SCFRow(int(r[0]), float(r[1]), tuple(SCFCell(t, float(v), s, unit=u) for t, s, u, v in zip(titles, scales, units, r[2:])))
            for r in d[:]]


# --------------------------------------------------------------------------
# Workspace (multi-run) save / restore -- the .qproj.h5 unit
# --------------------------------------------------------------------------
WORKSPACE_FORMAT = "qviz-workspace/2"   # /2: one RunSpec (JSON) + its back end per run, and its result when it has one
                                        # (an optional `result` dataset).  /1 files: scripts/convert_workspace_v1.py


def save_workspace(path: str, ws) -> None:
    """Persist a Workspace: one group per Run -- its RunSpec (JSON), the back end, status, error, and the RESULT when it
    has one (energy, SCF trace, levels, resolved settings, the code's run record: JSON in a `result` dataset), so a
    reopen restores it with zero compute.  3D fields are not stored: they need the live calculation (Recompute)."""
    with h5py.File(path, "w") as h:
        h.attrs.update(format=WORKSPACE_FORMAT, title=ws.title,
                       created=_dt.datetime.now().isoformat(timespec="seconds"),
                       selected=(ws.runs.index(ws.selected) if ws.selected in ws.runs else -1))
        g = h.create_group("runs")
        for i, r in enumerate(ws.runs):
            rg = g.create_group(f"run{i:03d}")
            rg.attrs.update(spec=r.spec.to_json(), backend=r.backend_name,
                            energy=(float(r.energy) if r.energy is not None else np.nan),
                            status=r.status.value, error=r.error or "")
            if r.result is not None and r.status.value in ("ready", "failed"):
                rg.create_dataset("result", data=json.dumps(_result_dict(r.result), default=_json_default))
    ws.path = path


def load_workspace(path: str, registry, compute: bool = True):
    """Rebuild a Workspace from a .qproj.h5.  A run saved WITH its result is restored as it was (no compute; its 3D
    fields come back when it is recomputed).  A run saved without one: a finished one is recomputed (compute=True now,
    for library use; compute=False leaves it pending so a GUI computes it off-thread), an unfinished one (the app quit
    while it ran) comes back failed, saying so.  A run whose back end is not registered goes to the registry's default."""
    from .workspace import RunStatus, Workspace
    from .spec import RunSpec
    with h5py.File(path, "r") as h:
        fmt = str(h.attrs.get("format", ""))
        if fmt != WORKSPACE_FORMAT:
            raise ValueError(f"{path}: workspace format {fmt!r}, this reader knows {WORKSPACE_FORMAT!r}"
                             + (" -- convert it with scripts/convert_workspace_v1.py" if fmt == "qviz-workspace/1" else ""))
        ws = Workspace(str(h.attrs.get("title", "Untitled")), registry=registry)
        g = h["runs"]
        for key in sorted(g):
            rg = g[key]
            name = str(rg.attrs["backend"])
            spec = RunSpec.from_json(str(rg.attrs["spec"]))
            status, error = str(rg.attrs.get("status", "")), str(rg.attrs.get("error", "")) or None
            if "result" in rg:
                run = ws.add_run(spec, backend=name if name in registry.names() else None, compute=False)
                raw = rg["result"][()]
                run.restore(_result_from(json.loads(raw.decode() if isinstance(raw, bytes) else raw), spec),
                            RunStatus(status), error)
            elif status == RunStatus.READY.value:
                ws.add_run(spec, backend=name if name in registry.names() else None, compute=compute)
            else:
                run = ws.add_run(spec, backend=name if name in registry.names() else None, compute=False)
                run.restore(None, RunStatus.FAILED, error or "interrupted: the app quit while this run was pending or running")
        sel = int(h.attrs.get("selected", -1))
        if 0 <= sel < len(ws.runs):
            ws.select(ws.runs[sel])
    ws.path = path
    return ws


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _result_dict(r) -> dict:
    """The parts of a Result that outlive its calculation (no fields, no orbital sampler: those need the live code)."""
    return {"converged": r.converged, "energy_Ha": r.energy_Ha, "failure": r.failure, "polarized": r.polarized,
            "levels": r.levels, "resolved": r.resolved, "provenance": r.provenance, "record": r.record,
            "scf": [{"iteration": s.iteration, "energy": s.energy,
                     "cells": [{"title": c.title, "value": None if math.isnan(c.value) else c.value, "scale": c.scale,
                                "text": c.text, "tolerance": c.tolerance, "unit": c.unit} for c in s.cells]}
                    for s in r.scf]}


def _result_from(d: dict, spec):
    from .backends.base import Result, data_structure
    scf = [SCFRow(s["iteration"], s["energy"],
                  tuple(SCFCell(c["title"], math.nan if c["value"] is None else c["value"], c["scale"], c["text"],
                                c["tolerance"], c["unit"]) for c in s["cells"])) for s in d["scf"]]
    return Result(converged=d["converged"], energy_Ha=d["energy_Ha"], structure=data_structure(spec), scf=scf,
                  levels=d["levels"], polarized=d["polarized"], resolved=d["resolved"], provenance=d["provenance"],
                  failure=d["failure"], record=d["record"])
