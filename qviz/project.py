"""
qviz.project -- the workspace / save-restore concept, on HDF5.

A *project* is a single self-describing .h5 file (or a directory with one):
  /meta            attrs: title, backend, created
  /structure       symbols, positions, numbers, [lattice]
  /fields/<name>   dataset + attrs(origin, spacing, signed)   <- densities, MOs
  /scf             table: iteration, energy, dE, commutator, drho
  /viewstate       attrs: camera, active iso level, slice axis/pos (JSON)

HDF5 is the right container: native readers in C++ (HighFive) and Python (h5py),
self-describing, and it lets the GUI reopen a finished run with *zero* compute.
This mirrors what the real app would persist after an SCF.
"""
from __future__ import annotations

import json
import datetime as _dt
import numpy as np
import h5py

from .data import Structure, ScalarField, SCFStep


def save(path: str, *, title: str, backend: str,
         structure: Structure,
         fields: dict[str, ScalarField] | None = None,
         scf: list[SCFStep] | None = None,
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
            arr = np.array([(s.iteration, s.energy, s.dE, s.commutator, s.drho)
                            for s in scf],
                           dtype=[("iteration", "i4"), ("energy", "f8"),
                                  ("dE", "f8"), ("commutator", "f8"),
                                  ("drho", "f8")])
            h.create_dataset("scf", data=arr)

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
        out["scf"] = h["scf"][:] if "scf" in h else None
        out["viewstate"] = json.loads(h["viewstate"].attrs.get("json", "{}"))
    return out


# --------------------------------------------------------------------------
# Workspace (multi-run) save / restore -- the .qproj.h5 unit
# --------------------------------------------------------------------------
def save_workspace(path: str, ws) -> None:
    """Persist a Workspace: one group per Run (its RunSpec + energy + status).
    Runs recompute on load (fast for small molecules); caching the sampled fields
    for true instant-open is a later enhancement."""
    with h5py.File(path, "w") as h:
        h.attrs.update(format="qviz-workspace/1", title=ws.title,
                       created=_dt.datetime.now().isoformat(timespec="seconds"),
                       selected=(ws.runs.index(ws.selected) if ws.selected in ws.runs else -1))
        g = h.create_group("runs")
        for i, r in enumerate(ws.runs):
            s = r.spec
            rg = g.create_group(f"run{i:03d}")
            rg.attrs.update(label=s.label, method=s.method, basis=s.basis,
                            energy=(float(r.energy) if r.energy is not None else np.nan),
                            status=r.status.value)
            rg.create_dataset("numbers", data=np.asarray(s.numbers, int))
            rg.create_dataset("positions", data=np.asarray(s.positions, float))


def load_workspace(path: str, make_backend):
    """Rebuild a Workspace from a .qproj.h5, recomputing each Run via make_backend."""
    from .workspace import Workspace, RunSpec
    with h5py.File(path, "r") as h:
        ws = Workspace(str(h.attrs.get("title", "Untitled")), make_backend=make_backend)
        g = h["runs"]
        for key in sorted(g):
            rg = g[key]
            ws.add_run(RunSpec(
                label=str(rg.attrs["label"]),
                numbers=tuple(int(z) for z in rg["numbers"][:]),
                positions=tuple(float(x) for x in rg["positions"][:]),
                method=str(rg.attrs["method"]), basis=str(rg.attrs["basis"])))
        sel = int(h.attrs.get("selected", -1))
        if 0 <= sel < len(ws.runs):
            ws.select(ws.runs[sel])
    return ws
