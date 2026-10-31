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
WORKSPACE_FORMAT = "qviz-workspace/2"   # /2: one RunSpec (JSON) + its back end per run.  /1 files: scripts/convert_workspace_v1.py


def save_workspace(path: str, ws) -> None:
    """Persist a Workspace: one group per Run -- its RunSpec as JSON, the back end it ran on, energy, status.
    Runs recompute on load (fast for small molecules); caching the sampled fields for true instant-open is a
    later enhancement."""
    with h5py.File(path, "w") as h:
        h.attrs.update(format=WORKSPACE_FORMAT, title=ws.title,
                       created=_dt.datetime.now().isoformat(timespec="seconds"),
                       selected=(ws.runs.index(ws.selected) if ws.selected in ws.runs else -1))
        g = h.create_group("runs")
        for i, r in enumerate(ws.runs):
            rg = g.create_group(f"run{i:03d}")
            rg.attrs.update(spec=r.spec.to_json(), backend=r.backend_name,
                            energy=(float(r.energy) if r.energy is not None else np.nan),
                            status=r.status.value)


def load_workspace(path: str, registry, compute: bool = True):
    """Rebuild a Workspace from a .qproj.h5.  compute=True recomputes each Run now (lib use); compute=False leaves
    them pending so a GUI can compute off-thread.  A run whose back end is not registered here goes to the registry's
    default for its spec."""
    from .workspace import Workspace
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
            ws.add_run(RunSpec.from_json(str(rg.attrs["spec"])),
                       backend=name if name in registry.names() else None, compute=compute)
        sel = int(h.attrs.get("selected", -1))
        if 0 <= sel < len(ws.runs):
            ws.select(ws.runs[sel])
    return ws
