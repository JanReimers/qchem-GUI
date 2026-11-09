"""Crystal import / export and the materials picker, through the library (qchem_py.materials).

THE ENTRY IS THE INTERNAL FORMAT (OpenWork §2, ruled by the user 2026-11-09).  A CIF, POSCAR or extended xyz file is
translated by the LIBRARY (qchem.CrystalFiles) into one materials.json entry -- the GUI never parses a crystal file -- and
the picker lists the library's layers: the SHIPPED materials (curated) and the user's own library (~/.qchem/materials.json,
where "Save to my materials" writes).  This module turns an entry into a crystal spec and a spec back into an entry; nothing
else.  Site spins are stated after import (the Structure tab), and ride into a saved entry as its `spin` decoration.
"""
from __future__ import annotations

import json
import pathlib
from typing import Optional

from .spec import RunSpec


#: The user's own library.  THE GUI resolves it: the C++ library reads no environment and is handed every path it layers over the
#: shipped file (controller ruling 2026-11-09).  None = ~/.qchem/materials.json; a test sets a temporary file here.
USER_LIBRARY: Optional[str] = None


def user_library() -> str:
    return USER_LIBRARY or str(pathlib.Path.home() / ".qchem" / "materials.json")


def _layers() -> list[dict]:
    return [{"layer": "user", "path": user_library()}]


class MaterialsError(RuntimeError):
    """The library refused (an import it cannot represent, a name already taken, ...), in its own words."""


#: the export formats, as the library's `export` op names them -> (the dialog's filter, the default suffix)
FORMATS = {"cif": ("CIF (*.cif)", ".cif"), "poscar": ("POSCAR (*.vasp *.poscar POSCAR*)", ".vasp"),
           "xyz": ("Extended xyz (*.xyz *.extxyz)", ".xyz")}
#: what File > Import crystal offers (the library's FormatOf decides; this is only the dialog's filter)
IMPORT_FILTER = "Crystal files (*.cif *.xyz *.extxyz *.vasp *.poscar POSCAR* CONTCAR*);;All files (*)"


def _qc():
    from .backends.solid2 import _import_qchem_py
    mod, why = _import_qchem_py()
    if mod is None:
        raise MaterialsError(f"the library is not available: {why}")
    return mod


def ask(op: str, **kw) -> dict:
    """One request to the library's materials face (qcb_materials); a refusal raises MaterialsError with its text."""
    out = json.loads(_qc().materials(json.dumps({"op": op, **kw})))
    if "error" in out:
        raise MaterialsError(out["error"])
    return out


def available() -> bool:
    try:
        _qc()
        return True
    except MaterialsError:
        return False


def listing() -> list[dict]:
    """Every material over the shipped file and the user's library, shipped first: [{name, layer, file, entry, lattice_bohr}]."""
    return ask("list", libraries=_layers())["materials"]


def spec_of(label: str, entry: dict, lattice_bohr, source: dict) -> RunSpec:
    """The crystal spec an entry denotes: its cell (rows a, b, c, bohr -- the library resolved a Bravais entry) and its sites,
    with the default physics a picked crystal starts from (LDA with the pseudopotentials; the user edits the draft).  The
    entry's `spin` decoration becomes the site moments (±1 μB: only the signs reach a deck)."""
    spins = [a.get("spin", 0) for a in entry["atoms"]]
    spin = {"polarized": any(spins), **({"site_moments": [float(s) for s in spins]} if any(spins) else {})}
    return RunSpec.from_dict({
        "label": label,
        "structure": {"kind": "crystal", "lattice_bohr": [list(r) for r in lattice_bohr],
                      "sites": [{"element": a["el"], "frac": list(a["frac"])} for a in entry["atoms"]], "source": source},
        "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential", "spin": spin},
        "accuracy": {"kmesh": [1, 1, 1]}}).validate()


def pick(material: dict) -> RunSpec:
    """A listed material as a draft spec.  `source` names it (a shipped name lets the deck refer to it by name) and keeps a
    user entry's provenance."""
    e = material["entry"]
    source = {"materials": material["name"], "layer": material["layer"]}
    if "provenance" in e:
        source["provenance"] = e["provenance"]
    if "comment" in e:
        source["comment"] = e["comment"]
    return spec_of(material["name"], e, material["lattice_bohr"], source)


def import_file(path: str) -> tuple[RunSpec, list[str]]:
    """File > Import crystal: the library translates the file; (the draft spec, the library's warnings)."""
    out = ask("import", path=path)
    e = out["entry"]
    return spec_of(out["name"], e, out["lattice_bohr"], {"import": e["provenance"], "comment": e.get("comment", "")}), out["warnings"]


def entry_of(spec: RunSpec) -> dict:
    """The materials.json entry a crystal spec IS: its cell as vectors_bohr, its sites, the signs of its site moments as the
    spin decoration, and the provenance its source carries (an import's, or a user entry's)."""
    st = spec.structure
    if st.kind != "crystal":
        raise MaterialsError("only a crystal is a material (a molecule exports as .xyz)")
    m = spec.physics.spin.site_moments if spec.physics.spin.polarized else None
    atoms = []
    for i, s in enumerate(st.sites):
        a = {"el": s.element, "frac": list(s.frac)}
        if m and m[i]:
            a["spin"] = 1 if m[i] > 0 else -1
        atoms.append(a)
    src = st.source or {}
    e: dict = {"comment": src.get("comment") or spec.label, "lattice": {"vectors_bohr": [list(r) for r in st.lattice_bohr]},
               "atoms": atoms}
    prov = src.get("import") or src.get("provenance")
    if prov:
        e["provenance"] = prov
    return e


def export_text(spec: RunSpec, fmt: str) -> str:
    """The crystal as a CIF / POSCAR / extended xyz file's text, written by the library."""
    return ask("export", entry=entry_of(spec), format=fmt, name=spec.label)["text"]


def save(name: str, spec: RunSpec, replace: bool = False) -> str:
    """Save to my materials: the user's library (never the shipped file); returns its path.  A shipped name, or an existing
    one without `replace`, is refused by the library."""
    return ask("save", into={"layer": "user", "path": user_library()}, name=name, entry=entry_of(spec), replace=replace)["path"]


def parse_spins(text: str, n_sites: int) -> Optional[tuple[float, ...]]:
    """The Structure tab's site-spin line: one of + - 0 (or +1 -1 0) per site, in site order; all zero (or empty) = none."""
    toks = text.replace(",", " ").split()
    if not toks:
        return None
    if len(toks) != n_sites:
        raise ValueError(f"{len(toks)} spins for {n_sites} sites: give one per site (+ - 0)")
    val = {"+": 1.0, "+1": 1.0, "1": 1.0, "-": -1.0, "-1": -1.0, "0": 0.0}
    bad = [t for t in toks if t not in val]
    if bad:
        raise ValueError(f"site spins are + - 0 (or +1 -1 0), not {', '.join(bad)}")
    m = tuple(val[t] for t in toks)
    return m if any(m) else None


def spins_text(spec: RunSpec) -> str:
    m = spec.physics.spin.site_moments if spec.physics.spin.polarized else None
    return " ".join("+" if x > 0 else "-" if x < 0 else "0" for x in m) if m else ""
