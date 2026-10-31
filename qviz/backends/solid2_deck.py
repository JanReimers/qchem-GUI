"""
qviz.backends.solid2_deck -- a crystal RunSpec as a qchem INPUT DECK (doc/RunSpecPlan.md §9, route (b): the deck is the seam).

The GUI's RunSpec never reaches the library.  This module is the ONE translator: RunSpec -> deck JSON (qchem
src/Calculation/Deck.C, schema 1), which `rundeck` runs today and the bridge will hand to `deck::Build(json, Observer)`
once lane B lands it.  So every GUI crystal run is a deck `rundeck` reproduces, and there is no second copy of the
deck's resolve logic in pybind.

What the deck is told, and what it is left to derive:
  * structure -- the NAME of a materials.json entry when `structure.source.materials` names one AND the spec's geometry is
    that entry's (elements, fractional sites, cell volume: a drift guard, not a re-derivation); otherwise an INLINE
    structure (PROVISIONAL shape -- lane B's inline `structure` section, OpenWork; rundeck refuses it until that lands).
  * Nelec / species -- derived by the deck from the material and the pp set, never restated.
  * spin -- unpolarized = multiplicity 0 (the deck's minimal-spin default, not written); polarized = the two-channel run,
    multiplicity |Σ site_moments| + 1, and the ordering as `siteSpins` (the moments' signs: the deck's decoration is a
    sublattice sign, the magnitude is the seed's business).
  * +U -- one manifold per site; projector ortho-atomic | pseudo-atomic | lowdin (CP2K's LOWDIN = the deck's siteShells).
  * ε -- NOT mapped yet: the deck's own defaults (the anchors' recipe) stand, and the resolved list says so.  The k-mesh
    must be stated (no ε -> k-mesh rule is banked).
  * representation["solid2"]["deck"] -- a partial deck, deep-merged LAST: the stated per-code recipe (a parity recipe's
    SCF gates, an xcMesh).  Like every per-code block it is read by this code alone.
"""
from __future__ import annotations

import copy
import functools
import json
import os
import pathlib
from typing import Optional

from ..spec import RunSpec
from .base import code_block, shared_representation

_PP_SET = {"LDA": "gth-lda-valence"}                 # functional -> the pp set generated for it (pseudopotentials.json)
_BASES = ("SIPP", "SIPP_SR", "VALENCE_LOWQ", "VALENCE_LOWQ_SR", "VALENCE_LOWQ_SR2", "VALENCE_LOWQ_SPH",
          "VALENCE_LOWQ_VA", "VALENCE_LOWQ_VB")       # BasisSetData values with a pseudopotential valence block
_PROJECTOR = {"ortho-atomic": {"orthoAtomic": True}, "pseudo-atomic": {"projector": "pseudoAtomic"},
              "lowdin": {"projector": "siteShells"}}
_SOLID2_KEYS = {"deck"}                               # representation["solid2"] keys a crystal reads
_VOLUME = {"CubicP": lambda l: l["a"] ** 3, "CubicF": lambda l: l["a"] ** 3 / 4, "TetragonalP": lambda l: l["a"] ** 2 * l["c"]}


def qchem_src() -> pathlib.Path:
    return pathlib.Path(os.environ.get("QCHEM_SRC", pathlib.Path.home() / "Code/qchem-main"))


@functools.lru_cache(maxsize=None)
def materials() -> dict:
    return json.loads((qchem_src() / "src/Structure/Data/materials.json").read_text())


def _det3(m) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _material_mismatch(spec: RunSpec, name: str) -> Optional[str]:
    """None when materials.json[name] is the spec's geometry, else what differs."""
    e = materials().get(name)
    if e is None or name.startswith("_"):
        return f"no materials.json entry {name!r}"
    lat = e["lattice"]
    if lat["type"] not in _VOLUME:
        return f"lattice type {lat['type']} not checked"
    v = _VOLUME[lat["type"]](lat) * abs(_det3(lat["T"])) if "T" in lat else _VOLUME[lat["type"]](lat)
    ours = abs(_det3(spec.structure.lattice_bohr))
    if abs(v - ours) > 1e-6 * v:
        return f"cell volume {ours:.6f} vs {name}'s {v:.6f} bohr³"
    theirs = [(a["el"], tuple(a["frac"])) for a in e["atoms"]]
    mine = [(s.element, tuple(s.frac)) for s in spec.structure.sites]
    if len(theirs) != len(mine) or any(a != b or max(abs(x - y) for x, y in zip(f, g)) > 1e-9
                                       for (a, f), (b, g) in zip(mine, theirs)):
        return f"sites {mine} vs {name}'s {theirs}"
    return None


def why_not(spec: RunSpec) -> Optional[str]:
    """None when the spec translates to a deck, else the reason (for a capability)."""
    ph, ac, tk = spec.physics, spec.accuracy, spec.task
    if spec.structure.kind != "crystal":
        return "the deck drives periodic runs only (a molecule goes through MoleculeSetup)"
    if ph.theory != "DFT" or ph.core != "pseudopotential":
        return "the periodic facade is pseudopotential DFT"
    if ph.functional not in _PP_SET:
        return f"no pseudopotential set for {ph.functional} (sets: {', '.join(_PP_SET)})"
    if ph.occupations.smearing not in ("none", "fermi-dirac"):
        return f"smearing {ph.occupations.smearing}: the periodic facade smears Fermi-Dirac only"
    if ac.kmesh is None:
        return "state accuracy.kmesh (no ε -> k-mesh rule is banked yet)"
    rep = shared_representation(spec)
    if set(rep) - {"basis"}:
        return f"representation {sorted(set(rep) - {'basis'})} not read by the periodic facade"
    if "basis" not in rep:
        return "state representation.basis (no ε -> periodic basis rule is banked yet)"
    if rep["basis"].upper() not in _BASES:
        return f"basis {rep['basis']} has no pseudopotential valence block (periodic bases: {', '.join(b.lower() for b in _BASES)})"
    if set(code_block(spec, "solid2")) - _SOLID2_KEYS:
        return f"representation.solid2 {sorted(set(code_block(spec, 'solid2')) - _SOLID2_KEYS)} not read by a periodic run"
    for h in ph.hubbard:
        if h.form != "Dudarev" or h.J_eV:
            return "+U: Dudarev with J = 0 only"
        if h.projector not in _PROJECTOR:
            return f"+U projector {h.projector} (the deck's: {', '.join(_PROJECTOR)})"
        if h.U_eV is None:
            return "+U with U to be computed: state U_eV (0 to probe) and ask task.compute_U"
    if tk.kind not in ("energy", "compute_U"):
        return f"task {tk.kind} not mapped to a deck yet"
    if ph.spin.polarized and ph.hubbard and ph.spin.site_moments is None:
        return "a polarized +U run states its magnetic ordering (physics.spin.site_moments)"
    return None


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _sign(m: float) -> int:
    return (m > 0) - (m < 0)


def structure_entry(spec: RunSpec) -> str | dict:
    """The deck's `structure`: a materials.json name when the spec is that entry, else the inline form (PROVISIONAL)."""
    src = (spec.structure.source or {}).get("materials")
    if src and _material_mismatch(spec, src) is None:
        return src
    return {"lattice": {"vectors_bohr": [list(r) for r in spec.structure.lattice_bohr]},
            "atoms": [{"el": s.element, "frac": list(s.frac)} for s in spec.structure.sites]}


def to_deck(spec: RunSpec) -> dict:
    """The deck for `spec` (asserts why_not(spec) is None).  Only what the spec STATES or derives is written: the deck's
    defaults are the library's, and its revision file records them."""
    assert why_not(spec) is None, why_not(spec)
    ph, ac, tk = spec.physics, spec.accuracy, spec.task
    tags = {s.tag: i for i, s in enumerate(spec.structure.sites) if s.tag}
    deck: dict = {"structure": structure_entry(spec), "kmesh": list(ac.kmesh),
                  "basis": {"data": shared_representation(spec)["basis"].upper()}}
    solid: dict = {}
    if _PP_SET[ph.functional] != "gth-lda-valence":
        solid["pp"] = _PP_SET[ph.functional]
    if ph.spin.polarized:
        m = ph.spin.site_moments
        solid["multiplicity"] = int(round(abs(sum(m)))) + 1 if m else (ph.spin.multiplicity or 1)
        if m:
            solid["siteSpins"] = [_sign(x) for x in m]
    if ph.hubbard:
        solid["hubbard"] = [{"site": tags[t], "l": h.l, "U_eV": h.U_eV, **_PROJECTOR[h.projector]}
                            for h in ph.hubbard for t in h.sites]
    if solid:
        deck["solid"] = solid
    scf: dict = {}
    if ph.occupations.smearing == "fermi-dirac":
        scf["smearingkT"] = ph.occupations.kT_Ha
    if (ac.scf or {}).get("max_iter"):
        scf["NMaxIter"] = ac.scf["max_iter"]
    if scf:
        deck["scf"] = scf
    if tk.kind == "compute_U":
        cu = tk.compute_U
        lr = cu.method == "linear-response"
        deck["postSCF"] = [{"hubbardLoop": {"estimator": "linearResponse"} if lr else {}} if cu.self_consistent
                           else {"hubbardLinearResponse" if lr else "estimateHubbardU": {}}]
    return _merge(deck, code_block(spec, "solid2").get("deck") or {})


def resolved(spec: RunSpec) -> list[tuple[str, str]]:
    """The settings a person reads beside the run (the Codes panel rows)."""
    d = to_deck(spec)
    st = d["structure"]
    return [("structure", st if isinstance(st, str) else "inline (lattice + sites)"),
            ("k-mesh", "×".join(map(str, d["kmesh"]))),
            ("basis", d["basis"]["data"]),
            ("ε", "not mapped: the deck's own defaults (the anchors' recipe)")]
