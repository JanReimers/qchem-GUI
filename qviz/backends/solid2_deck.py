"""
qviz.backends.solid2_deck -- a crystal RunSpec as a qchem INPUT DECK (doc/RunSpecPlan.md §9, route (b): the deck is the seam).

The GUI's RunSpec never reaches the library.  This module is the ONE translator: RunSpec -> deck JSON (qchem
src/Calculation/Deck.C, schema 1): the file `rundeck` runs, and the JSON the bridge hands to `deck::Build`
(`qchem_py.SolidRun`, qviz.backends.solid2).  So every GUI crystal run is a deck `rundeck` reproduces, and there is no second copy of the
deck's resolve logic in pybind.

What the deck is told, and what it is left to derive:
  * structure -- the NAME of a materials.json entry when `structure.source.materials` names one AND the spec's geometry is
    that entry's (elements, fractional sites, cell volume: a drift guard, not a re-derivation); otherwise an INLINE
    structure: the materials.json entry itself (`lattice.vectors_bohr` rows a, b, c; `atoms` el + frac -- qchem's
    decks/Si_diamond_inline.json).
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


def periodic_bases(elements) -> list[str]:
    """The periodic (pseudopotential valence) bases that cover every one of `elements`, as the Basis box spells them."""
    from . import gaussian_basis
    want = {e.upper() for e in elements}
    out = []
    for b in _BASES:
        try:
            if want <= set(gaussian_basis.read_bsd(b.lower())):
                out.append(b.lower())
        except (OSError, ValueError):                 # a basis file this checkout lacks: not offered
            pass
    return out


_PROJECTOR = {"ortho-atomic": {"orthoAtomic": True}, "pseudo-atomic": {"projector": "pseudoAtomic"},
              "lowdin": {"projector": "siteShells"}}
_SOLID2_KEYS = {"deck", "threads"}                    # representation["solid2"] keys a crystal reads
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
    if set(rep) - {"basis", "spherical"}:
        return f"representation {sorted(set(rep) - {'basis', 'spherical'})} not read by the periodic facade"
    if "basis" not in rep:
        return "Orbital basis must be user selected"
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
    return None              # a polarized +U run with no stated ordering: the library refuses it itself (SolidSetup's siteSpins rule)


#: the library's accuracy target when a deck states none (pinned by test against the deck's resolved "accuracy")
LIB_EPS = 1e-5

#: the spec's convergence gates (accuracy.scf) -> the deck's SCFParams Gates
DECK_GATES = (("gate_fd", "minFD"), ("gate_drho", "minDeltaRhoPerElectron"), ("gate_de", "minDeltaE"), ("gate_dfd", "minDeltaFD"))


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _sign(m: float) -> int:
    return (m > 0) - (m < 0)


def structure_entry(spec: RunSpec) -> str | dict:
    """The deck's `structure`: a materials.json name when the spec is that entry, else the inline entry."""
    src = (spec.structure.source or {}).get("materials")
    shipped = (spec.structure.source or {}).get("layer", "shipped") == "shipped"   # a deck names SHIPPED cells only (yours may shadow one)
    if src and shipped and _material_mismatch(spec, src) is None:
        return src
    # each atom's `spin` rides the entry as it does in materials.json: the library's SEED reads its magnetic decoration from the
    # structure's atoms (solid.siteSpins drives the imposition); without it an inline AFM cell seeded NON-magnetic (the user's NiO,
    # 2026-11-12: m_site 0.000000 at iteration 1 vs 2.48 for the same cell by name)
    moments = spec.physics.spin.site_moments or ()
    entry: dict = {"lattice": {"vectors_bohr": [list(r) for r in spec.structure.lattice_bohr]},
                   "atoms": [{"el": s.element, "frac": list(s.frac), **({"spin": _sign(moments[i])} if i < len(moments) and moments[i] else {})}
                             for i, s in enumerate(spec.structure.sites)]}
    prov = (spec.structure.source or {}).get("import") or (spec.structure.source or {}).get("provenance")
    if prov:                              # an imported (or saved-import) cell: the deck keeps where it came from (Inline takes it)
        entry["provenance"] = prov
    return entry


def to_deck(spec: RunSpec) -> dict:
    """The deck for `spec` (asserts why_not(spec) is None).  Only what the spec STATES or derives is written: the deck's
    defaults are the library's, and its revision file records them."""
    assert why_not(spec) is None, why_not(spec)
    ph, ac, tk = spec.physics, spec.accuracy, spec.task
    tags = {s.tag: i for i, s in enumerate(spec.structure.sites) if s.tag}
    deck: dict = {"structure": structure_entry(spec), "kmesh": list(ac.kmesh),
                  "basis": {"data": shared_representation(spec)["basis"].upper()}}
    if (ac.representation or {}).get("spherical"):
        deck["basis"]["spherical"] = True                 # real solid-harmonic Gaussians (5 d, 7 f); unstated: Cartesian (6 d, 10 f)
    if ac.epsilon_Ha_per_atom != LIB_EPS:                 # the run's ONE accuracy target (V-FACADE-ACCURACY), stated off its default
        deck["accuracy"] = ac.epsilon_Ha_per_atom
    solid: dict = {}
    if _PP_SET[ph.functional] != "gth-lda-valence":
        solid["pp"] = _PP_SET[ph.functional]
    if ph.spin.polarized:
        m = ph.spin.site_moments
        solid["multiplicity"] = int(round(abs(sum(m)))) + 1 if m else (ph.spin.multiplicity or 1)
        if m:
            solid["siteSpins"] = [_sign(x) for x in m]
    if (ac.scf or {}).get("accelerator"):
        solid["accelerator"] = ac.scf["accelerator"]      # SolidCalcOptions::accelerator by name: Auto | DIIS | GDM | Ladder | Null
    if (ac.scf or {}).get("impose_symmetry"):
        solid["imposeSymmetry"] = True                    # the SHUBNIKOV group of the stated siteSpins (a polarized crystal), else the space group
    if (ac.scf or {}).get("seed"):
        solid["seed"] = ac.scf["seed"]                    # SolidCalcOptions::seed by name: IonicSAD (the facade's) | SAD | CoreGuess | Uniform
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
    for ours, deck_key in (("mixing_alpha", "startingRelaxRo"), ("mixing_alpha_min", "minRelaxRo"), ("mixing_alpha_max", "maxRelaxRo")):
        if (ac.scf or {}).get(ours):
            scf[deck_key] = float(ac.scf[ours])           # linear D-mixing's start / adaptive ceiling (SCFParams)
    for ours, deck_key in DECK_GATES:
        v = (ac.scf or {}).get(ours)
        if v is not None:
            scf[deck_key] = v if v == "off" else float(v)   # a Gate: a number, or "off"
    if (ac.scf or {}).get("mixer"):
        scf["mixer"] = ac.scf["mixer"]                    # SCFParams::Mixing by name: Auto | Linear | Kerker | Pulay
    for ours, deck_key, type_ in MIXER_KNOBS:             # what a stated Kerker / Pulay reads (the library refuses one without them)
        if (ac.scf or {}).get(ours):
            scf[deck_key] = type_(ac.scf[ours])
    if (ac.scf or {}).get("mom"):
        scf["useMOM"] = True                              # SCFParams::UseMOM, and its delayed reference capture
        if (ac.scf or {}).get("mom_start"):
            scf["momStartIter"] = int(ac.scf["mom_start"])
    if scf:
        deck["scf"] = scf
    sc = ac.scf or {}
    diis = {deck_key: type_(sc[ours]) for ours, deck_key, type_ in (("diis_depth", "nProj", int), ("diis_sv_tol", "svTol", float),
                                                                    ("diis_engage_fd", "engageBelowFD", float)) if sc.get(ours)}
    accel: dict = {}
    if diis:
        accel["diis"] = diis                              # SolidAcceleratorOptions.diis: DIIS's and the Ladder's DIIS rung
    if sc.get("ladder_switch_at") is not None:
        accel["ladder"] = {"switchAt": float(sc["ladder_switch_at"])}   # the Ladder's DIIS -> GDM hand-off (0: on stall only)
    if accel:
        deck["acceleratorOptions"] = accel
    if tk.kind == "compute_U":
        cu = tk.compute_U
        lr = cu.method == "linear-response"
        deck["postSCF"] = [{"hubbardLoop": {"estimator": "linearResponse"} if lr else {}} if cu.self_consistent
                           else {"hubbardLinearResponse" if lr else "estimateHubbardU": {}}]
    if code_block(spec, "solid2").get("threads"):         # the run's OpenMP threads (deck `threads`; a resource, not physics)
        deck["threads"] = {"openmp": int(code_block(spec, "solid2")["threads"])}
    return _merge(deck, code_block(spec, "solid2").get("deck") or {})


#: the spec's mixer knobs (accuracy.scf) -> the deck's SCFParams: Kerker reads kerkerG0 (> 0); Pulay reads pulayDepth (> 0), pulayStart,
#: and kerkerG0 (0 or absent: a history on D, unpreconditioned).  No GUI default: the library states none for Kerker's G0.
MIXER_KNOBS = (("kerker_g0", "kerkerG0", float), ("pulay_depth", "pulayDepth", int), ("pulay_start", "pulayStart", int))


#: The SCF seeds a user may state (accuracy.scf["seed"]): (name, menu text, a crystal may, a molecule may).  Unstated = Auto, the
#: facade's own choice (seed_auto says which).  The library's SeedStrategy names (ChargeDensity/Seed.C).
SEEDS = (("SAD", "SAD: superposed neutral-atom densities", True, True),
         ("IonicSAD", "Ionic SAD: superposed ionic densities", True, False),
         ("CoreGuess", "Core guess: ρ = 0, so the first orbitals are the bare nuclei's (hydrogen-like)", True, True),
         ("Uniform", "Uniform ρ over the cell", True, False))


def imposed_text(spec: RunSpec) -> str:
    """What an imposed crystal imposes: the magnetic (Shubnikov, black-white) space group of its site spins when it carries an
    ordering -- the AFM order is then a SYMMETRY of the run, its sublattices never averaged together -- else the space group."""
    m = spec.physics.spin.site_moments
    if spec.physics.spin.polarized and m and any(m):
        return "IMPOSED: the Shubnikov (magnetic) group of the site spins " + " ".join("+" if x > 0 else "-" if x < 0 else "0" for x in m)
    return "IMPOSED: the space group"


def seed_auto(spec: RunSpec) -> str:
    """What an unstated seed resolves to (SolidCalcOptions::seed's default; MoleculeSetup's AutoSeed)."""
    if spec.structure.kind == "crystal":
        return "Ionic SAD"
    if spec.physics.theory == "DFT" or spec.physics.core == "pseudopotential":
        return "SAD (the core guess if an element has no atomic density)"
    return "Core guess (ρ = 0)"


def mixer_auto(spec: RunSpec) -> str:
    """What the Auto mixer resolves to (SCFParams ResolveMixing at a cold start): a molecule mixes linearly; a crystal beside a
    stated DIIS / Ladder mixes linearly (two histories fight); else a crystal runs Pulay on D (theAutoMixing: α 0.45, depth 8,
    from iteration 5)."""
    sc = spec.accuracy.scf or {}
    if spec.structure.kind != "crystal":
        return "Linear"
    if sc.get("accelerator") in ("DIIS", "Ladder"):
        return "Linear"
    if sc.get("pulay_depth"):
        return "Pulay"
    if sc.get("kerker_g0"):
        return "Kerker"
    return "Pulay on D (α 0.45, depth 8, from iteration 5)"


def continues_from(spec: RunSpec) -> Optional[str]:
    """The saved state `spec` starts from (its deck's state.restartFrom), None for a fresh start."""
    return ((code_block(spec, "solid2").get("deck") or {}).get("state") or {}).get("restartFrom")


def continued(spec: RunSpec, state_file: Optional[str]) -> RunSpec:
    """`spec` starting from `state_file` (None: from scratch again).  The path rides the per-code deck block, so the run is
    still a deck `rundeck` reproduces; the library judges the file's fingerprint against the built basis and REFUSES a
    state that means nothing here (another structure, basis, k-mesh, electron count, spin group, functional) -- it continues
    an exact resume, or a warm start across +U / grids / multiplicity / kT, under whatever accelerator and gates `spec`
    states (no accelerator history is carried: a continuation is a fresh stage)."""
    d = spec.to_dict()
    ac = d.setdefault("accuracy", {})
    rep = dict(ac.get("representation") or {})
    own = copy.deepcopy(rep.get("solid2") or {})
    deck = own.setdefault("deck", {})
    state = deck.setdefault("state", {})
    if state_file:
        state["restartFrom"] = state_file
    else:
        state.pop("restartFrom", None)
        if not state: deck.pop("state")
        if not deck: own.pop("deck")
    if own: rep["solid2"] = own
    else: rep.pop("solid2", None)
    ac["representation"] = rep or None
    return RunSpec.from_dict(d).validate()


def resolved(spec: RunSpec) -> list[tuple[str, str]]:
    """The settings a person reads beside the run (the Codes panel rows)."""
    d = to_deck(spec)
    st = d["structure"]
    return [("structure", st if isinstance(st, str) else "inline (lattice + sites)"),
            ("k-mesh", "×".join(map(str, d["kmesh"]))),
            ("basis", d["basis"]["data"]),
            ("Gaussians", "spherical" if d["basis"].get("spherical") else "Cartesian"),
            ("ε", f"{d['accuracy']:g} Ha/atom, stated: it sizes the grids (rasters, mesh)" if "accuracy" in d
                  else f"{LIB_EPS:g} Ha/atom, the library's default: it sizes the grids (rasters, mesh)"),
            ("symmetry", imposed_text(spec) if d.get("solid", {}).get("imposeSymmetry") else "free (not imposed)"),
            ("threads", f"{d['threads']['openmp']} (OpenMP)" if "threads" in d else "the process's (QCHEM_OPENMP_THREADS; unset = serial)"),
            ("accelerator", d.get("solid", {}).get("accelerator", "Auto (the facade's policy)")),
            ("ρ mixer", (d.get("scf") or {}).get("mixer") or f"Auto: {mixer_auto(spec)}"),
            ("starts from", continues_from(spec) or f"scratch, seeded by {d.get('solid', {}).get('seed') or 'Auto: ' + seed_auto(spec)}")]
