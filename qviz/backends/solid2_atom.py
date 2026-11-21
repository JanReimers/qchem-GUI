"""
qviz.backends.solid2_atom -- a spec of kind "atom" as the RADIAL atom facade's options (qchem AtomSetup / AtomCalculation).

The library reads the options in the deck's own spelling (schema 2, `deck::FromJson(json, RunOptions&)`): the shared top-level
keys an atom consumes (spin, model, accuracy, accelerator, seed, xc) and its `atom` block (type, basisPool, N / emin / emax,
exponents, configuration).  A key the atom does not take is the LIBRARY's to refuse, in its own words, never the GUI's to
drop: so the per-code deck block (accuracy.representation.solid2.deck) is merged in as stated.

The radial atom's orbital basis is its `atom.type`: the spec's shared representation.basis names it ("Slater", "Gaussian",
"BSpline6", ...), the GUI's basis menu offers them for a one-site structure (the user, 2026-11-21: "for a single atom structure the
user should automatically see only atom friendly orbital basis sets").  RADIAL_TYPES is a STUB until the library answers which
bases a structure admits (lane G's design note laneG-basis-menu-design.md, `OrbitalBasisOffers`).
"""
from __future__ import annotations

import copy
import json
from typing import Optional

from ..spec import RunSpec
from .base import code_block, shared_representation
from .solid2_deck import LIB_EPS, _merge

#: the radial atom's basis families (deck atom.type, RunOptions::Atom::type), the menu's order.  STUB: the library's query
#: replaces it.  The four-component RKB types are not offered: the GUI has no relativity knob yet.
RADIAL_TYPES = ("Slater", "Gaussian", "BSpline6", "BSpliner6")
#: the preset exponent pools (atom.basisPool); the library's default is Medium
POOLS = ("Low", "Medium", "High")
DEFAULT_TYPE = "Slater"           # RunOptions::Atom::type's default

#: the spec's (theory, functional) -> the library's model name (Hamiltonian::Name), as the molecule's
MODELS = {("HF", None): "HF", ("DFT", "LDA"): "LDA", ("DFT", "PADE"): "PADE", ("DFT", "Xalpha"): "Xalpha"}


def radial_type(spec: RunSpec) -> str:
    """The atom's basis family: the shared representation's basis, else the library's default."""
    return shared_representation(spec).get("basis") or DEFAULT_TYPE


def why_not(spec: RunSpec) -> Optional[str]:
    """Why the radial atom cannot run `spec` as stated (None: the library decides the rest)."""
    ph, sc = spec.physics, spec.accuracy.scf or {}
    if (ph.theory, ph.functional if ph.theory == "DFT" else None) not in MODELS:
        return "the radial atom's models are HF, LDA, PADE and Xalpha"
    if radial_type(spec) not in RADIAL_TYPES:
        return f"{radial_type(spec)!r} is not a radial atom basis ({', '.join(RADIAL_TYPES)})"
    if ph.core != "all-electron":
        return "the GUI's radial atom is all-electron (the atom block's pseudopotential is not wired yet)"
    if ph.hubbard:
        return "no DFT+U on the radial atom"
    if ph.occupations.smearing != "none":
        return "the radial atom fills by its configuration: no smearing"
    if sc.get("impose_symmetry") is False:
        return "the radial atom is spherical by construction: it cannot run unimposed"
    if set(code_block(spec, "solid2")) - {"deck", "threads"}:
        return "the radial atom takes only a deck block (and threads) under representation['solid2']"
    return None


def to_options(spec: RunSpec) -> dict:
    """The atom's options, in the deck's schema-2 spelling (without `state`: the start says where it continues from)."""
    ph, ac = spec.physics, spec.accuracy
    sc = ac.scf or {}
    o: dict = {"schema": 2, "spin": "Polarized" if ph.spin.polarized else "UnPolarized",
               "model": MODELS[(ph.theory, ph.functional if ph.theory == "DFT" else None)],
               "atom": {"type": radial_type(spec)}}
    if ac.epsilon_Ha_per_atom != LIB_EPS:
        o["accuracy"] = ac.epsilon_Ha_per_atom
    if sc.get("accelerator") and sc["accelerator"] != "Auto":
        o["accelerator"] = sc["accelerator"]
    if sc.get("seed"):
        o["seed"] = sc["seed"]
    deck = copy.deepcopy(code_block(spec, "solid2").get("deck") or {})
    deck.pop("state", None)
    return _merge(o, deck)


def options_text(spec: RunSpec) -> str:
    return json.dumps(to_options(spec), sort_keys=True)


def resolved(spec: RunSpec) -> dict:
    """The settings a person reads beside the run."""
    o = to_options(spec)
    a = o.get("atom") or {}
    return {"facade": "the radial atom (spherical; radial bases, blocked by l)",
            "basis": f"{a.get('type')} radial, pool {a.get('basisPool', 'Medium (default)')}"
                     + (f", N={a['N']}" if a.get("N") else ""),
            "model": o["model"],
            "spin": ("polarized" if o["spin"] == "Polarized" else "unpolarized")
                    + ", configuration " + (a.get("configuration") or "Hund's rules (the library's)"),
            "ε": f"{o.get('accuracy', LIB_EPS):g} Ha/atom" + (" (stated)" if "accuracy" in o else " (the library's default)"),
            "accelerator": o.get("accelerator", "Auto (the facade's policy)")}
