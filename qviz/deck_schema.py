"""The qchem input deck's LAYOUT: schema 2 (doc/InputDeck.md §3, V-FACADE-PARITY P0, 2026-11-17).

Schema 2 states the structure-neutral options (spin, charge, pseudopotential, model, seed, orthogonalisation, symmetry, spins,
accelerator, label, accuracy, XC) at the TOP of the deck; only the periodic-only ones stay in `solid`.  The GUI writes schema 2
(`"schema": 2`, so the library refuses a key at its old place instead of translating it).

`migrate` moves a SAVED schema-1 deck block (a workspace or spec file's representation.solid2.deck, written before 2026-11-17) to
schema 2, with each key's schema-1 MEANING kept -- the library's translator's rule, applied once at load so nothing the GUI holds is
schema 1 afterwards.  No dependencies: spec.py reads through it.
"""
from __future__ import annotations

import copy

SCHEMA = 2

#: solid.<old> -> its schema-2 home ("xc.mesh" = the xc object's mesh); the library's kSchema1Moves (src/Calculation/Imp/Deck.C)
MOVES = (("Nelec", "Nelec"), ("charge", "charge"), ("multiplicity", "multiplicity"), ("pp", "pp"), ("species", "species"),
         ("model", "model"), ("seed", "seed"), ("ortho", "ortho"), ("orthoTol", "orthoTol"), ("imposeSymmetry", "imposeSymmetry"),
         ("siteSpins", "spins"), ("accelerator", "accelerator"), ("label", "label"), ("saveStateTo", "saveStateTo"),
         ("xcMesh", "xc.mesh"))


class DeckSchemaError(ValueError):
    """A saved deck block that states one setting at both places (which would run?)."""


def migrate(block: dict | None) -> dict | None:
    """`block` (a partial deck) in schema 2: each schema-1 `solid.<key>` at its top-level home.  A schema-1 multiplicity >= 1
    meant a POLARIZED crystal (the solid had no spin knob before R6), so it arrives beside `"spin": "Polarized"`.  A block with
    nothing at an old place comes back unchanged (the same object)."""
    if not isinstance(block, dict) or not isinstance(block.get("solid"), dict) \
            or not any(old in block["solid"] for old, _ in MOVES):
        return block
    out = copy.deepcopy(block)
    solid = out["solid"]
    for old, home in MOVES:
        if old not in solid:
            continue
        v = solid.pop(old)
        if home == "xc.mesh":
            if "mesh" in (out.get("xc") or {}):
                raise DeckSchemaError(f"deck block states both 'xc.mesh' and its schema-1 place 'solid.{old}'")
            out.setdefault("xc", {})["mesh"] = v
            continue
        if home in out:
            raise DeckSchemaError(f"deck block states both '{home}' and its schema-1 place 'solid.{old}'")
        out[home] = v
        if home == "multiplicity" and isinstance(v, int) and v >= 1:
            if out.get("spin", "Polarized") != "Polarized":
                raise DeckSchemaError(f"deck block's schema-1 'solid.multiplicity' = {v} means a polarized run, but 'spin' is {out['spin']!r}")
            out["spin"] = "Polarized"
    if not solid:
        del out["solid"]
    return out
