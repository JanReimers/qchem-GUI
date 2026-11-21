"""
qviz.basismenu -- the Basis menu a structure offers, as the LIBRARY answers it (qchem::OrbitalBasisOffers, bound as
qchem_py.basis_offers; the user, 2026-11-21: "for a single atom structure the user should automatically see only atom friendly
orbital basis sets in the drop down menu").  The GUI keeps no list of its own for an atom or a molecule; it SORTS and shows.

Each offer: {route, key, name, admitted, reason, blockGroup, note, optIn}.  An offer that is not admitted is shown greyed with the
library's reason; an opt-in offer (the one-atom molecule beside the radial atom, user ruling Q1) is shown greyed but can be picked.
A crystal still reads the GUI's periodic list (solid2_deck.periodic_bases): which files are pseudopotential-valence is not yet a
library fact the menu can ask.
"""
from __future__ import annotations

import json
from typing import Optional

from .spec import RunSpec


def offers(spec: RunSpec, qc) -> Optional[list[dict]]:
    """The library's offers for `spec`'s structure, None when it cannot be asked (no qchem_py, a crystal, or an older build)."""
    st = spec.structure
    if qc is None or not hasattr(qc, "basis_offers") or st.kind == "crystal":
        return None
    req = {"numbers": [s.Z for s in st.sites], "positions": [x for s in st.sites for x in s.xyz]}
    if st.kind == "molecule":
        req["route"] = "molecule"                     # one site stated as a MOLECULE (the opt-in route): the Gaussian menu
    got = json.loads(qc.basis_offers(json.dumps(req)))
    if "error" in got:
        raise RuntimeError(got["error"])
    return got["offers"]


def ordered(menu: list[dict]) -> list[dict]:
    """The GUI's order (the library's is unsorted): admitted first, then opt-in routes, then the greyed ones; stable otherwise."""
    return sorted(menu, key=lambda o: (not o["admitted"], o["optIn"]))
