"""
qviz.basismenu -- the Basis menu a structure offers, as the LIBRARY answers it (qchem::OrbitalBasisOffers, bound as
qchem_py.basis_offers; the user, 2026-11-21: "for a single atom structure the user should automatically see only atom friendly
orbital basis sets in the drop down menu").  The GUI keeps no list of its own for an atom or a molecule; it SORTS and shows.

Each offer: {route, key, name, admitted, reason, blockGroup, note, optIn, kind, ppSet}.  An offer that is not admitted is shown
greyed with the library's reason; an opt-in offer (the one-atom molecule beside the radial atom, user ruling Q1) is shown greyed but
can be picked.  A crystal's menu is the library's too: each basis file states its KIND (all-electron / pp-valence and the
pseudopotential set it pairs with), and the library greys an all-electron file on a crystal with the reason.  The GUI keeps no
basis list of its own.
"""
from __future__ import annotations

import functools
import json
from typing import Optional

from .spec import RunSpec

#: the crystal deck the GUI writes names a Gaussian file (basis.data); an offer the deck writer cannot say yet is shown greyed with this
_NOT_WRITTEN = "the GUI's crystal deck writes a Gaussian basis file (basis.data) only, not yet {name}"


def offers(spec: RunSpec, qc) -> Optional[list[dict]]:
    """The library's offers for `spec`'s structure, None when it cannot be asked (no qchem_py, or an older build)."""
    st = spec.structure
    if qc is None or not hasattr(qc, "basis_offers"):
        return None
    if st.kind == "crystal":
        from .backends import solid2_deck
        e = solid2_deck.structure_entry(spec)
        got = _ask(qc, json.dumps({"cell": solid2_deck.materials()[e] if isinstance(e, str) else e}, sort_keys=True))
        return [o if o["key"].startswith("basis.data=") else
                o | {"admitted": False, "reason": _NOT_WRITTEN.format(name=o["name"])} for o in got]
    req = {"numbers": [s.Z for s in st.sites], "positions": [x for s in st.sites for x in s.xyz]}
    if st.kind == "molecule":
        req["route"] = "molecule"                     # one site stated as a MOLECULE (the opt-in route): the Gaussian menu
    return _ask(qc, json.dumps(req, sort_keys=True))


@functools.lru_cache(maxsize=64)
def _ask(qc, req: str) -> list[dict]:
    """qchem_py.basis_offers for one request, cached: the answer is a pure function of the structure and the shipped files, and a
    capability check asks it on every refresh."""
    got = json.loads(qc.basis_offers(req))
    if "error" in got:
        raise RuntimeError(got["error"])
    return got["offers"]


def crystal_offer(spec: RunSpec, qc, basis: str) -> Optional[dict]:
    """The library's crystal offer for the file named `basis` (any case), None when the menu lists no such file."""
    return next((o for o in offers(spec, qc) or [] if o["key"] == "basis.data=" + basis.lower()), None)


def ordered(menu: list[dict]) -> list[dict]:
    """The GUI's order (the library's is unsorted): admitted first, then opt-in routes, then the greyed ones; stable otherwise."""
    return sorted(menu, key=lambda o: (not o["admitted"], o["optIn"]))
