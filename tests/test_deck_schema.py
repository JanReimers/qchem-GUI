"""The GUI writes the qchem deck's SCHEMA 2 (doc/InputDeck.md §3, V-FACADE-PARITY P0) and moves a saved schema-1 block at load.

THE DONE-CHECK for retiring the library's schema-1 translator (Deck.C ToSchema2): every deck the GUI writes resolves under
`rundeck --resolved` to the SAME run as its schema-1 spelling read through that translator.  The schema-2 deck is stamped
`"schema": 2`, so the library refuses (never translates) a key at its old place: the translator plays no part in reading it.

    QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest tests.test_deck_schema
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
import copy
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

from qviz import deck_schema
from qviz.backends import solid2_deck as D
from qviz.spec import RunSpec, SpecError
from tests.test_solid2_deck import BENCH, NAF, RUNDECK, SI, crystal, from_bench, mno_plus_u, si_inline


def schema1(deck: dict) -> dict:
    """`deck`'s schema-1 spelling, as the GUI wrote it before 2026-11-17: the shared keys inside `solid`, no `spin` (a schema-1
    multiplicity >= 1 MEANT Polarized), no stamp."""
    d = copy.deepcopy(deck)
    d.pop("schema", None)
    solid = d.setdefault("solid", {})
    if d.get("spin") == "Polarized" and d.get("multiplicity", 0) >= 1:
        d.pop("spin")
    for old, home in deck_schema.MOVES:
        if home == "xc.mesh":
            if "mesh" in (d.get("xc") or {}):
                solid[old] = d["xc"].pop("mesh")
                if not d["xc"]:
                    del d["xc"]
        elif home in d:
            solid[old] = d.pop(home)
    assert "spin" not in d, "a schema-1 deck has no spin knob"
    return d


def resolve(deck: dict) -> dict:
    """`rundeck --resolved`: the deck as it will RUN (read, resolved against the structure files; no basis, no SCF)."""
    with tempfile.TemporaryDirectory() as t:
        p = pathlib.Path(t) / "gui.json"
        p.write_text(json.dumps(deck))
        r = subprocess.run([str(RUNDECK), str(p), "--resolved"], capture_output=True, text=True, timeout=120,
                           env=os.environ | {"QCHEM_OPENMP_THREADS": "1"})
        if r.returncode != 0:
            raise AssertionError(f"rundeck --resolved refused the deck:\n{r.stderr}{r.stdout}\n{json.dumps(deck, indent=1)}")
        return json.loads(r.stdout[r.stdout.index("{"):])


def gui_specs() -> list[RunSpec]:
    """Every option the writer states: the bench recipes, the inline cell's knobs, polarized AFM / FM / net-moment, +U, a trim."""
    mno = mno_plus_u()
    fm = RunSpec.from_dict(mno.to_dict() | {"label": "MnO FM"} | {"physics": mno.to_dict()["physics"] | {
        "spin": {"polarized": True, "site_moments": [5, 5, 0, 0]}, "hubbard": []}}).validate()
    pol_no_order = RunSpec.from_dict(si_inline().to_dict() | {"label": "Si two-channel"} | {"physics": si_inline().to_dict()["physics"] | {
        "spin": {"polarized": True}}}).validate()
    pade = RunSpec.from_dict(si_inline().to_dict() | {"physics": si_inline().to_dict()["physics"] | {"functional": "PADE"}}).validate()
    trim = RunSpec.from_dict(si_inline().to_dict() | {"accuracy": si_inline().to_dict()["accuracy"] | {"representation": {
        "basis": "sipp_sr", "solid2": {"deck": {"basis": {"autoTrim": True}, "orthoTol": 1e-6, "solid": {"densityEcut": 20.0}}}}}}).validate()
    return [from_bench("Si_Gamma_parity.json", *SI), from_bench("NaF_SR2_Gamma_parity.json", *NAF), si_inline(),
            si_inline(impose_symmetry=True), si_inline(accelerator="Ladder", diis_depth=6, diis_engage_fd=0.1, ladder_switch_at=1e-5),
            si_inline(seed="SAD"), pade, trim, pol_no_order, mno, fm,
            crystal("MnO imposed", mno.to_dict()["structure"]["lattice_bohr"],
                    [{k: v for k, v in site.items() if k != "tag"} for site in mno.to_dict()["structure"]["sites"]],
                    physics={"spin": {"polarized": True, "site_moments": [5, -5, 0, 0]}},
                    accuracy={"representation": {"basis": "valence_lowq_va"}, "scf": {"impose_symmetry": True, "accelerator": "GDM"}})]


class TheWriterIsSchema2(unittest.TestCase):
    def test_the_shared_keys_are_top_level_and_spin_is_its_own_knob(self):
        for spec in gui_specs():
            with self.subTest(spec.label):
                d = D.to_deck(spec)
                self.assertEqual(d["schema"], 2)
                self.assertFalse({old for old, _ in deck_schema.MOVES} & set(d.get("solid", {})), d.get("solid"))
                if spec.physics.spin.polarized:
                    self.assertEqual(d["spin"], "Polarized")
                    m = spec.physics.spin.site_moments
                    self.assertEqual(d["multiplicity"], int(round(abs(sum(m)))) + 1 if m else 1)
                else:
                    self.assertNotIn("spin", d)
                    self.assertNotIn("multiplicity", d)


@unittest.skipUnless(RUNDECK.exists() and (BENCH / "Si_Gamma_parity.json").exists(), "rundeck / qchem's decks not found (QCHEM_BUILD, QCHEM_SRC)")
class TheTranslatorIsNotNeeded(unittest.TestCase):
    def test_every_gui_deck_resolves_as_its_schema1_spelling_did(self):
        for spec in gui_specs():
            with self.subTest(spec.label):
                d2 = D.to_deck(spec)
                self.assertEqual(resolve(d2), resolve(schema1(d2)))

    def test_a_schema2_deck_refuses_a_key_at_its_old_place(self):
        d = D.to_deck(si_inline())
        d.setdefault("solid", {})["seed"] = "SAD"
        with self.assertRaisesRegex(AssertionError, "solid.seed.*SCHEMA-1 place of 'seed'"):
            resolve(d)


class SavedBlocksMove(unittest.TestCase):
    OLD = {"basis": {"autoTrim": True}, "solid": {"orthoTol": 1e-6, "densityEcut": 20.0, "multiplicity": 1, "siteSpins": [1, -1],
                                                  "xcMesh": {"nRadial": 40}, "seed": "SAD"}}

    def test_a_schema1_block_moves_with_its_meaning(self):
        new = deck_schema.migrate(self.OLD)
        self.assertEqual(new, {"basis": {"autoTrim": True}, "solid": {"densityEcut": 20.0}, "orthoTol": 1e-6, "multiplicity": 1,
                               "spin": "Polarized", "spins": [1, -1], "xc": {"mesh": {"nRadial": 40}}, "seed": "SAD"})
        self.assertIn("multiplicity", self.OLD["solid"], "the saved block itself is not edited")
        self.assertIs(deck_schema.migrate(new), new, "a schema-2 block comes back as it is")

    def test_a_saved_spec_loads_in_schema2(self):
        spec = si_inline()
        d = spec.to_dict()
        d["accuracy"]["representation"]["solid2"]["deck"] = {"solid": {"densityEcut": 20.0, "orthoTol": 1e-6}, "basis": {"autoTrim": True}}
        deck = RunSpec.from_dict(d).accuracy.representation["solid2"]["deck"]
        self.assertEqual(deck, {"solid": {"densityEcut": 20.0}, "orthoTol": 1e-6, "basis": {"autoTrim": True}})

    def test_a_block_stating_both_places_is_refused(self):
        d = si_inline().to_dict()
        d["accuracy"]["representation"]["solid2"]["deck"] = {"solid": {"seed": "SAD"}, "seed": "Uniform"}
        with self.assertRaisesRegex(SpecError, "both 'seed' and its schema-1 place 'solid.seed'"):
            RunSpec.from_dict(d)


if __name__ == "__main__":
    unittest.main()
