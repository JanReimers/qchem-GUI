"""A lone atom is the RADIAL atom facade (qchem AtomCalculation, bound as qchem_py.Calculator.atom): the spec's third structure
kind, its options in the deck's spelling, Build -> Run, save / continue, its fields, and the Setup panel's atom-only basis menu
(the user, 2026-11-21: "for a single atom structure the user should automatically see only atom friendly orbital basis sets").

    QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest tests.test_atom
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
import dataclasses
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import basismenu, molecules                                        # noqa: E402
from qviz.agent.tools import Toolset                              # noqa: E402
from qviz.backends import FakeBackend, solid2_atom as A          # noqa: E402
from qviz.backends import solid2_deck                             # noqa: E402
from qviz.backends.base import BackendRegistry                    # noqa: E402
from qviz.backends.cp2k import Cp2kBackend                        # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402
from qviz.setup_panel import SetupPanel                          # noqa: E402
from qviz.spec import RunSpec, SpecError, Site, Structure, atom  # noqa: E402
from tests.test_solid2_deck import solid2                         # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class TheSpec(unittest.TestCase):
    def test_an_atom_is_one_site_and_round_trips(self):
        s = atom("Li")
        self.assertEqual((s.structure.kind, s.label, s.physics.spin.polarized), ("atom", "Li", True))
        self.assertEqual(RunSpec.from_dict(s.to_dict()), s)

    def test_two_sites_or_a_multiplicity_are_refused(self):
        two = Structure("atom", (Site("He", xyz=(0, 0, 0)), Site("He", xyz=(0, 0, 3))))
        with self.assertRaisesRegex(SpecError, "exactly one site"):
            RunSpec("He2", two).validate()
        d = atom("Li").to_dict(); d["physics"]["spin"]["multiplicity"] = 2
        with self.assertRaisesRegex(SpecError, "configuration"):
            RunSpec.from_dict(d).validate()


class TheCharge(unittest.TestCase):
    def test_an_ion_is_an_atom_with_a_charge(self):
        s = atom("Li", charge=1)
        self.assertEqual((s.label, s.structure.charge, s.num_electrons), ("Li+", 1, 2))
        self.assertEqual(atom("O", charge=-2).label, "O-2")
        self.assertEqual(RunSpec.from_dict(s.to_dict()), s)

    def test_no_electron_left_or_a_charged_molecule_is_refused(self):
        with self.assertRaisesRegex(SpecError, "no electron"):
            atom("He", charge=2)
        d = molecules.preset_spec("Water (H2O)").to_dict(); d["structure"]["charge"] = 1
        with self.assertRaisesRegex(SpecError, "not wired"):
            RunSpec.from_dict(d).validate()


class TheOptions(unittest.TestCase):
    def test_the_deck_spelling_with_the_deck_block_merged_and_no_state(self):
        d = atom("C", theory="DFT", functional="LDA").to_dict()
        d["accuracy"]["representation"] = {"basis": "Gaussian", "solid2": {"deck": {"atom": {"basisPool": "High"},
                                                                                     "state": {"restartFrom": "x.h5"}}}}
        d["accuracy"]["scf"] = {"accelerator": "GDM"}
        o = A.to_options(RunSpec.from_dict(d))
        self.assertEqual(o, {"schema": 2, "spin": "Polarized", "model": "LDA", "accelerator": "GDM",
                             "atom": {"type": "Gaussian", "basisPool": "High"}})

    def test_what_the_radial_atom_cannot_run_says_why(self):
        pp = RunSpec.from_dict(atom("O").to_dict() | {"physics": atom("O").to_dict()["physics"] | {"core": "pseudopotential"}})
        self.assertIn("all-electron", A.why_not(pp))
        dzvp = RunSpec.from_dict(atom("O").to_dict() | {"accuracy": {"representation": {"basis": "dzvp"}}})
        self.assertIn("not a radial atom basis", A.why_not(dzvp))
        self.assertIsNone(A.why_not(atom("O")))

    def test_another_code_refuses_the_radial_atom_by_name(self):
        s = atom("Ne", theory="DFT", functional="LDA")
        reasons = [c.reason for c in Cp2kBackend().capabilities().features if not c.check(s)]
        self.assertTrue(any("radial atom" in str(why) for why in reasons), reasons)


@unittest.skipUnless(solid2()._qc is not None, "qchem_py not built (QCHEM_BUILD)")
class BuildRunContinue(unittest.TestCase):
    def test_he_build_then_run_is_the_hf_limit(self):
        be = solid2()
        job = be.translate(atom("He", polarized=False))
        record = be.build(job)
        self.assertTrue(any(k.endswith(tuple("0123456789")) and " build@" in k for k in record), list(record))
        r = be.run(job)
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.energy_Ha, -2.8616799, places=6)      # the HF limit (Clementi-Roetti -2.8616800)
        self.assertEqual([L["label"] for L in r.levels][:2], ["s", "s"])
        self.assertIsNone(r.orbital, "the radial atom draws no per-orbital 3-D field")
        self.assertIn("radial atom", be.why_no_snapshot(atom("He")))

    def test_a_cation_runs(self):
        be = solid2()
        r = be.run(be.translate(atom("Li", charge=1, polarized=False)))
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.energy_Ha, -7.2364, places=3)          # Li+ HF limit -7.23642 (Clementi-Roetti)

    def test_li_polarized_saves_and_continues_exactly(self):
        be = solid2()
        s = atom("Li")
        r = be.run(be.translate(s))
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.energy_Ha, -7.4327362, places=6)
        self.assertIn("spin_density", r.fields)
        self.assertTrue(r.state_file and os.path.exists(r.state_file))
        again = be.run(be.translate(solid2_deck.continued(s, r.state_file)))
        self.assertEqual(again.provenance["iterations"], 1)
        self.assertTrue(again.provenance["restartedFrom"]["exact"])
        self.assertAlmostEqual(again.energy_Ha, r.energy_Ha, places=9)
        view = dataclasses.replace(r, fields={})     # a reopened run: its fields come back from the saved state
        self.assertIsNone(be.fields_from_state(s, view))
        rho = view.fields["density"].sample(16, 4.0)
        self.assertGreater(float(rho.values.max()), 0.0)


class TheSetupPanel(unittest.TestCase):
    def setUp(self):
        self.draft = SpecDraft()
        self.panel = SetupPanel(self.draft, BackendRegistry([FakeBackend()]))

    def test_the_picker_offers_atoms_and_their_menu_is_radial_only(self):
        i = self.panel.structure_box.findData("atom:Li")
        self.assertGreater(i, 0)
        self.panel.structure_box.setCurrentIndex(i); self.panel.structure_box.activated.emit(i)
        self.assertEqual(self.draft.spec.structure.kind, "atom")
        box = self.panel.basis_box
        self.assertEqual([box.itemText(k) for k in range(box.count())], list(A.RADIAL_TYPES))
        self.assertEqual(self.panel.structure_box.currentData(), "atom:Li")
        box.setCurrentText("Gaussian"); box.activated.emit(box.currentIndex())
        self.assertEqual(self.draft.spec.accuracy.representation["basis"], "Gaussian")
        self.assertFalse(self.panel.mult_spin.isVisibleTo(self.panel), "an atom's spin is its configuration")

    def test_the_charge_box_makes_an_ion(self):
        self.draft.set(molecules.atom_spec("Li"), "user", "Li")
        self.panel.charge_spin.setValue(1)
        self.assertEqual((self.draft.spec.label, self.draft.spec.structure.charge), ("Li+", 1))

    def test_a_one_atom_xyz_imports_as_the_radial_atom(self):
        self.panel.set_structure("neon", [10], [0.0, 0.0, 0.0])
        self.assertEqual((self.draft.spec.structure.kind, self.draft.spec.label), ("atom", "Ne"))


QC = solid2()._qc


@unittest.skipUnless(QC is not None and hasattr(QC, "basis_offers"), "qchem_py with basis_offers not built (QCHEM_BUILD)")
class TheLibrarysMenu(unittest.TestCase):
    """The Basis box is the library's answer (qchem::OrbitalBasisOffers); the GUI only orders and shows it."""
    def setUp(self):
        self.draft = SpecDraft()
        self.panel = SetupPanel(self.draft, BackendRegistry([FakeBackend()]), qc=QC)

    def test_the_fallback_list_is_the_librarys(self):
        got = basismenu.offers(atom("Ne"), QC)
        self.assertEqual([o["name"] for o in got if o["route"] == "atom"], list(A.RADIAL_TYPES))
        self.assertEqual([o for o in got if o["optIn"]][0]["route"], "molecule")

    def test_an_atom_shows_the_rkb_note_and_the_opt_in_molecule_leads_to_the_gaussian_menu(self):
        self.draft.set(molecules.atom_spec("Ne"), "user", "Ne")
        box = self.panel.basis_box
        texts = [box.itemText(k) for k in range(box.count())]
        self.assertIn("Slater_RKB  — four-component (RKB): relativistic", texts)
        k = next(k for k in range(box.count()) if box.itemData(k)["offer"]["optIn"])
        self.assertEqual(k, box.count() - 1, "the opt-in route sorts after the admitted families")
        box.setCurrentIndex(k); box.activated.emit(k)
        self.assertEqual(self.draft.spec.structure.kind, "molecule")
        offers = {box.itemData(i)["offer"]["name"]: i for i in range(box.count()) if isinstance(box.itemData(i), dict)}
        self.assertIn("dzvp", offers)
        vb = box.model().item(offers["valence_lowq_vb"])                 # F Na Al Mn O: no Ne block
        self.assertFalse(vb.isEnabled())
        self.assertIn("has no block for Ne", vb.toolTip())


class TheAgent(unittest.TestCase):
    def test_build_structure_makes_a_radial_atom(self):
        draft = SpecDraft()
        got = Toolset(draft, BackendRegistry([FakeBackend()])).call("build_structure", {"kind": "atom", "name": "fe"})
        self.assertTrue(got.get("applied"), got)
        self.assertEqual((draft.spec.structure.kind, draft.spec.label), ("atom", "Fe"))


if __name__ == "__main__":
    unittest.main()
