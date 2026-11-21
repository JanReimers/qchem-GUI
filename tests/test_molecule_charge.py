"""A charged MOLECULE: its charge is its structure's, per site (qchem's Site.netCharge; MoleculeSetup refuses a run-level
charge).  The total goes on the most electronegative site by default and the sites table's q column splits it; solid2 passes
the per-site charges to qchem_py (net_charges), and the saved state reopens with them.  CP2K refuses a charged structure until
it writes &DFT CHARGE.  Controller ruling 2026-11-21, option (a).

    QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest tests.test_molecule_charge
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
import dataclasses
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import molecules                                        # noqa: E402
from qviz.backends import FakeBackend                             # noqa: E402
from qviz.backends.base import BackendRegistry                    # noqa: E402
from qviz.backends.cp2k import Cp2kBackend                        # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402
from qviz.setup_panel import SetupPanel                          # noqa: E402
from qviz.spec import RunSpec, SpecError, Site, Structure, molecule   # noqa: E402
from tests.test_solid2_deck import solid2                         # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def water(charge=0, site_charges=None, doublet=False):
    d = molecules.preset_spec("Water (H2O)").to_dict()
    d["structure"]["charge"] = charge
    d["structure"]["site_charges"] = site_charges
    if doublet:
        d["physics"]["spin"].update(polarized=True, multiplicity=2)
    return RunSpec.from_dict(d).validate()


def hydroxide():   # OH-: O-H 1.8330 bohr, as qchem's M_DFT hydroxide
    return molecule("OH-", [8, 1], [0, 0, 0, 0, 0, 1.8330], charge=-1)


class TheDefaultRule(unittest.TestCase):
    def test_the_charge_goes_on_the_most_electronegative_site(self):
        self.assertEqual(water(1).structure.net_charges(), tuple(1 if s.element == "O" else 0 for s in water().structure.sites))
        self.assertEqual(hydroxide().structure.net_charges(), (-1, 0))
        self.assertEqual(water().structure.net_charges(), (0, 0, 0))
        self.assertEqual(water(1).num_electrons, 9)

    def test_a_stated_split_is_kept_and_must_sum_to_the_charge(self):
        o = [s.element for s in water().structure.sites].index("O")
        split = [1 if i != o else -1 for i in range(3)]               # H+ H+ O-: total +1
        self.assertEqual(water(1, split).structure.net_charges(), tuple(split))
        with self.assertRaisesRegex(SpecError, "summing to the charge"):
            water(0, split)
        with self.assertRaisesRegex(SpecError, "one per site"):
            water(1, [1])

    def test_it_round_trips_and_a_crystal_charge_is_still_refused(self):
        s = water(1)
        self.assertEqual(RunSpec.from_dict(s.to_dict()), s)
        cry = Structure("crystal", (Site("Na", frac=(0, 0, 0)),), lattice_bohr=((5, 0, 0), (0, 5, 0), (0, 0, 5)), charge=1)
        with self.assertRaisesRegex(SpecError, "crystal's charge"):
            RunSpec("Na+", cry).validate()


class TheBackends(unittest.TestCase):
    def test_solid2_passes_the_per_site_charges_only_when_charged(self):
        be = solid2()
        self.assertEqual(be.translate(hydroxide()).native["net_charges"], [-1.0, 0.0])
        neutral = be.translate(water()).native
        self.assertNotIn("net_charges", neutral, "a neutral run keeps its state file's name")

    def test_cp2k_refuses_a_charged_structure(self):
        reasons = [c.reason for c in Cp2kBackend().capabilities().features if not c.check(hydroxide())]
        self.assertTrue(any("charged" in str(r) for r in reasons), reasons)


class TheSetupPanel(unittest.TestCase):
    def setUp(self):
        self.draft = SpecDraft()
        self.panel = SetupPanel(self.draft, BackendRegistry([FakeBackend()]))
        self.draft.set(water(), "user", "water")

    def test_the_charge_box_makes_a_molecular_ion(self):
        self.assertTrue(self.panel.charge_spin.isVisibleTo(self.panel))
        self.panel.charge_spin.setValue(1)
        s = self.draft.spec
        self.assertEqual((s.structure.charge, s.structure.site_charges), (1, None))
        self.assertTrue(s.label.endswith("+"), s.label)
        q = self.panel._q_column
        o = [x.element for x in s.structure.sites].index("O")
        self.assertEqual(self.panel.sites.item(o, q).text(), "+1")

    def test_the_q_column_splits_the_charge_and_the_total_follows(self):
        self.panel.charge_spin.setValue(1)
        q = self.panel._q_column
        h = [x.element for x in self.draft.spec.structure.sites].index("H")
        self.panel.sites.item(h, q).setText("+1")                       # a user edit: the O keeps its +1
        s = self.draft.spec
        self.assertEqual(s.structure.charge, 2)
        self.assertEqual(sum(s.structure.site_charges), 2)
        self.assertEqual(self.panel.charge_spin.value(), 2)

    def test_only_the_q_column_is_editable(self):
        from PySide6 import QtCore
        it = self.panel.sites.item(0, 2)
        self.assertFalse(it.flags() & QtCore.Qt.ItemIsEditable)
        self.assertTrue(self.panel.sites.item(0, self.panel._q_column).flags() & QtCore.Qt.ItemIsEditable)


QC = solid2()._qc


@unittest.skipUnless(QC is not None and "net_charges" in (QC.Calculator.__init__.__doc__ or ""),
                     "qchem_py with net_charges not built (QCHEM_BUILD)")
class BuildRunReopen(unittest.TestCase):
    def test_a_cation_builds_runs_and_reopens(self):
        be = solid2()
        s = water(1, doublet=True)                                       # H2O+, HF, the doublet
        job = be.translate(s)
        be.build(job)
        r = be.run(job)
        self.assertTrue(r.converged)
        neutral = be.run(be.translate(water()))
        self.assertTrue(neutral.converged)
        ip = r.energy_Ha - neutral.energy_Ha
        self.assertGreater(ip, 0.3, "H2O+ lies above H2O by its ionization energy (HF Koopmans-ish ~0.4 Ha)")
        self.assertLess(ip, 0.6)
        view = dataclasses.replace(r, fields={})                     # a reopened run: fields from the saved state
        self.assertIsNone(be.fields_from_state(s, view))
        rho = view.fields["density"].sample(16, 6.0)
        self.assertGreater(float(rho.values.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
