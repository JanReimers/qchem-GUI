"""The policies that acted on a run, read from its record (doc/OpenWork.md, "Every existing policy reports itself under
policy"): refusals lead, warnings follow with their cost, automatic choices are one click away; a refused run keeps its record.

    .venv/bin/python -m unittest tests.test_policy
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import molecules, policyinfo                            # noqa: E402
from qviz.backends import FakeBackend                             # noqa: E402
from qviz.backends.base import BackendRegistry, Refused, Result   # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402
from qviz.setup_panel import SetupPanel                           # noqa: E402
from qviz.workspace import Run, RunStatus                         # noqa: E402
import tempfile as _bse_tmp; from qviz import bse as _bse; _bse.BASES_DIR = _bse_tmp.mkdtemp(prefix="qchem-gui-bases-")   # never ~/.solid2/bases

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

#: the contract's shape: a build run and the SCF run, which repeats one of the build's rows
RECORD = {
    "inline build@1": {"policy": [
        {"name": "DensityEcutFloor", "decided": "density Ecut 4 Ha", "verdict": "choose", "reason": "the floor: cutoff factor 2 x the tightest exponent"},
        {"name": "BeckeOnSolid", "decided": "a Becke XC mesh on a periodic cell", "verdict": "warn", "reason": "not a production route",
         "cost": {"wallSeconds": 840, "memoryBytes": 3.2 * 2**30}}],
        "basis": {"nFunctions": 26, "policy": [
            {"name": "AutoTrim", "decided": "2 near-dependent functions dropped", "verdict": "choose", "reason": "min eig S below orthoTol"}]}},
    "inline@2": {"policy": [
        {"name": "DensityEcutFloor", "decided": "density Ecut 4 Ha", "verdict": "choose", "reason": "the floor: cutoff factor 2 x the tightest exponent"},
        {"name": "MemoryBudget", "decided": "the ERI4 cache", "verdict": "resource", "reason": "over the box's free memory",
         "cost": {"memoryBytes": 40 * 2**30}},
        {"name": "AdmitCoulomb", "decided": "a multipole Coulomb fit on a periodic cell", "verdict": "refuse",
         "reason": "site multipoles have no periodic form"}]},
}


class Reading(unittest.TestCase):
    def test_refusals_lead_then_warnings_and_the_choices_apart(self):
        loud, chosen = policyinfo.lines(RECORD)
        self.assertEqual(loud, [
            "✗ refused: a multipole Coulomb fit on a periodic cell -- site multipoles have no periodic form  [AdmitCoulomb]",
            "✗ refused on resources: the ERI4 cache -- over the box's free memory (would need 40.0 GB)  [MemoryBudget]",
            "⚠ warning: a Becke XC mesh on a periodic cell -- not a production route (estimated 14 min, 3.2 GB)  [BeckeOnSolid]"])
        self.assertEqual(chosen, ["density Ecut 4 Ha -- the floor: cutoff factor 2 x the tightest exponent  [DensityEcutFloor]",
                                  "2 near-dependent functions dropped -- min eig S below orthoTol  [AutoTrim, in basis]"],
                         "a row the build and the SCF both file is listed once; a nested row says where it fired")

    def test_no_policy_rows_is_nothing_to_say(self):
        self.assertEqual(policyinfo.lines({"m@1": {"basis": {}}}), ([], []))
        self.assertEqual(policyinfo.lines(None), ([], []))


class SetupShowsThem(unittest.TestCase):
    def test_a_run_s_refusal_is_shown_and_its_choices_are_one_click_away(self):
        spec = molecules.preset_spec("Water (H2O)")
        run = Run(spec, FakeBackend())
        run.result = Result(converged=False, energy_Ha=None, structure=None, record=RECORD)
        panel = SetupPanel(SpecDraft(spec), BackendRegistry([FakeBackend()]))
        self.assertTrue(panel.policy_loud.isHidden(), "the draft has no record: nothing to say")
        panel.show_run(run)
        self.assertFalse(panel.policy_loud.isHidden())
        self.assertIn("✗ refused: a multipole Coulomb fit", panel.policy_loud.text())
        self.assertIn("2 automatic choices by policy", panel.policy_more.text())
        self.assertTrue(panel.policy_chosen.isHidden(), "automatic choices are collapsed")
        panel.policy_more.setChecked(True)
        self.assertFalse(panel.policy_chosen.isHidden())
        self.assertIn("density Ecut 4 Ha", panel.policy_chosen.text())
        panel.show_run(None)
        self.assertTrue(panel.policy_loud.isHidden() and panel.policy_more.isHidden())


class RefusedKeepsItsRecord(unittest.TestCase):
    def test_a_refusal_settles_failed_with_the_record_it_filed(self):
        class Refusing(FakeBackend):
            def run(self, job, on_step=None):
                raise Refused("AdmitCoulomb: a multipole Coulomb fit on a periodic cell", RECORD)
        run = Run(molecules.preset_spec("Water (H2O)"), Refusing())
        run.compute()
        self.assertIs(run.status, RunStatus.FAILED)
        self.assertEqual(run.error, "AdmitCoulomb: a multipole Coulomb fit on a periodic cell")
        self.assertIs(run.result.record, RECORD)
        self.assertFalse(run.result.converged)


if __name__ == "__main__":
    unittest.main()
