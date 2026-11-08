"""Save these grids into my deck: the raster pilot's rasterPilot.restate goes into the draft's deck (undoable), and a run of a
deck that restates an earlier run's grids says whether it built the same ones -- by the grids report, never the energy.

    .venv/bin/python -m unittest tests.test_restate
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
import copy
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import gridinfo                                         # noqa: E402
from qviz.backends import FakeBackend                             # noqa: E402
from qviz.backends.base import BackendRegistry, Result            # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402
from qviz.setup_panel import SetupPanel                           # noqa: E402
from qviz.spec import RunSpec                                     # noqa: E402
from qviz.workspace import Run                                    # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

KEYS = {"accuracy": 0, "solid": {"cutoffFactor": 2.0, "tolerances": {"rasterX": 0.0, "vlocRasterX": 0.0, "xcRasterX": 0.0,
                                                                      "vlocEps": 1.0000000000000002e-06}}}
GRIDS = {"densityEcut": 20.0, "cutoffFactor": 2.0, "vlocEps": {"value": 1.0000000000000002e-06, "source": "derived",
                                                               "rigorous": 1e-05, "etaRouting": 10.0}}
PILOTED = {"inline build@1": {"grids": GRIDS, "hamiltonian": {"xcQuadrature": {"kind": "Uniform", "N": [17, 17, 17]}}},
           "inline@2": {"rasterPilot": {"settled": True, "restate": KEYS}}}
RESTATED = {"inline build@3": {"grids": {**GRIDS, "vlocEps": {"value": 1.0000000000000002e-06, "source": "stated"}},
                               "hamiltonian": {"xcQuadrature": {"kind": "Uniform", "N": [17, 17, 17]}}},
            "inline@4": {}}


def crystal(deck=None) -> RunSpec:
    return RunSpec.from_dict({"label": "Si", "structure": {"kind": "crystal", "lattice_bohr": [[0, 5.13, 5.13], [5.13, 0, 5.13], [5.13, 5.13, 0]],
                                                           "sites": [{"element": "Si", "frac": [0, 0, 0]}, {"element": "Si", "frac": [0.25, 0.25, 0.25]}]},
                              "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential", "spin": {"polarized": False}},
                              "accuracy": {"epsilon_Ha_per_atom": 1e-5,
                                           "representation": {"basis": "valence_lowq_sr", "solid2": {"deck": deck or {"solid": {"densityEcut": 20.0}}}}}})


class Reading(unittest.TestCase):
    def test_the_restate_and_whether_it_settled(self):
        self.assertEqual(gridinfo.restate(PILOTED), (KEYS, True))
        self.assertEqual(gridinfo.restate(RESTATED), (None, False), "a restated run pilots nothing")

    def test_a_deck_restates_the_keys_it_states_with_their_values(self):
        deck = {"solid": {"densityEcut": 20.0, **copy.deepcopy(KEYS["solid"])}, "accuracy": 0}
        self.assertTrue(gridinfo.restated_by(deck, KEYS))
        deck["solid"]["tolerances"]["vlocEps"] = 1e-6                           # not the full-precision value
        self.assertFalse(gridinfo.restated_by(deck, KEYS))

    def test_densityecut_reads_stated_or_floor_and_a_saved_number_compares_by_value(self):
        new = copy.deepcopy(RESTATED); new["inline build@3"]["grids"]["densityEcut"] = {"value": 20.0, "source": "stated"}
        self.assertIn("density Ecut: 20 Ha (stated)", gridinfo.built(new)[0])
        self.assertIn("density Ecut floor: 20 Ha (cutoff factor 2)", gridinfo.built(PILOTED)[0], "a saved run's bare number is the floor")
        self.assertEqual(gridinfo.grid_differences(PILOTED, new), [], "20 (a saved number) and {value 20, stated}: the same grid")

    def test_the_grids_that_ran_are_the_scf_run_s_after_the_pilot_raised_them(self):
        raised = copy.deepcopy(PILOTED); raised["inline@2"]["grids"] = {**GRIDS, "cutoffFactor": 4.0, "densityEcut": 8.0}
        self.assertEqual(gridinfo._section(raised, "grids")["cutoffFactor"], 4.0, "the SCF run's grids, not the build's")
        self.assertEqual(gridinfo.grid_differences(raised, RESTATED), ["grids.cutoffFactor", "grids.densityEcut"])

    def test_the_rasters_say_who_set_them_and_a_restated_deck_builds_the_same_ones(self):
        piloted = copy.deepcopy(PILOTED)
        piloted["inline build@1"]["grids"] = {**GRIDS, "raster": {"x": 14.5, "vlocX": 0.0, "xcX": 29.0}, "rasterSource": "pilot"}
        self.assertIn("rasters: x(density, V_long, XC) = (14.5, ball, 29) (sized by the pilot)", gridinfo.built(piloted)[0])
        restated = copy.deepcopy(RESTATED)
        restated["inline build@3"]["grids"] = {**restated["inline build@3"]["grids"], "raster": {"x": 14.5, "vlocX": 0.0, "xcX": 29.0},
                                               "rasterSource": "stated"}
        self.assertEqual(gridinfo.grid_differences(piloted, restated), [], "who set the rasters is not a grid")
        ball = copy.deepcopy(RESTATED); ball["inline build@3"]["grids"]["raster"] = {"x": 0, "vlocX": 0, "xcX": 0}
        self.assertIn("rasters: the ball's own", gridinfo.built(ball)[0], "a run saved before rasterSource: no source")

    def test_a_stated_value_is_the_grid_it_restates_and_a_changed_grid_is_named(self):
        self.assertEqual(gridinfo.grid_differences(PILOTED, RESTATED), [], "stated vs derived: the same value, the same grid")
        moved = copy.deepcopy(RESTATED); moved["inline build@3"]["hamiltonian"]["xcQuadrature"]["N"] = [18, 18, 18]
        self.assertEqual(gridinfo.grid_differences(PILOTED, moved), ["hamiltonian.xcQuadrature.N"])


class TheButton(unittest.TestCase):
    def test_a_run_s_grids_go_into_the_draft_s_deck_and_undo_takes_them_out(self):
        source = Run(crystal(), FakeBackend())
        source.result = Result(converged=True, energy_Ha=-7.1, structure=None, record=PILOTED)
        runs = []
        draft = SpecDraft(crystal())
        panel = SetupPanel(draft, BackendRegistry([FakeBackend()]), runs=lambda: runs)
        self.assertTrue(panel.grids_save.isHidden(), "no record with a pilot yet")
        runs.append(source)
        panel.show_run(source)
        self.assertFalse(panel.grids_save.isHidden())
        panel.grids_save.click()
        self.assertIsNone(panel.viewing_run, "the run's spec became the draft")
        deck = draft.spec.accuracy.representation["solid2"]["deck"]
        self.assertTrue(gridinfo.restated_by(deck, KEYS))
        self.assertEqual(deck["solid"]["densityEcut"], 20.0, "the deck's own settings stay")

        rerun = Run(draft.spec, FakeBackend())
        rerun.result = Result(converged=True, energy_Ha=-7.1 + 6e-8, structure=None, record=RESTATED)
        runs.append(rerun)
        panel.show_run(rerun)
        self.assertIn("the same grids as run", panel.grids_same.text())

        panel.show_run(None); draft.undo()
        self.assertEqual(draft.spec.accuracy.representation["solid2"]["deck"], {"solid": {"densityEcut": 20.0}})

    def test_an_unsettled_pilot_says_so_on_the_button(self):
        rec = copy.deepcopy(PILOTED); rec["inline@2"]["rasterPilot"]["settled"] = False
        run = Run(crystal(), FakeBackend()); run.result = Result(converged=True, energy_Ha=-7.1, structure=None, record=rec)
        panel = SetupPanel(SpecDraft(crystal()), BackendRegistry([FakeBackend()]))
        panel.show_run(run)
        self.assertIn("NOT SETTLED", panel.grids_save.text())


if __name__ == "__main__":
    unittest.main()
