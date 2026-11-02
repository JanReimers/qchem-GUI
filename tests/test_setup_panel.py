"""The Setup panel is a VIEW of the shared SpecDraft, beside the Agent dock (offscreen): one spec, one undo stack, one
refusal path; a run's spec shows read-only; the Spec tab is the same draft as JSON; the Codes tab each code's translation.

    .venv/bin/python -m unittest tests.test_setup_panel
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import molecules                                        # noqa: E402
from qviz.agent.client import OfflinePlanner                      # noqa: E402
from qviz.agent.panel import AgentPanel                           # noqa: E402
from qviz.agent.session import AgentSession                       # noqa: E402
from qviz.agent.tools import Toolset                              # noqa: E402
from qviz.backends import FakeBackend                             # noqa: E402
from qviz.backends.base import BackendRegistry                    # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402
from qviz.setup_panel import SetupPanel                          # noqa: E402
from qviz.workspace import Workspace                             # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class OneDraftTwoViews(unittest.TestCase):
    def setUp(self):
        reg = BackendRegistry([FakeBackend()])
        self.draft = SpecDraft(molecules.preset_spec("Water (H2O)"))
        self.reg = reg
        self.insp = SetupPanel(self.draft, reg)
        self.agent = AgentPanel(AgentSession(OfflinePlanner(), Toolset(self.draft, reg)), self.draft)
        self.asked = []
        self.insp.run_requested.connect(lambda spec, names: self.asked.append((spec, names)))

    def say(self, line):
        self.agent.entry.setText(line)
        self.agent._send()

    def test_an_agent_edit_shows_in_the_setup_and_a_setup_edit_in_the_agents_draft(self):
        self.say("molecule F2")
        self.assertEqual(self.insp.mol_box.currentText(), "Fluorine (F2)")
        self.assertTrue(self.insp.pp_check.isChecked())
        self.assertEqual(self.insp.method_box.currentText(), "LDA")
        self.insp.basis_box.setCurrentText("sipp_sr"); self.insp.basis_box.activated.emit(0)
        self.assertEqual(self.draft.spec.accuracy.representation["basis"], "sipp_sr")
        self.assertIn('"sipp_sr"', self.insp.spec_view.toPlainText(), "the Spec tab is the same draft")
        self.assertEqual([e.author for e in self.draft.history], ["agent", "user"])

    def test_a_refused_setup_edit_leaves_the_draft_and_the_widgets_as_they_were(self):
        before = self.draft.spec
        self.insp.mult_spin.setValue(2)                         # wrong parity for water
        self.assertIs(self.draft.spec, before)
        self.assertEqual(self.insp.mult_spin.value(), 0, "the widget re-renders from the draft")
        self.assertIn("parity", self.insp.reason.text())

    def test_the_alpha_rows_show_for_linear_mixing_only(self):
        self.say("molecule F2")
        form, rows = self.insp._scf_form, (self.insp.alpha, self.insp.alpha_max)
        self.assertTrue(all(form.isRowVisible(b) for b in rows), "a molecule's Auto mixer is linear")
        self.insp.alpha_max.setValue(0.3); self.insp.alpha_max.editingFinished.emit()
        self.assertEqual(self.draft.spec.accuracy.scf["mixing_alpha_max"], 0.3)
        self.insp.mix_box.setCurrentText("Kerker"); self.insp.mix_box.activated.emit(0)
        self.assertFalse(any(form.isRowVisible(b) for b in rows), "α means nothing to Kerker")

    def test_one_undo_stack(self):
        self.say("set accuracy.epsilon_Ha_per_atom=1e-6")
        self.insp.sym_check.click()
        self.insp.undo_btn.click()                              # the user's symmetry edit
        self.assertTrue(self.insp.sym_check.isChecked())
        self.say("undo")                                        # the agent's ε edit
        self.assertEqual(self.draft.spec.accuracy.epsilon_Ha_per_atom, 1e-5)

    def test_add_asks_for_the_draft_and_a_crystal_draft_greys_the_molecular_controls(self):
        self.insp.add_btn.click()
        self.assertEqual(self.asked, [(self.draft.spec, ["Fake"])])
        self.say("set structure.kind=crystal")                 # refused: a crystal needs a lattice
        self.assertEqual(self.draft.spec.structure.kind, "molecule")
        Toolset(self.draft, BackendRegistry([FakeBackend()])).call(
            "build_structure", {"kind": "crystal", "prototype": "rocksalt", "elements": ["Na", "F"], "a_bohr": 8.73})
        self.assertFalse(self.insp.method_box.isEnabled())
        self.assertTrue(self.insp.mol_box.currentText().startswith("draft:"))

    def test_the_spec_tab_applies_through_the_draft_and_refuses_in_the_panel(self):
        text = self.insp.spec_view.toPlainText().replace('"epsilon_Ha_per_atom": 1e-05', '"epsilon_Ha_per_atom": 1e-07')
        self.insp.spec_view.setPlainText(text); self.insp.apply_btn.click()
        self.assertEqual(self.draft.spec.accuracy.epsilon_Ha_per_atom, 1e-7)
        self.assertEqual(self.insp.eps_box.currentText(), "1e-7")
        self.insp.spec_view.setPlainText("{not json"); self.insp.apply_btn.click()
        self.assertIn("not applied", self.insp.reason.text())

    def test_a_run_shows_read_only_and_copies_to_a_new_draft(self):
        ws = Workspace("t", registry=self.reg)
        run = ws.add_run(molecules.preset_spec("Oxygen (O2, triplet)"), backend="Fake", compute=False)
        self.insp.show_run(run)
        self.assertFalse(self.insp.method_box.isEnabled(), "a run's spec is not editable")
        self.assertFalse(self.insp._foot.isVisibleTo(self.insp), "no ▶ Run while viewing a run")
        self.assertTrue(self.insp.spec_view.isReadOnly())
        self.assertEqual(self.insp.mult_spin.value(), 3, "the run's spec, not the draft's")
        self.assertEqual(self.draft.spec.label, "Water", "viewing changes nothing")
        self.insp.copy_btn.click()
        self.assertIsNone(self.insp.viewing_run)
        self.assertEqual(self.draft.spec, run.spec)
        self.assertTrue(self.insp.method_box.isEnabled())

    def test_the_codes_tab_shows_each_ticked_code_s_translation(self):
        self.assertEqual([self.insp.codes.horizontalHeaderItem(j).text() for j in range(self.insp.codes.columnCount())], ["Fake"])
        self.assertEqual(self.insp.code_pick.currentText(), "Fake")
        self.assertTrue(self.insp.native.toPlainText())

    def test_the_structure_tab_lists_the_sites_and_says_detection_needs_the_extension(self):
        self.assertEqual(self.insp.sites.rowCount(), 3)
        self.assertIn("qchem_py", self.insp.sym_detected.text(), "no extension handed in: the readout says so")


class DetectedSymmetry(unittest.TestCase):
    def test_the_library_detects_and_the_classes_are_named(self):
        from qviz.backends.solid2 import _import_qchem_py
        from qviz.symmetry import detected
        qc, why = _import_qchem_py()
        if qc is None:
            self.skipTest(why)
        self.assertIn("C2v", detected(molecules.preset_spec("Water (H2O)"), qc)[0])
        from qviz.spec import molecule
        self.assertIn("O(3)", detected(molecule("Ne", [10], [0.0, 0.0, 0.0]), qc)[0])
        from tests.test_solid2_deck import SI, crystal, mno_plus_u
        si = detected(crystal("Si", *SI, accuracy={"representation": {"basis": "sipp_sr"}}), qc)
        self.assertEqual(si, ["space group: 48 operations, class m-3m (Oh), non-symmorphic"])
        mno = detected(mno_plus_u(), qc)
        self.assertEqual(len(mno), 2, "the AFM moments are seen")
        self.assertIn("class -3m (D3d)", mno[1])



class BasisDetails(unittest.TestCase):
    def test_the_shells_per_element_add_up_to_the_library_s_function_count(self):
        from qviz import basisinfo
        from qviz.backends.solid2 import _import_qchem_py
        from qviz.spec import Accuracy, molecule
        qc, why = _import_qchem_py()
        if qc is None:
            self.skipTest(why)
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107],
                     accuracy=Accuracy(representation={"basis": "dzvp"}))
        rows = basisinfo.element_rows(w)
        self.assertEqual(rows, [("O", 1, "3s 2p 1d", 15), ("H", 2, "2s", 2)], "dzvp.bsd, read past its free-text trailer")
        lines = basisinfo.summary(w, qc)
        self.assertEqual(lines[0], f"{sum(n * f for _, n, _, f in rows)} functions (dzvp, Cartesian)")
        self.assertIn("condition number", lines[1])


class ScfKnobs(unittest.TestCase):
    def setUp(self):
        from qviz.backends.solid2 import Solid2Backend
        self.be = Solid2Backend()
        self.draft = SpecDraft(molecules.preset_spec("Water (H2O)"))
        self.panel = SetupPanel(self.draft, BackendRegistry([FakeBackend(), self.be]))

    def test_diis_and_mom_knobs_reach_the_facade_s_arguments(self):
        p = self.panel
        p.acc_box.setCurrentText("Ladder"); p.acc_box.activated.emit(0)
        self.assertFalse(p.ladder_info.isHidden(), "the Ladder says what it does")
        p.diis_depth.setValue(8); p.diis_depth.editingFinished.emit()
        p.diis_sv.setCurrentText("1e-7"); p.diis_sv.activated.emit(0)
        p.mom_check.setChecked(True)
        self.assertFalse(p.mom_start.isHidden())
        p.mom_start.setValue(5); p.mom_start.editingFinished.emit()
        scf = self.draft.spec.accuracy.scf
        self.assertEqual((scf["diis_depth"], scf["diis_sv_tol"], scf["mom"], scf["mom_start"]), (8, 1e-7, True, 5))
        n = self.be.translate(self.draft.spec).native
        self.assertEqual((n["diis_depth"], n["diis_sv_tol"], n["diis_engage_fd"], n["mom"], n["mom_start"]), (8, 1e-7, 0.0, True, 5))
        p.mom_check.setChecked(False)
        self.assertNotIn("mom", self.draft.spec.accuracy.scf, "unticking removes the key (the lib's default)")

    def test_a_crystal_takes_mom_through_its_deck_and_refuses_diis_knobs(self):
        from qviz.backends import solid2_deck
        from tests.test_solid2_deck import SI, crystal
        si = crystal("Si", *SI, accuracy={"representation": {"basis": "sipp_sr"}, "scf": {"mom": True, "mom_start": 4}})
        self.assertEqual({k: v for k, v in solid2_deck.to_deck(si)["scf"].items() if "MOM" in k or "mom" in k},
                         {"useMOM": True, "momStartIter": 4})
        bad = crystal("Si", *SI, accuracy={"representation": {"basis": "sipp_sr"}, "scf": {"diis_depth": 6}})
        self.assertIn("no DIIS knobs", "; ".join(self.be.capabilities().can_run(bad).reasons))

    def test_the_offline_agent_explains_itself_in_plain_lines(self):
        from qviz.agent.client import OfflinePlanner, _offline_help
        text = _offline_help()
        self.assertTrue(text.startswith("No language model is connected yet"))
        self.assertIn("\n  molecule water", text)


def gridinfo_built(rec):
    from qviz import gridinfo
    return gridinfo.built(rec)[0]


class GridsTab(unittest.TestCase):
    def setUp(self):
        from qviz.backends.solid2 import _import_qchem_py
        self.qc, self.why = _import_qchem_py()

    def test_a_molecule_plans_its_mesh_from_epsilon_and_hf_has_none(self):
        from qviz import gridinfo
        from qviz.spec import molecule
        lda = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107], theory="DFT", functional="LDA")
        self.assertRegex(gridinfo.planned(lda)[0], r"XC mesh: \d+ radial × Lebedev \d+ per atom")
        hf = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107])
        self.assertIn("no XC mesh", gridinfo.planned(hf)[0])

    def test_build_fills_the_tabs_and_run_converges_that_build(self):
        if self.qc is None:
            self.skipTest(self.why)
        from qviz.backends.solid2 import Solid2Backend
        from tests.test_solid2_deck import SI, crystal
        si = crystal("Si", *SI, accuracy={"representation": {"basis": "sipp_sr"}})
        be = Solid2Backend()
        panel = SetupPanel(SpecDraft(si), BackendRegistry([FakeBackend(), be]), qc=self.qc)
        self.assertTrue(panel.build_btn.isEnabled())
        self.assertIn("not built yet", panel.grids_source.text())
        panel.build_btn.click()                        # no `serial`: runs here, synchronously
        APP.processEvents()
        self.assertEqual(panel.grids_source.text(), "from this spec's Build (no SCF)")
        rows = [[panel.grids_table.item(r, c).text() for c in range(5)] for r in range(panel.grids_table.rowCount())]
        self.assertTrue(rows and rows[0][4] == "reference", rows)
        self.assertIn("density Ecut floor", panel.grids_built.text())
        self.assertIn("condition number", panel.basis_summary.text(), "a crystal's per-k overlap, from the same record")

        made = []
        real = self.qc.SolidRun
        self.qc.SolidRun = lambda deck: made.append(deck) or real(deck)
        try:
            job = be.translate(si)
            res = be.run(job)                          # the kept Build of this deck: converged, not rebuilt
            self.assertEqual(made, [])
            self.assertTrue(any(k.split("@")[0].endswith(" build") for k in res.record), "the build run stays in the record")
            self.assertIn("density Ecut floor", "\n".join(gridinfo_built(job.live)))
            be.run(be.translate(si))                   # the Build was consumed: a second Run builds afresh
            self.assertEqual(len(made), 1)
        finally:
            self.qc.SolidRun = real

    def test_the_draft_reads_the_newest_run_of_its_own_spec(self):
        from qviz.backends.base import Result
        from qviz.workspace import Run
        spec = molecules.preset_spec("Water (H2O)")
        run = Run(spec, FakeBackend())
        run.result = Result(converged=True, energy_Ha=-1.0, structure=None,
                            record={"water@t": {"grids": {"xcQuadrature": {"kind": "Becke", "natom": 3, "nRadial": 50,
                                                                           "nDirs": 194, "L": 23, "points": 29100}}}})
        runs = []
        panel = SetupPanel(SpecDraft(spec), BackendRegistry([FakeBackend()]), runs=lambda: runs)
        self.assertNotIn("XC mesh", panel.grids_built.text())
        runs.append(run); panel.refresh_records()
        self.assertIn("from the run of this spec", panel.grids_source.text())
        self.assertIn("29,100 points", panel.grids_built.text())

    def test_a_molecule_cannot_build_yet(self):
        from qviz.backends.solid2 import Solid2Backend
        panel = SetupPanel(SpecDraft(molecules.preset_spec("Water (H2O)")), BackendRegistry([FakeBackend(), Solid2Backend()]))
        self.assertFalse(panel.build_btn.isEnabled())
        self.assertIn("build-only step", panel.build_btn.toolTip())

    def test_a_molecule_run_carries_its_record(self):
        if self.qc is None:
            self.skipTest(self.why)
        from qviz import gridinfo
        from qviz.backends.solid2 import Solid2Backend
        from qviz.spec import Accuracy, molecule
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107], theory="DFT", functional="LDA",
                     accuracy=Accuracy(representation={"basis": "dzvp"}))
        be = Solid2Backend()
        job = be.translate(w)
        res = be.run(job)
        self.assertIn("basis", job.live["running@live"], "the record so far, taken at iteration 1 on the run's own thread")
        lines, rows = gridinfo.built(res.record)
        self.assertRegex(lines[0], r"^XC mesh: Becke, 3 atoms × \d+ radial × \d+ directions \(Lebedev L=\d+\) = [\d,]+ points",
                         "the mesh's own report (grids.xcQuadrature), filed before iteration 1")
        self.assertEqual(gridinfo.built(job.live)[0][0], lines[0], "the live snapshot already has it")
        self.assertEqual(rows, [], "a molecule has no density ladder")
        cond = gridinfo.conditioning(res.record)
        self.assertEqual(cond[0], "19 functions (dzvp)", "the run's basis section, read")
        self.assertIn("condition number", cond[1])
        self.assertRegex(res.resolved["SCF gates"], r"^run: \[F,D\] 1e-05, Δρ 0.0001; at most \d+ iterations$",
                         "the gates the record says the SCF used; off (1e30) gates are not listed")
        self.assertTrue(be.translate(w).resolved["SCF gates"].startswith("planned:"))

    def test_a_mesh_that_reports_itself_is_read_as_filed(self):
        from qviz import gridinfo
        rec = {"water@t": {"grids": {"xcQuadrature": {"kind": "Becke", "natom": 3, "radial": "MHL", "nRadial": 50,
                                                      "angular": "Lebedev", "L": 23, "nDirs": 194, "points": 29100,
                                                      "dropped": 0}},
                           "folds": {"XC mesh": {"ops": 0, "raw": 29100, "reps": 29100, "factor": 1.0}}}}
        lines, _ = gridinfo.built(rec)
        self.assertEqual(lines, ["XC mesh: Becke, 3 atoms × 50 radial × 194 directions (Lebedev L=23) = 29,100 points"])

if __name__ == "__main__":
    unittest.main()
