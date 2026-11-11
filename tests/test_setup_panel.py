"""The Setup panel is a VIEW of the shared SpecDraft, beside the Agent dock (offscreen): one spec, one undo stack, one
refusal path; a run's spec shows read-only; the Spec tab is the same draft as JSON; the Codes tab each code's translation.

    .venv/bin/python -m unittest tests.test_setup_panel
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
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
import tempfile as _bse_tmp; from qviz import bse as _bse; _bse.BASES_DIR = _bse_tmp.mkdtemp(prefix="qchem-gui-bases-")   # never ~/.solid2/bases

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

    def test_new_draft_starts_fresh_from_a_preset_even_while_a_run_is_shown(self):
        ws = Workspace("t", registry=self.reg)
        run = ws.add_run(molecules.preset_spec("Oxygen (O2, triplet)"), backend="Fake", compute=False)
        self.insp.show_run(run)
        self.insp._fill_new_menu()
        mol = next(a.menu() for a in self.insp.new_menu.actions() if a.text() == "Molecule")
        next(a for a in mol.actions() if a.text() == "Methane (CH4)").trigger()
        self.assertIsNone(self.insp.viewing_run, "the new draft is shown, editable")
        self.assertEqual(self.draft.spec, molecules.preset_spec("Methane (CH4)"), "the preset's default settings")
        self.assertTrue(self.insp.method_box.isEnabled())
        self.assertEqual(self.draft.history[-1].rationale, "new draft: preset Methane (CH4)")

    def test_the_codes_tab_shows_each_ticked_code_s_translation(self):
        self.assertEqual([self.insp.codes.horizontalHeaderItem(j).text() for j in range(self.insp.codes.columnCount())], ["Fake"])
        self.assertEqual(self.insp.code_pick.currentText(), "Fake")
        self.assertTrue(self.insp.native.toPlainText())

    def test_the_structure_tab_lists_the_sites_and_says_detection_needs_the_extension(self):
        self.assertEqual(self.insp.sites.rowCount(), 3)
        self.assertIn("qchem_py", self.insp.sym_detected.text(), "no extension handed in: the readout says so")


class CrystalBasis(unittest.TestCase):
    """A picked crystal states no basis (the orbital basis is user selected): the Basis box offers the periodic bases that
    cover its elements, and picking one unlocks Build and SOLID² -- the way out of item 26 (2026-11-11)."""

    def test_a_crystal_picks_a_periodic_basis_covering_its_elements_and_build_says_why_until_then(self):
        from qviz.backends.solid2 import Solid2Backend
        from qviz.spec import RunSpec
        nio = RunSpec.from_dict({"label": "NiO", "structure": {"kind": "crystal",
                                 "lattice_bohr": [[0, 3.94, 3.94], [3.94, 0, 3.94], [3.94, 3.94, 0]],
                                 "sites": [{"element": "Ni", "frac": [0, 0, 0]}, {"element": "O", "frac": [0.5, 0.5, 0.5]}]},
                                 "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential"},
                                 "accuracy": {"kmesh": [1, 1, 1]}}).validate()
        draft = SpecDraft(nio)
        be = Solid2Backend()
        panel = SetupPanel(draft, BackendRegistry([FakeBackend(), be]))
        box = panel.basis_box
        self.assertTrue(box.isEnabled())
        items = [box.itemText(i) for i in range(box.count())]
        self.assertEqual(items[0], "— choose a basis —")
        self.assertIn("valence_lowq_va", items)
        self.assertNotIn("dzvp", items, "a molecular basis has no pseudopotential valence block")
        self.assertNotIn("sipp_sr", items, "sipp_sr covers Si and O, not Ni")
        self.assertFalse(panel.method_box.isEnabled())
        self.assertIn("periodic facade", panel.method_box.toolTip(), "a disabled control says why")
        if be.why_no_build(nio) != "Orbital basis must be user selected":
            self.skipTest(be.why_no_build(nio))       # SOLID² not built here
        self.assertFalse(panel.build_btn.isEnabled())
        self.assertIn("Orbital basis must be user selected", panel.build_why.text(), "the reason in plain sight, not a tooltip")
        box.setCurrentIndex(items.index("valence_lowq_va")); box.activated.emit(0)
        self.assertEqual(draft.spec.accuracy.representation, {"basis": "valence_lowq_va"})
        self.assertTrue(panel.build_btn.isEnabled())
        self.assertFalse(panel.build_why.isVisibleTo(panel))
        self.assertTrue(panel.be_checks["SOLID²"].isEnabled())


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
        self.assertIn("Build (or Run) shows", basisinfo.summary(w)[0], "the conditioning is the library's record, not recomputed")


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

    def test_a_crystal_takes_mom_diis_and_the_ladder_through_its_deck(self):
        from qviz.backends import solid2_deck
        from tests.test_solid2_deck import SI, crystal
        si = crystal("Si", *SI, accuracy={"representation": {"basis": "sipp_sr"}, "scf": {
            "mom": True, "mom_start": 4, "diis_depth": 6, "diis_sv_tol": 1e-7, "accelerator": "Ladder", "ladder_switch_at": 1e-5,
            "mixer": "Linear", "mixing_alpha_min": 0.05}})
        deck = solid2_deck.to_deck(si)
        self.assertEqual({k: v for k, v in deck["scf"].items() if "MOM" in k or "mom" in k}, {"useMOM": True, "momStartIter": 4})
        self.assertEqual(deck["acceleratorOptions"], {"diis": {"nProj": 6, "svTol": 1e-7}, "ladder": {"switchAt": 1e-5}})
        self.assertEqual(deck["scf"]["minRelaxRo"], 0.05)
        self.assertTrue(self.be.capabilities().can_run(si))

    def test_the_gates_reach_the_facade_and_the_deck(self):
        from qviz.backends import solid2_deck
        from tests.test_solid2_deck import SI, crystal
        p = self.panel
        p.gates["gate_de"].setCurrentText("1e-8"); p.gates["gate_de"].activated.emit(p.gates["gate_de"].currentIndex())
        p.gates["gate_fd"].setCurrentText("off"); p.gates["gate_fd"].activated.emit(p.gates["gate_fd"].currentIndex())
        scf = self.draft.spec.accuracy.scf
        self.assertEqual((scf["gate_de"], scf["gate_fd"]), (1e-8, "off"))
        n = self.be.translate(self.draft.spec).native
        self.assertEqual((n["gate_fd"], n["gate_drho"], n["gate_de"], n["gate_dfd"]), (0.0, -1.0, 1e-8, -1.0))
        p.gates["gate_fd"].setCurrentText("1e-5"); p.gates["gate_fd"].activated.emit(p.gates["gate_fd"].currentIndex())
        self.assertNotIn("gate_fd", self.draft.spec.accuracy.scf, "picking the default's value removes the key")
        si = crystal("Si", *SI, accuracy={"representation": {"basis": "sipp_sr"}, "scf": {"gate_drho": 1e-6, "gate_dfd": "off"}})
        deck = solid2_deck.to_deck(si)["scf"]
        self.assertEqual((deck["minDeltaRhoPerElectron"], deck["minDeltaFD"]), (1e-6, "off"))

    def test_an_unstated_gate_shows_its_value(self):
        p = self.panel
        self.assertEqual({k: b.currentText() for k, b in p.gates.items()},
                         {"gate_fd": "1e-5", "gate_drho": "1e-4", "gate_de": "off", "gate_dfd": "off"})

    def test_auto_and_default_say_what_they_resolve_to(self):
        p = self.panel                                                     # the fixture's draft: water, HF, all-electron
        self.assertEqual(p.mult_spin.text(), "auto: 1, singlet, unpolarized")
        self.assertEqual((p.diis_depth.text(), p.diis_sv.itemText(0), p.diis_engage.itemText(0)),
                         ("default (4)", "default (5e-9)", "default (0.25, Z-scaled)"))
        self.draft.patch({"physics": {"theory": "DFT", "functional": "LDA"}}, "test")
        self.assertEqual(p.diis_engage.itemText(0), "default (from the start)")

    def test_the_xc_mesh_knobs_show_eps_s_mesh_and_state_both_when_one_is_set(self):
        from qviz.backends.solid2 import mesh_for
        p = self.panel
        self.draft.patch({"physics": {"theory": "DFT", "functional": "LDA"}}, "test")
        mr, ma = mesh_for(self.draft.spec.accuracy.epsilon_Ha_per_atom)
        self.assertEqual((p.mesh_radial.value(), p.mesh_angular.currentText()), (mr, str(ma)))
        self.assertTrue(p.mesh_reset.isHidden(), "ε's own mesh: nothing to go back to")
        self.assertFalse(p.mesh_auto.isHidden())
        self.assertEqual(p.mesh_auto.text(), f"auto, from ε = {self.draft.spec.accuracy.epsilon_Ha_per_atom:g}")
        p.mesh_angular.setCurrentText("17"); p.mesh_angular.activated.emit(p.mesh_angular.currentIndex())
        own = self.draft.spec.accuracy.representation["solid2"]
        self.assertEqual((own["mesh_radial"], own["mesh_angular"]), (mr, 17))
        self.assertEqual((self.be.translate(self.draft.spec).native["mesh_radial"],
                          self.be.translate(self.draft.spec).native["mesh_angular"]), (mr, 17))
        self.assertFalse(p.mesh_reset.isHidden()); self.assertTrue(p.mesh_auto.isHidden())
        p.mesh_reset.click()
        self.assertFalse(p.mesh_auto.isHidden())
        self.assertNotIn("mesh_angular", (self.draft.spec.accuracy.representation or {}).get("solid2") or {})

    def test_the_ladder_switch_over_and_alpha_min_reach_the_molecular_facade(self):
        p = self.panel
        p.acc_box.setCurrentText("Ladder"); p.acc_box.activated.emit(0)
        self.assertFalse(p.ladder_switch.isHidden(), "the Ladder's hand-off is a knob")
        p.ladder_switch.setCurrentText("1e-5"); p.ladder_switch.activated.emit(p.ladder_switch.currentIndex())
        self.assertEqual(self.be.translate(self.draft.spec).native["ladder_switch_at"], 1e-5)
        p.ladder_switch.setCurrentIndex(p.ladder_switch.count() - 1); p.ladder_switch.activated.emit(p.ladder_switch.currentIndex())
        self.assertEqual(self.be.translate(self.draft.spec).native["ladder_switch_at"], 0.0, "off = on stall only")
        p.ladder_switch.setCurrentIndex(0); p.ladder_switch.activated.emit(0)
        self.assertEqual(self.be.translate(self.draft.spec).native["ladder_switch_at"], -1.0, "default = the lib's")
        p.mix_box.setCurrentText("Linear"); p.mix_box.activated.emit(0)
        p.alpha_min.setValue(0.1); p.alpha_min.editingFinished.emit()
        self.assertEqual(self.be.translate(self.draft.spec).native["relax_min"], 0.1)

    def test_the_offline_agent_explains_itself_in_plain_lines(self):
        from qviz.agent.client import OfflinePlanner, _offline_help
        text = _offline_help()
        self.assertTrue(text.startswith("No language model is connected"))
        self.assertIn("ANTHROPIC_API_KEY", text, "it says how to connect Claude")
        self.assertIn("\n  molecule water", text)


class XyzExport(unittest.TestCase):
    def test_a_molecule_round_trips_and_a_crystal_carries_its_cell(self):
        import tempfile, pathlib
        from qviz.molecules import from_xyz, to_xyz
        w = molecules.preset_spec("Water (H2O)")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "water.xyz"
            p.write_text(to_xyz(w))
            Z, pos, _ = from_xyz(str(p))
        self.assertEqual(Z, [s.Z for s in w.structure.sites])
        for a, b in zip(pos, [c for s in w.structure.sites for c in s.xyz]):
            self.assertAlmostEqual(a, b, places=6)
        from tests.test_solid2_deck import SI, crystal
        text = to_xyz(crystal("Si", *SI))
        self.assertIn('Lattice="', text.splitlines()[1])
        self.assertEqual(text.splitlines()[0], "2")

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

    def test_hartree_says_how_it_is_done_in_grids_and_basis(self):
        from qviz import basisinfo, gridinfo
        from qviz.spec import molecule
        xyz = [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107]
        lda = molecule("water", [8, 1, 1], xyz, theory="DFT", functional="LDA")
        hf = molecule("water", [8, 1, 1], xyz)
        self.assertIn("analytic Dunlap fit on A1_coul", gridinfo.planned(lda)[1])
        self.assertIn("no mesh", gridinfo.planned(lda)[1], "the default route projects ρ by 3-centre integrals, not on a mesh")
        mesh = {"kind": "Becke", "natom": 3, "nRadial": 50, "nDirs": 194, "L": 23, "points": 29100}
        dunlap = gridinfo.built({"r": {"hamiltonian": {"Hartree": {"fit": "A1_coul", "auxiliaryFit": mesh,
                                                                   "fitbasis": [{"role": "Hartree ρ fit", "name": "A1_coul", "nFunctions": 70}]}}}})[0]
        self.assertIn("Hartree: analytic Dunlap fit on A1_coul (3-centre Coulomb integrals), no mesh", dunlap)
        self.assertTrue(any(l.startswith("Hartree fit basis's own mesh (seed projection, normalisation)") for l in dunlap), dunlap)
        delta = gridinfo.built({"r": {"hamiltonian": {"Hartree": {"coulombFit": "delta", "auxiliaryFit": mesh,
                                                                  "fitbasis": [{"role": "Hartree ρ fit", "name": "DeltaFit", "nFunctions": 9}]}}}})[0]
        self.assertIn("Hartree: ρ on a mesh (delta Coulomb fit)", delta)
        self.assertTrue(any(l.startswith("Hartree fit mesh:") for l in delta), delta)
        self.assertIn("Hartree: analytic", gridinfo.planned(hf)[1])
        self.assertIn("analytic", basisinfo.fits(hf, None)[0])
        self.assertIn("Build (or Run) fills this", basisinfo.fits(lda, None)[0])
        rec = {"run": {"hamiltonian": {"Vxc_DFT": {"fitbasis": [{"role": "Vxc fit", "name": "DeltaFit", "nFunctions": 900}]},
                                        "Hartree": {"fitbasis": [{"role": "Hartree ρ fit", "name": "A1_coul", "nFunctions": 70}]}}}}
        self.assertEqual(basisinfo.fits(lda, rec), ["Hartree ρ fit: A1_coul, 70 functions", "Vxc fit: DeltaFit, 900 functions"],
                         "read by key: the record's member order is construction order")

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
                            record={"water@t": {"hamiltonian": {"xcQuadrature": {"kind": "Becke", "natom": 3, "nRadial": 50,
                                                                           "nDirs": 194, "L": 23, "points": 29100}}}})
        runs = []
        panel = SetupPanel(SpecDraft(spec), BackendRegistry([FakeBackend()]), runs=lambda: runs)
        self.assertNotIn("XC mesh", panel.grids_built.text())
        runs.append(run); panel.refresh_records()
        self.assertIn("from the run of this spec", panel.grids_source.text())
        self.assertIn("29,100 points", panel.grids_built.text())

    def test_a_molecule_builds_and_run_converges_that_build(self):
        if self.qc is None:
            self.skipTest(self.why)
        from qviz import gridinfo
        from qviz.backends.solid2 import Solid2Backend
        from qviz.spec import Accuracy, molecule
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107], theory="DFT", functional="LDA",
                     accuracy=Accuracy(representation={"basis": "dzvp"}))
        be = Solid2Backend()
        panel = SetupPanel(SpecDraft(w), BackendRegistry([FakeBackend(), be]), qc=self.qc)
        self.assertTrue(panel.build_btn.isEnabled(), panel.build_btn.toolTip())
        panel.build_btn.click()                        # no `serial`: runs here, synchronously
        APP.processEvents()
        self.assertEqual(panel.grids_source.text(), "from this spec's Build (no SCF)")
        self.assertRegex(panel.grids_built.text(), r"XC mesh: Becke, 3 atoms", "the mesh, from the build run")
        self.assertIn("19 functions (dzvp)", panel.basis_summary.text())
        self.assertIn("condition number", panel.basis_summary.text(), "basis.perIrrep, from the same record")

        made = []
        real = self.qc.Calculator
        self.qc.Calculator = lambda *a, **k: made.append(k) or real(*a, **k)
        try:
            rows = []
            res = be.run(be.translate(w), rows.append)   # the kept Build of this job: converged, not rebuilt
            self.assertEqual(made, [])
            self.assertTrue(res.converged, res.failure)
            self.assertEqual(rows[0].iteration, 1, "the SCF on the built graph streams from its first iteration")
            self.assertTrue(any(k.split("@")[0].endswith(" build") for k in res.record), "the build run stays in the record")
            self.assertTrue(gridinfo.built(res.record)[0], "the SCF run's own mesh report")
            fresh = be.run(be.translate(w))            # the Build was consumed: a second Run builds afresh
            self.assertEqual(len(made), 1)
            self.assertAlmostEqual(fresh.energy_Ha, res.energy_Ha, delta=1e-8, msg="Build -> Run is the same calculation")
        finally:
            self.qc.Calculator = real

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
                         "the mesh's own report (hamiltonian.xcQuadrature), filed before iteration 1")
        self.assertEqual(gridinfo.built(job.live)[0][0], lines[0], "the live snapshot already has it")
        self.assertEqual(rows, [], "a molecule has no density ladder")
        cond = gridinfo.conditioning(res.record)
        self.assertEqual(cond[0], "19 functions (dzvp)", "the run's basis section, read")
        self.assertIn("condition number", cond[1])
        self.assertRegex(res.resolved["SCF gates"], r"^run: \[F,D\] 1e-05, density change 0.0001; at most \d+ iterations$",
                         "the gates the record says the SCF used; off (1e30) gates are not listed")
        self.assertTrue(be.translate(w).resolved["SCF gates"].startswith("planned:"))

    def test_a_fit_grid_is_named_for_whose_it_is(self):
        from qviz import gridinfo
        fit = {"kind": "uniform", "N": [8, 8, 8], "points": 512}
        rec = {"si@t": {"hamiltonian": {"Hartree": {"densityFit": fit}, "Vxc_DFT": {"vxcFitGrid": fit}},
                        "seedDensity": {"densityFit": fit}, "densityMixer": {"vxcFitGrid": fit}}}
        self.assertEqual([l.split(":")[0] for l in gridinfo.built(rec)[0]],
                         ["density fit", "Vxc fit", "seed's density fit", "mixer's Vxc fit"])
        mixer_only = {"si@t": {"hamiltonian": {"xcQuadrature": {"kind": "Becke"}}, "densityMixer": {"vxcFitGrid": fit}}}
        self.assertFalse(any(l.startswith("Vxc fit") for l in gridinfo.built(mixer_only)[0]), "a Becke run's only raster is the mixer's")

    def test_a_mesh_that_reports_itself_is_read_as_filed(self):
        from qviz import gridinfo
        rec = {"water@t": {"hamiltonian": {"xcQuadrature": {"kind": "Becke", "natom": 3, "radial": "MHL", "nRadial": 50,
                                                      "angular": "Lebedev", "L": 23, "nDirs": 194, "points": 29100,
                                                      "dropped": 0}},
                           "folds": {"XC mesh": {"ops": 0, "raw": 29100, "reps": 29100, "factor": 1.0}}}}
        lines, _ = gridinfo.built(rec)
        self.assertEqual(lines, ["XC mesh: Becke, 3 atoms × 50 radial × 194 directions (Lebedev L=23) = 29,100 points"])

if __name__ == "__main__":
    unittest.main()
