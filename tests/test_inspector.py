"""The Inspector is a VIEW of the shared SpecDraft, beside the Agent dock (offscreen): one spec, one undo stack, one
refusal path.

    .venv/bin/python -m unittest tests.test_inspector
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
from qviz.inspector import Inspector                              # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class OneDraftTwoViews(unittest.TestCase):
    def setUp(self):
        reg = BackendRegistry([FakeBackend()])
        self.draft = SpecDraft(molecules.preset_spec("Water (H2O)"))
        self.insp = Inspector(self.draft, reg)
        self.agent = AgentPanel(AgentSession(OfflinePlanner(), Toolset(self.draft, reg)), self.draft)
        self.asked = []
        self.insp.add_requested.connect(lambda spec, names: self.asked.append((spec, names)))

    def say(self, line):
        self.agent.entry.setText(line)
        self.agent._send()

    def test_an_agent_edit_shows_in_the_inspector_and_an_inspector_edit_in_the_agents_draft(self):
        self.say("molecule F2")
        self.assertEqual(self.insp.mol_box.currentText(), "Fluorine (F2)")
        self.assertTrue(self.insp.pp_check.isChecked())
        self.assertEqual(self.insp.method_box.currentText(), "LDA")
        self.insp.basis_box.setCurrentText("sipp_sr"); self.insp.basis_box.activated.emit(0)
        self.assertEqual(self.draft.spec.accuracy.representation["basis"], "sipp_sr")
        self.assertIn('"sipp_sr"', self.agent.draft_view.toPlainText())
        self.assertEqual([e.author for e in self.draft.history], ["agent", "user"])

    def test_a_refused_inspector_edit_leaves_the_draft_and_the_widgets_as_they_were(self):
        before = self.draft.spec
        self.insp.mult_spin.setValue(2)                         # wrong parity for water
        self.assertIs(self.draft.spec, before)
        self.assertEqual(self.insp.mult_spin.value(), 0, "the widget re-renders from the draft")
        self.assertIn("parity", self.insp.reason.text())

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


if __name__ == "__main__":
    unittest.main()
