"""The in-app agent's tool layer, shared draft, loop and consent gate (qviz.agent; doc/RunSpecPlan.md §5), on a
scripted model client.  No model is called and nothing leaves the box: Q4 is unanswered.

    .venv/bin/python -m unittest tests.test_agent
"""
import unittest

from qviz.agent.client import Message, ModelTurn, OfflinePlanner, ScriptedClient, ToolCall
from qviz.agent.consent import ConsentGate, ConsentRefused, OutboundPolicy
from qviz.draft import SpecDraft, merge_patch
from qviz.agent.session import AgentSession
from qviz.agent.tools import Toolset
from qviz.backends import FakeBackend
from qviz.backends.base import BackendRegistry
from qviz.spec import SpecError
from qviz.workspace import RunStatus, Workspace


def call(tool, **args):
    call.n = getattr(call, "n", 0) + 1
    return ModelTurn("", (ToolCall(f"c{call.n}", tool, args),))


class OffBox:
    """A client that WOULD leave the box -- it must never be reached while Q4 is unanswered."""
    name, leaves_box = "off-box", True

    def __init__(self):
        self.calls = 0

    def respond(self, system, messages, tools):
        self.calls += 1
        return ModelTurn("sent")


def rig(start_run=None):
    reg = BackendRegistry([FakeBackend()])
    ws = Workspace(registry=reg)
    draft = SpecDraft()
    return draft, ws, Toolset(draft, reg, ws, start_run=start_run)


class TheDraft(unittest.TestCase):
    def test_merge_patch_is_rfc7386(self):
        self.assertEqual(merge_patch({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": None, "e": 4}, "f": 5}),
                         {"a": 1, "b": {"d": 3, "e": 4}, "f": 5})

    def test_both_authors_share_one_undo_stack_and_an_invalid_patch_changes_nothing(self):
        draft, _ws, tools = rig()
        tools.call("build_structure", {"kind": "molecule", "name": "water"})
        draft.patch({"accuracy": {"epsilon_Ha_per_atom": 1e-6}}, "user", "tighter")
        tools.call("propose_spec_patch", {"patch": {"physics": {"functional": "LDA", "theory": "DFT"}}, "rationale": "LDA"})
        self.assertEqual([e.author for e in draft.history], ["agent", "user", "agent"])
        before = draft.spec
        with self.assertRaises(SpecError):
            draft.patch({"physics": {"spin": {"multiplicity": 2}}}, "user")      # wrong parity for water
        self.assertIs(draft.spec, before)
        draft.undo()
        self.assertEqual(draft.spec.physics.theory, "HF")
        self.assertEqual(draft.spec.accuracy.epsilon_Ha_per_atom, 1e-6, "the user's edit survives the agent's undo")


class TheTools(unittest.TestCase):
    def test_a_bad_call_is_a_result_not_an_exception(self):
        _d, _ws, tools = rig()
        self.assertIn("empty", tools.call("can_run", {})["error"])
        self.assertIn("no tool", tools.call("start_run", {})["error"])
        tools.call("build_structure", {"kind": "molecule", "name": "water"})
        self.assertIn("parity", tools.call("propose_spec_patch", {"patch": {"physics": {"spin": {"multiplicity": 2}}},
                                                                    "rationale": "x"})["error"])

    def test_no_tool_starts_cancels_or_deletes_a_run(self):
        _d, _ws, tools = rig()
        names = {s.name for s in tools.schemas}
        self.assertFalse(names & {"start_run", "cancel_run", "delete_run", "run_shell", "write_file"})
        self.assertIn("request_run", names)

    def test_a_run_request_is_a_card_only_the_users_click_starts(self):
        started = []
        _d, ws, tools = rig(start_run=lambda card: started.append(card) or [ws.add_run(card.spec, b) for b in card.backends])
        tools.call("build_structure", {"kind": "molecule", "name": "water"})
        r = tools.call("request_run", {"purpose": "a first look", "backends": ["Fake"]})
        self.assertEqual(r["card"], 1)
        self.assertEqual(ws.runs, [], "requesting runs nothing")
        tools.cards[0].accept()                                  # the user's click
        self.assertEqual(len(started), 1)
        self.assertEqual([x.status for x in ws.runs], [RunStatus.READY])
        res = tools.call("get_result", {"run_id": 0})
        self.assertTrue(res["is_result"] and res["converged"])
        self.assertIn("run #0", res["summary"])

    def test_a_request_on_a_back_end_that_cannot_run_the_draft_is_refused(self):
        _d, _ws, tools = rig()
        tools.call("build_structure", {"kind": "molecule", "name": "water"})
        self.assertIn("unknown", tools.call("request_run", {"purpose": "x", "backends": ["VASP"]})["error"])
        self.assertEqual(tools.cards, [])

    def test_crystal_prototypes_and_local_lookup(self):
        d, _ws, tools = rig()
        tools.call("build_structure", {"kind": "crystal", "prototype": "rocksalt", "elements": ["Na", "F"], "a_bohr": 8.73})
        self.assertEqual([s.element for s in d.spec.structure.sites], ["Na", "F"])
        self.assertIn("Fluorine (F2)", tools.call("lookup_structure", {"query": "fluorine"})["molecules"])
        self.assertIn("Kerker", tools.call("search_docs", {"query": "MnO"})["summary"])


class TheLoop(unittest.TestCase):
    def test_tool_results_reach_the_model_and_the_turn_ends_on_prose(self):
        _d, _ws, tools = rig()
        client = ScriptedClient([call("build_structure", kind="molecule", name="F2"), call("can_run"),
                                 lambda msgs: ModelTurn("F2 is drafted; " + msgs[-1].content["summary"])])
        s = AgentSession(client, tools)
        kinds = [e.kind for e in s.send("set up F2")]
        self.assertEqual(kinds, ["user", "tool_call", "tool_result", "tool_call", "tool_result", "assistant"])
        self.assertIn("Fake can run it", s.messages[-1].text)
        self.assertEqual(len(client.seen), 3)
        self.assertIn("request_run", client.seen[0][2])

    def test_the_offline_planner_drives_the_tools(self):
        d, _ws, tools = rig()
        s = AgentSession(OfflinePlanner(), tools)
        s.send("molecule water")
        s.send("set accuracy.epsilon_Ha_per_atom=1e-6")
        self.assertEqual(d.spec.accuracy.epsilon_Ha_per_atom, 1e-6)
        out = s.send("run on Fake")
        self.assertIn("waits for the user's click", out[-1].text)


class TheConsentGate(unittest.TestCase):
    def test_while_q4_is_unanswered_nothing_leaves_the_box(self):
        _d, _ws, tools = rig()
        client = OffBox()
        out = AgentSession(client, tools).send("hello")
        self.assertEqual(out[-1].kind, "refused")
        self.assertIn("Q4", out[-1].text)
        self.assertEqual(client.calls, 0)

    def test_licensed_results_and_unconsented_results_stay_home_even_with_a_policy(self):
        gate = ConsentGate(OutboundPolicy(True, send_results=True))
        with self.assertRaises(ConsentRefused):
            gate.outbound(OffBox(), [Message("tool", content={"energy_Ha": -1.0}, licensed=True)])
        specs_only = ConsentGate(OutboundPolicy(True, send_results=False))
        with self.assertRaises(ConsentRefused):
            specs_only.outbound(OffBox(), [Message("tool", content={"is_result": True})])
        self.assertTrue(specs_only.outbound(OffBox(), [Message("user", "hi")]))

    def test_an_external_query_defaults_to_no(self):
        self.assertFalse(ConsentGate().external_query("COD: rocksalt MnO"))
        self.assertTrue(ConsentGate(ask=lambda q: "COD" in q).external_query("COD: rocksalt MnO"))


class ThePanel(unittest.TestCase):
    """The Agent dock offscreen: the chat drives the tools, and only the card's Run button adds a run."""
    def test_the_card_button_is_the_only_start(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets
        from qviz.agent.panel import AgentPanel
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        draft, ws, _t = rig()
        tools = Toolset(draft, ws.registry, ws, start_run=lambda c: [ws.add_run(c.spec, b) for b in c.backends])
        panel = AgentPanel(AgentSession(OfflinePlanner(), tools), draft)
        for line in ("molecule F2", "set accuracy.epsilon_Ha_per_atom=1e-6", "run on Fake"):
            panel.entry.setText(line)
            panel._send()
        self.assertEqual(ws.runs, [])
        self.assertIn("1e-06", panel.draft_view.toPlainText())
        box = panel._cards.itemAt(0).widget()
        next(b for b in box.findChildren(QtWidgets.QPushButton) if b.text() == "Run").click()
        self.assertEqual([r.status for r in ws.runs], [RunStatus.READY])
        self.assertIn("accepted", box.title())
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
