"""
The Anthropic client against a STUBBED SDK: the request it builds, how a turn comes back, and the panel running a model
call off the GUI thread.  No network: nothing here needs ANTHROPIC_API_KEY.
"""
import os
import time
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import molecules                                        # noqa: E402
from qviz.agent.client import AnthropicClient, Message, ToolCall, _to_api   # noqa: E402
from qviz.agent.consent import ConsentGate, Q4_RULED, Q4_UNANSWERED          # noqa: E402
from qviz.agent.panel import AgentPanel                           # noqa: E402
from qviz.agent.session import AgentSession                       # noqa: E402
from qviz.agent.tools import Toolset                              # noqa: E402
from qviz.backends import FakeBackend                             # noqa: E402
from qviz.backends.base import BackendRegistry                    # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Block(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


def text(t):
    return _Block(type="text", text=t)


def tool_use(i, name, args):
    return _Block(type="tool_use", id=i, name=name, input=args)


def thinking(sig):
    return _Block(type="thinking", thinking="", signature=sig)


class FakeSDK:
    """anthropic.Anthropic's shape, as far as AnthropicClient uses it: beta.messages.create(**kw) -> a response."""
    def __init__(self, *responses):
        self._responses, self.requests = list(responses), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self._responses.pop(0)


def reply(*content, stop="end_turn", details=None):
    return SimpleNamespace(content=list(content), stop_reason=stop, stop_details=details)


def session(sdk, policy=Q4_RULED):
    draft = SpecDraft(molecules.preset_spec("Water (H2O)"))
    tools = Toolset(draft, BackendRegistry([FakeBackend()]))
    return AgentSession(AnthropicClient(sdk=sdk), tools, ConsentGate(policy)), draft


class TheRequest(unittest.TestCase):
    def test_one_call_per_turn_with_caching_fallback_and_the_tools(self):
        sdk = FakeSDK(reply(thinking("s1"), text("Checking."), tool_use("t1", "can_run", {}), stop="tool_use"),
                      reply(text("It can run on Fake.")))
        s, _ = session(sdk)
        events = s.send("can it run?")
        self.assertEqual([e.kind for e in events], ["user", "assistant", "tool_call", "tool_result", "assistant"])
        first, second = sdk.requests
        self.assertEqual(first["model"], "claude-opus-5-5")
        self.assertEqual(first["cache_control"], {"type": "ephemeral"})
        self.assertEqual((first["fallbacks"], first["betas"]), ("default", ["server-side-fallback-2026-07-01"]))
        self.assertIn("can_run", [t["name"] for t in first["tools"]])
        self.assertNotIn("temperature", first)
        # the second call hands the assistant turn back VERBATIM (thinking block and signature included) and the
        # tool result as one user message answering it
        self.assertEqual(second["messages"][1]["content"][0], {"type": "thinking", "thinking": "", "signature": "s1"})
        self.assertEqual(second["messages"][2]["role"], "user")
        self.assertEqual(second["messages"][2]["content"][0]["tool_use_id"], "t1")

    def test_consecutive_tool_results_fold_into_one_user_message(self):
        msgs = [Message("user", "go"),
                Message("assistant", "", (ToolCall("a", "x", {}), ToolCall("b", "y", {}))),
                Message("tool", tool_call_id="a", content={"summary": "ok"}),
                Message("tool", tool_call_id="b", content={"error": "bad"})]
        api = _to_api(msgs)
        self.assertEqual([m["role"] for m in api], ["user", "assistant", "user"])
        self.assertEqual([b["tool_use_id"] for b in api[2]["content"]], ["a", "b"])
        self.assertTrue(api[2]["content"][1]["is_error"])

    def test_a_refusal_and_a_cut_off_reply_end_the_turn_without_running_tools(self):
        s, _ = session(FakeSDK(reply(stop="refusal", details=SimpleNamespace(explanation="policy"))))
        self.assertIn("declined", s.send("x")[-1].text)
        s, draft = session(FakeSDK(reply(text("Half"), tool_use("t", "undo_last_edit", {}), stop="max_tokens")))
        events = s.send("x")
        self.assertNotIn("tool_call", [e.kind for e in events], "a truncated tool call never runs")
        self.assertIn("cut off", events[-1].text)

    def test_the_gate_still_holds(self):
        s, _ = session(FakeSDK(), policy=Q4_UNANSWERED)
        self.assertEqual(s.send("x")[-1].kind, "refused")

    def test_no_key_no_client(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            self.assertIn("ANTHROPIC_API_KEY", AnthropicClient.why_absent())
            with self.assertRaises(RuntimeError):
                AnthropicClient()
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old


class ThePanel(unittest.TestCase):
    def test_the_model_call_runs_off_the_gui_thread_and_tools_on_it(self):
        import threading
        gui = threading.current_thread()
        where = {}

        class Slow(FakeSDK):
            def _create(self, **kw):
                where["model"] = threading.current_thread()
                time.sleep(0.2)
                return super()._create(**kw)
        sdk = Slow(reply(tool_use("t", "build_structure", {"kind": "molecule", "name": "methanol", "atoms": [
                       {"element": "C", "xyz_angstrom": [0, 0, 0]}, {"element": "O", "xyz_angstrom": [1.43, 0, 0]},
                       {"element": "H", "xyz_angstrom": [-0.36, 1.03, 0]}, {"element": "H", "xyz_angstrom": [-0.36, -0.51, 0.89]},
                       {"element": "H", "xyz_angstrom": [-0.36, -0.51, -0.89]}, {"element": "H", "xyz_angstrom": [1.75, 0.9, 0]}]}),
                       stop="tool_use"),
                   reply(text("Methanol is in the draft; the geometry is approximate.")))
        s, draft = session(sdk)
        s.tools.draft.on_change(lambda *_: where.setdefault("tool", threading.current_thread()))
        panel = AgentPanel(s, draft)
        panel.entry.setText("set up methanol"); panel._send()
        self.assertFalse(panel.entry.isEnabled(), "one message at a time")
        t0 = time.time()
        while s.busy and time.time() - t0 < 10:
            APP.processEvents(); time.sleep(0.01)
        self.assertFalse(s.busy)
        self.assertTrue(panel.entry.isEnabled())
        self.assertIsNot(where["model"], gui)
        self.assertIs(where["tool"], gui, "the draft is GUI-thread state")
        self.assertEqual(draft.spec.label, "methanol")
        self.assertEqual(len(draft.spec.structure.sites), 6)
        self.assertIn("approximate", panel.chat.toPlainText())


if __name__ == "__main__":
    unittest.main()
