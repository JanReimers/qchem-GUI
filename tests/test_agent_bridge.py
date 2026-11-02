"""
Claude Code's road into the GUI: scripts/qchem_mcp.py (spoken to over stdio, as Claude Code does) -> the GUI's localhost
bridge -> the GUI's own Toolset and draft.  A real subprocess and a real socket; no Claude involved.
"""
import json
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import molecules                                        # noqa: E402
from qviz.agent.bridge import AgentBridge, withhold_licensed      # noqa: E402
from qviz.agent.tools import Toolset                              # noqa: E402
from qviz.backends import FakeBackend                             # noqa: E402
from qviz.backends.base import BackendRegistry                    # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ROOT = pathlib.Path(__file__).resolve().parent.parent


class McpOverStdio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.file = pathlib.Path(self.tmp.name) / "agent-bridge.json"
        self.draft = SpecDraft(molecules.preset_spec("Water (H2O)"))
        self.calls = []
        self.bridge = AgentBridge(Toolset(self.draft, BackendRegistry([FakeBackend()])),
                                  on_call=lambda n, a, r: self.calls.append(n), path=self.file)
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "qchem_mcp.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, env={**os.environ, "QCHEM_BRIDGE_FILE": str(self.file)})
        self.out = queue.Queue()
        threading.Thread(target=lambda: [self.out.put(l) for l in self.proc.stdout], daemon=True).start()
        self.n = 0

    def tearDown(self):
        self.proc.stdin.close(); self.proc.wait(5)
        self.bridge.close(); self.tmp.cleanup()

    def rpc(self, method, params=None):
        self.n += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        t0 = time.time()
        while time.time() - t0 < 20:
            APP.processEvents()                      # the bridge answers on this (the GUI) thread
            try:
                return json.loads(self.out.get(timeout=0.01))
            except queue.Empty:
                pass
        self.fail(f"no reply to {method}")

    def test_claude_code_sees_the_tools_and_edits_the_gui_s_draft(self):
        init = self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}})
        self.assertEqual(init["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("shared draft", init["result"]["instructions"])
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        names = [t["name"] for t in self.rpc("tools/list")["result"]["tools"]]
        self.assertIn("propose_spec_patch", names)
        r = self.rpc("tools/call", {"name": "propose_spec_patch", "arguments": {
            "patch": {"accuracy": {"scf": {"accelerator": "Ladder"}}}, "rationale": "the user asked for Ladder"}})["result"]
        self.assertFalse(r["isError"], r)
        self.assertEqual(self.draft.spec.accuracy.scf["accelerator"], "Ladder", "the GUI's own draft changed")
        self.assertEqual(self.calls, ["propose_spec_patch"], "the Agent dock is told")
        bad = self.rpc("tools/call", {"name": "no_such_tool", "arguments": {}})["result"]
        self.assertTrue(bad["isError"])

    def test_a_closed_gui_is_said_plainly_and_the_tools_still_list(self):
        self.bridge.close()
        self.assertFalse(self.file.exists())
        self.assertIn("propose_spec_patch", [t["name"] for t in self.rpc("tools/list")["result"]["tools"]])
        r = self.rpc("tools/call", {"name": "get_spec", "arguments": {}})["result"]
        self.assertTrue(r["isError"])
        self.assertIn("not running", r["content"][0]["text"])

    def test_the_token_and_the_licence(self):
        self.assertIn("bad token", self.bridge.handle(b'{"token": "x", "op": "list"}')["error"])
        self.assertTrue(withhold_licensed({"licensed": True, "energy": -1.0})["withheld"])
        self.assertEqual(oct(self.file.stat().st_mode & 0o777), "0o600")


if __name__ == "__main__":
    unittest.main()
