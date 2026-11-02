"""
qviz.agent.bridge -- the running GUI's agent tools, offered to a model OUTSIDE the app (Claude Code, through the MCP server
scripts/qchem_mcp.py), so an agent runs on the user's Claude subscription instead of API billing.

The GUI listens on 127.0.0.1 only, on a free port, and writes {port, token} to BRIDGE_FILE (mode 0600, removed on exit):
a caller must present the token, so another local user cannot drive the draft.  One JSON object per line each way:
    {"token": ..., "op": "list"}                         -> {"tools": [{name, description, input_schema}], "instructions": ...}
    {"token": ..., "op": "call", "name": ..., "args": {}} -> {"result": {...}}
    {"token": ..., "op": "hello", "client": ...}          -> {"ok": true}       (an MCP client attached: the dock says so)
Besides the Toolset's tools the bridge offers ONE of its own, tell_user: Claude Code's prose stays in Claude Code, so the
agent is asked to post its reply there too, and the Agent dock shows it beside the edits it made.
The tools run HERE, on the GUI thread (QTcpServer delivers its signals on the thread that owns it): the draft and the
workspace are GUI-thread state, exactly as for the in-app agent.

What leaves the box: Claude Code sends tool results to the Anthropic API, so the bridge is an off-box client under RunSpec
Q4 like AnthropicClient -- and a LICENSED (VASP) result is withheld, its place taken by a note saying so.
"""
from __future__ import annotations

import json
import os
import pathlib
import secrets
from typing import Callable, Optional

from PySide6 import QtCore, QtNetwork

from .session import INSTRUCTIONS, SYSTEM, TELL_USER
from .tools import Toolset

BRIDGE_FILE = pathlib.Path.home() / ".config" / "qchem-GUI" / "agent-bridge.json"




def withhold_licensed(result: dict) -> dict:
    """A tool result as it may leave the box: a licensed (VASP) one is replaced by a note."""
    if isinstance(result, dict) and result.get("licensed"):
        return {"withheld": True, "summary": "this result comes from a licensed code (VASP) and never leaves the machine; "
                                             "the user can read it in the GUI"}
    return result


class AgentBridge(QtCore.QObject):
    """The localhost door into a Toolset.  `on_call(name, args, result)` reports each call (the Agent dock logs it)."""
    def __init__(self, tools: Toolset, on_call: Optional[Callable[[str, dict, dict], None]] = None,
                 path: pathlib.Path = BRIDGE_FILE, on_say: Optional[Callable[[str], None]] = None,
                 on_hello: Optional[Callable[[str], None]] = None, parent=None):
        """`on_call(name, args, result)` each tool call, `on_say(text)` each tell_user, `on_hello(text)` a client attaching."""
        super().__init__(parent)
        self.tools, self._on_call, self.path = tools, on_call, path
        self._on_say, self._on_hello = on_say, on_hello
        self.token = secrets.token_hex(16)
        self._server = QtNetwork.QTcpServer(self)
        if not self._server.listen(QtNetwork.QHostAddress.LocalHost, 0):
            raise RuntimeError(f"agent bridge: cannot listen on 127.0.0.1: {self._server.errorString()}")
        self.port = self._server.serverPort()
        self._server.newConnection.connect(self._accept)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"port": self.port, "token": self.token, "pid": os.getpid()}, f)

    def close(self) -> None:
        self._server.close()
        try:
            if json.loads(self.path.read_text()).get("token") == self.token:   # only our own file: a newer GUI may own it
                self.path.unlink()
        except (OSError, ValueError):
            pass

    # -- the wire -----------------------------------------------------------------------------------------------------
    def _accept(self) -> None:
        while self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            buf = bytearray()
            sock.readyRead.connect(lambda s=sock, b=buf: self._read(s, b))
            sock.disconnected.connect(sock.deleteLater)

    def _read(self, sock, buf: bytearray) -> None:
        buf += bytes(sock.readAll())
        while b"\n" in buf:
            line, _, rest = bytes(buf).partition(b"\n")
            buf[:] = rest
            sock.write((json.dumps(self.handle(line), default=str) + "\n").encode())
            sock.flush()

    def handle(self, line: bytes) -> dict:
        """One request line -> its reply (never raises: an error is a reply)."""
        try:
            req = json.loads(line)
        except ValueError:
            return {"error": "not JSON"}
        if not secrets.compare_digest(str(req.get("token", "")), self.token):
            return {"error": "bad token: the GUI was restarted, or this is not its bridge"}
        if req.get("op") == "list":
            return {"tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                              for t in self.tools.schemas] + [TELL_USER], "instructions": SYSTEM + INSTRUCTIONS}
        if req.get("op") == "hello":
            if self._on_hello:
                self._on_hello(f"{req.get('client') or 'an MCP client'} is connected and can edit this draft")
            return {"ok": True}
        if req.get("op") == "call" and req.get("name") == TELL_USER["name"]:
            text = str((req.get("args") or {}).get("text", "")).strip()
            if self._on_say and text:
                self._on_say(text)
            return {"result": {"shown": bool(text), "summary": "shown in the Agent dock"}}
        if req.get("op") == "call":
            name, args = req.get("name", ""), req.get("args") or {}
            try:
                result = self.tools.call(name, args)
            except Exception as e:                   # noqa: BLE001 -- the model hears about it, as a tool error
                result = {"error": f"{type(e).__name__}: {e}"}
            if self._on_call:
                self._on_call(name, args, result)
            return {"result": withhold_licensed(result)}
        return {"error": f"unknown op {req.get('op')!r}"}
