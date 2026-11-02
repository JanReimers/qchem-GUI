#!/usr/bin/env python3
"""
qchem_mcp -- an MCP server (stdio) that hands the RUNNING qchem GUI's agent tools to Claude Code.

Claude Code launches this process and speaks MCP (JSON-RPC 2.0, one message per line) on its stdin/stdout.  Every tool call
is forwarded to the GUI's agent bridge (qviz.agent.bridge: 127.0.0.1, the port and token in ~/.config/qchem-GUI/
agent-bridge.json), where it runs against the GUI's own draft and workspace -- the edits appear in the Setup panel, undoable.

Register once (from any directory):
    claude mcp add --scope user qchem-gui -- /home/janr/Code/qchem-GUI/.venv/bin/python /home/janr/Code/qchem-GUI/scripts/qchem_mcp.py

The tool list comes from the GUI when it is running, else from the same Toolset built here (so Claude Code sees the tools
even when the GUI starts later; a call then says to start the GUI).  Nothing but stdout carries the protocol: diagnostics
go to stderr.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BRIDGE_FILE = pathlib.Path(os.environ.get("QCHEM_BRIDGE_FILE") or
                           pathlib.Path.home() / ".config" / "qchem-GUI" / "agent-bridge.json")   # override: tests
SERVER = {"name": "qchem-gui", "version": "1.0"}


def bridge(req: dict, timeout: float = 120.0) -> dict:
    """One request to the GUI's bridge; raises ConnectionError (with a reason a person can act on) when it is not there."""
    try:
        info = json.loads(BRIDGE_FILE.read_text())
    except (OSError, ValueError):
        raise ConnectionError("the qchem GUI is not running (no agent bridge file): start it with "
                              "`.venv/bin/python app_desktop.py` in ~/Code/qchem-GUI")
    try:
        with socket.create_connection(("127.0.0.1", int(info["port"])), timeout=timeout) as s:
            s.sendall((json.dumps({**req, "token": info["token"]}) + "\n").encode())
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
    except OSError as e:
        raise ConnectionError(f"the qchem GUI's agent bridge did not answer ({e}): is the GUI still running?")
    return json.loads(data)


def static_tools() -> tuple[list[dict], str]:
    """The tool list and instructions without a running GUI: the same Toolset, over an empty draft."""
    from qviz.agent.session import SYSTEM
    from qviz.agent.tools import Toolset
    from qviz.backends.base import BackendRegistry
    from qviz.draft import SpecDraft
    t = Toolset(SpecDraft(), BackendRegistry([]))
    return [{"name": s.name, "description": s.description, "input_schema": s.input_schema} for s in t.schemas], SYSTEM


def tools_and_instructions() -> tuple[list[dict], str]:
    try:
        got = bridge({"op": "list"}, timeout=5.0)
        if "tools" in got:
            return got["tools"], got.get("instructions", "")
    except ConnectionError:
        pass
    return static_tools()


INSTRUCTIONS_HEAD = ("These tools drive the qchem GUI the user has open: its ONE shared draft run spec (the Setup panel shows "
                     "every edit as a diff and can undo it) and its workspace of runs.  ")


def handle(msg: dict) -> dict | None:
    """One JSON-RPC message -> its response (None for a notification)."""
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None                                              # notifications/initialized, cancelled, ...
    if method == "initialize":
        _tools, instructions = tools_and_instructions()
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                  "capabilities": {"tools": {}}, "serverInfo": SERVER,
                  "instructions": INSTRUCTIONS_HEAD + instructions}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        tools, _ = tools_and_instructions()
        result = {"tools": [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]}
                            for t in tools]}
    elif method == "tools/call":
        p = msg.get("params", {})
        try:
            got = bridge({"op": "call", "name": p.get("name", ""), "args": p.get("arguments") or {}})
            body = got.get("result", got)
            failed = "error" in got or (isinstance(body, dict) and "error" in body)
        except ConnectionError as e:
            body, failed = {"error": str(e)}, True
        result = {"content": [{"type": "text", "text": json.dumps(body, default=str)}], "isError": failed}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            print(f"qchem_mcp: not JSON: {line[:200]}", file=sys.stderr)
            continue
        try:
            reply = handle(msg)
        except Exception as e:                                   # noqa: BLE001 -- a protocol error, never a crash
            reply = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32603, "message": str(e)}}
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
