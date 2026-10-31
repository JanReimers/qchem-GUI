"""
qviz.agent.client -- the model client INTERFACE, and the scripted client the agent runs on until RunSpec Q4 is ruled.

A client turns (system prompt, conversation, tool schemas) into one assistant turn: prose and/or tool calls.  It
declares whether a call LEAVES THE BOX; the consent gate (qviz.agent.consent) reads that before every call.  The real
Anthropic client drops in behind this interface once the user answers Q4 (model, key, what may leave the machine);
until then there is none in the code -- no API call, no key, nothing sent anywhere.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class Message:
    role: str                                   # "user" | "assistant" | "tool"
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()       # assistant
    tool_call_id: str = ""                      # tool: the call this answers
    content: Any = None                         # tool: the result (JSON-able)
    licensed: bool = False                      # tool: derived from licensed (VASP) inputs/outputs -- never leaves the box


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: dict


@dataclass(frozen=True)
class ModelTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class ModelClient(Protocol):
    name: str
    leaves_box: bool                            # True = respond() sends the conversation off this machine

    def respond(self, system: str, messages: Sequence[Message], tools: Sequence[ToolSchema]) -> ModelTurn: ...


class ScriptedClient:
    """Replays a script: each entry is a ModelTurn, or a callable(messages) -> ModelTurn.  For tests."""
    name = "scripted"
    leaves_box = False

    def __init__(self, script: Sequence[ModelTurn | Callable[[Sequence[Message]], ModelTurn]]):
        self._script = list(script)
        self.seen: list[tuple[str, list[Message], list[str]]] = []   # what each call was handed (system, messages, tool names)

    def respond(self, system, messages, tools) -> ModelTurn:
        self.seen.append((system, list(messages), [t.name for t in tools]))
        if not self._script:
            return ModelTurn("(script exhausted)")
        step = self._script.pop(0)
        return step(messages) if callable(step) else step


class OfflinePlanner:
    """A small DETERMINISTIC stand-in for the model, so the agent panel is usable with no model at all: it maps a few
    phrasings onto tool calls and reports what the tools said.  It is not an assistant; it exists to exercise the tool
    layer, the draft and the run cards end to end.  Commands (one per message):

        molecule <name>            build a preset molecule into the draft (water, F2, ...)
        set <dotted.path>=<json>   patch the draft (e.g. set accuracy.epsilon_Ha_per_atom=1e-6)
        check                      validate the draft and list which back ends can run it
        run [on <backend>...]      put a run-request card in the chat
        undo                       revert the last draft edit
    """
    name = "offline planner"
    leaves_box = False

    def __init__(self):
        self._n = 0

    def _call(self, name: str, args: dict) -> ModelTurn:
        self._n += 1
        return ModelTurn("", (ToolCall(f"op{self._n}", name, args),))

    def respond(self, system, messages, tools) -> ModelTurn:
        last = messages[-1]
        if last.role == "tool":
            return ModelTurn(_say(messages))
        text = last.text.strip()
        if m := re.fullmatch(r"molecule\s+(.+)", text, re.I):
            return self._call("build_structure", {"kind": "molecule", "name": m.group(1).strip()})
        if m := re.fullmatch(r"set\s+([\w.]+)\s*=\s*(.+)", text, re.I):
            try:
                value = json.loads(m.group(2))
            except json.JSONDecodeError:
                value = m.group(2).strip()
            patch: dict = {}
            node = patch
            *head, tail = m.group(1).split(".")
            for k in head:
                node = node.setdefault(k, {})
            node[tail] = value
            return self._call("propose_spec_patch", {"patch": patch, "rationale": f"user asked: {text}"})
        if re.fullmatch(r"check", text, re.I):
            return self._call("can_run", {})
        if m := re.fullmatch(r"run(?:\s+on\s+(.+))?", text, re.I):
            names = [b.strip() for b in m.group(1).split(",")] if m.group(1) else []
            return self._call("request_run", {"backends": names, "purpose": "requested in the agent panel"})
        if re.fullmatch(r"undo", text, re.I):
            return self._call("undo_last_edit", {})
        return ModelTurn("I am the offline planner (no model is connected).  " + " ".join(OfflinePlanner.__doc__.split("Commands")[1].split()))


def _say(messages: Sequence[Message]) -> str:
    """Report the tool results since the last user message, verbatim enough that nothing is invented."""
    out = []
    for m in reversed(messages):
        if m.role == "user":
            break
        if m.role == "tool":
            c = m.content
            out.append(c.get("summary") or c.get("error") or json.dumps(c)[:400] if isinstance(c, dict) else str(c))
    return "\n".join(reversed(out))
