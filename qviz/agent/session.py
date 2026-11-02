"""
qviz.agent.session -- the agent loop: user message -> model turn -> tool calls -> model turn ... (doc/RunSpecPlan.md §5).

The loop is code-agnostic: it knows a ModelClient, a Toolset and a ConsentGate, nothing else.  EVERY model call passes
the consent gate first (a refused call ends the turn with the reason in the chat; nothing is sent).  A tool result
from a licensed back end is marked on its message, so the gate can keep it on the box.

`start` runs the loop with each MODEL CALL handed to a caller-supplied runner (the panel runs it off the GUI thread: a
Claude call takes seconds) while every TOOL runs where `start` was called (the draft and workspace are GUI-thread state).
`send` is the same loop run synchronously.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .client import Message, ModelClient, ModelTurn
from .consent import ConsentGate, ConsentRefused
from .tools import Toolset

SYSTEM = """You help a scientist set up electronic-structure runs in the SOLID² GUI.  You edit ONE shared draft spec through
your tools; the user sees every edit as a diff and can undo it.  You cannot start, cancel or delete a run: request_run puts
a card in the chat and the user decides.  State no number (an energy, a U, a cutoff) that a tool did not give you, and cite
the run id or tool it came from.  The one exception is a molecule's starting geometry when no preset has it: you may give
build_structure atoms from your own knowledge, and you then say plainly that the geometry is approximate.  When a choice
disables a back end, say so (can_run).  Read the draft and the resolved settings with your tools before changing them, and
keep replies short: the user sees each edit as a diff."""

#: for an agent driving the GUI from OUTSIDE (Claude Code through scripts/qchem_mcp.py -> qviz.agent.bridge): its prose
#: stays in its own window, so it is given one extra tool to post its reply into the Agent dock
TELL_USER = {"name": "tell_user",
             "description": "Show a message to the user in the GUI's Agent dock, next to the edits you made.  Use it for your "
                            "answer at the end of each request (a short summary: what you changed, what to check, what the "
                            "user should click), and whenever you need the user to look at something in the GUI.",
             "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
                              "additionalProperties": False}}
INSTRUCTIONS = ("  You are driving the GUI from outside it: the user may be looking at the GUI rather than at you, so end "
                "each request by posting your reply with tell_user.")

MAX_STEPS = 12
#: runs one model call: runner(call, done) runs call() (anywhere) and then done(its ModelTurn, or the exception it raised)
#: on the thread that owns the session
Runner = Callable[[Callable[[], ModelTurn], Callable[[object], None]], None]


def _inline(call, done) -> None:
    try:
        got = call()
    except Exception as e:                       # noqa: BLE001 -- reported in the chat, never raised into the UI
        got = e
    done(got)          # model turns per user message: a loop that does not settle is stopped, and says so


@dataclass(frozen=True)
class Event:
    kind: str           # "user" | "assistant" | "tool_call" | "tool_result" | "refused" | "stopped" | "error" | "idle"
    text: str
    data: object = None


class AgentSession:
    def __init__(self, client: ModelClient, tools: Toolset, gate: Optional[ConsentGate] = None, system: str = SYSTEM):
        self.client, self.tools, self.gate, self.system = client, tools, gate or ConsentGate(), system
        self.messages: list[Message] = []
        self._listeners: list[Callable[[Event], None]] = []
        self._busy, self._steps = False, 0

    def on_event(self, cb: Callable[[Event], None]) -> None:
        self._listeners.append(cb)

    def _emit(self, kind: str, text: str, data=None) -> Event:
        e = Event(kind, text, data)
        for cb in self._listeners:
            cb(e)
        return e

    @property
    def busy(self) -> bool:
        """A user message is still being answered (a model call is in flight or tools are running)."""
        return self._busy

    def send(self, text: str) -> list[Event]:
        """One user message, run to the model's final prose (or a refusal / the step cap), synchronously.  Returns the events."""
        events: list[Event] = []
        self._listeners.append(events.append)
        try:
            self.start(text, _inline)
        finally:
            self._listeners.remove(events.append)
        return [e for e in events if e.kind != "idle"]     # "idle" is the panel's cue, not part of the answer

    def start(self, text: str, runner: Runner) -> None:
        """One user message; each model call goes through `runner`.  Ends with an "idle" event."""
        if self._busy:
            raise RuntimeError("the agent is still answering the previous message")
        self._busy, self._steps = True, 0
        self._emit("user", text)
        self.messages.append(Message("user", text))
        self._next(runner)

    def _finish(self, kind: str = "", text: str = "") -> None:
        if kind:
            self._emit(kind, text)
        self._busy = False
        self._emit("idle", "")

    def _next(self, runner: Runner) -> None:
        if self._steps >= MAX_STEPS:
            self._finish("stopped", f"stopped after {MAX_STEPS} model turns without a final answer")
            return
        self._steps += 1
        try:
            outbound = list(self.gate.outbound(self.client, self.messages))
        except ConsentRefused as e:
            self._finish("refused", str(e))
            return
        runner(lambda: self.client.respond(self.system, outbound, self.tools.schemas), lambda got: self._take(got, runner))

    def _take(self, got, runner: Runner) -> None:
        if isinstance(got, Exception):
            self._finish("error", f"the model call failed: {got}")
            return
        turn: ModelTurn = got
        self.messages.append(Message("assistant", turn.text, turn.tool_calls, raw=turn.raw))
        if turn.text:
            self._emit("assistant", turn.text)
        if not turn.tool_calls:
            self._finish()
            return
        for call in turn.tool_calls:
            self._emit("tool_call", call.name, call)
            try:
                result = self.tools.call(call.name, call.args)
            except Exception as e:                   # noqa: BLE001 -- the model hears about it, as a tool error
                result = {"error": f"{type(e).__name__}: {e}"}
            self.messages.append(Message("tool", tool_call_id=call.id, content=result, licensed=bool(result.get("licensed"))))
            self._emit("tool_result", result.get("summary") or result.get("error", ""), result)
        self._next(runner)
