"""
qviz.agent.session -- the agent loop: user message -> model turn -> tool calls -> model turn ... (doc/RunSpecPlan.md §5).

The loop is code-agnostic: it knows a ModelClient, a Toolset and a ConsentGate, nothing else.  EVERY model call passes
the consent gate first (a refused call ends the turn with the reason in the chat; nothing is sent).  A tool result
from a licensed back end is marked on its message, so the gate can keep it on the box.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .client import Message, ModelClient
from .consent import ConsentGate, ConsentRefused
from .tools import Toolset

SYSTEM = """You help a scientist set up electronic-structure runs in the SOLID² GUI.  You edit ONE shared draft spec through
your tools; the user sees every edit as a diff and can undo it.  You cannot start, cancel or delete a run: request_run puts
a card in the chat and the user decides.  State no number (an energy, a U, a cutoff) that a tool did not give you, and cite
the run id or tool it came from.  When a choice disables a back end, say so (can_run)."""

MAX_STEPS = 12          # model turns per user message: a loop that does not settle is stopped, and says so


@dataclass(frozen=True)
class Event:
    kind: str           # "user" | "assistant" | "tool_call" | "tool_result" | "refused" | "stopped"
    text: str
    data: object = None


class AgentSession:
    def __init__(self, client: ModelClient, tools: Toolset, gate: Optional[ConsentGate] = None, system: str = SYSTEM):
        self.client, self.tools, self.gate, self.system = client, tools, gate or ConsentGate(), system
        self.messages: list[Message] = []
        self._listeners: list[Callable[[Event], None]] = []

    def on_event(self, cb: Callable[[Event], None]) -> None:
        self._listeners.append(cb)

    def _emit(self, kind: str, text: str, data=None) -> Event:
        e = Event(kind, text, data)
        for cb in self._listeners:
            cb(e)
        return e

    def send(self, text: str) -> list[Event]:
        """One user message, run to the model's final prose (or a refusal / the step cap).  Returns the events emitted."""
        events = [self._emit("user", text)]
        self.messages.append(Message("user", text))
        for _ in range(MAX_STEPS):
            try:
                outbound = self.gate.outbound(self.client, self.messages)
            except ConsentRefused as e:
                events.append(self._emit("refused", str(e)))
                return events
            turn = self.client.respond(self.system, outbound, self.tools.schemas)
            self.messages.append(Message("assistant", turn.text, turn.tool_calls))
            if turn.text:
                events.append(self._emit("assistant", turn.text))
            if not turn.tool_calls:
                return events
            for call in turn.tool_calls:
                events.append(self._emit("tool_call", call.name, call))
                result = self.tools.call(call.name, call.args)
                self.messages.append(Message("tool", tool_call_id=call.id, content=result, licensed=bool(result.get("licensed"))))
                events.append(self._emit("tool_result", result.get("summary") or result.get("error", ""), result))
        events.append(self._emit("stopped", f"stopped after {MAX_STEPS} model turns without a final answer"))
        return events
