"""
qviz.agent.client -- the model client INTERFACE; the Anthropic client (RunSpec Q4, ruled 2026-11-01); and the scripted
and offline clients for tests and for running with no key.

A client turns (system prompt, conversation, tool schemas) into one assistant turn: prose and/or tool calls.  It
declares whether a call LEAVES THE BOX; the consent gate (qviz.agent.consent) reads that before every call.

AnthropicClient is the one that leaves the box: Claude through the Anthropic API, the key from ANTHROPIC_API_KEY ONLY
(Q4b), sending specs, structures and results (Q4c) -- never a licensed (VASP) result, which the gate stops.  One call per
model turn: the agent session owns the tool loop, so every call passes the gate and every tool runs on the GUI thread.
"""
from __future__ import annotations

import json
import os
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
    raw: Any = None                             # assistant: the provider's own content, handed back to it VERBATIM next call


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: dict


@dataclass(frozen=True)
class ModelTurn:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = None                             # the provider's own content blocks (Claude's thinking must go back unchanged)


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
        return ModelTurn(_offline_help())


MODEL = "claude-opus-5-5"
KEY_ENV = "ANTHROPIC_API_KEY"                   # Q4b: the key comes from this environment variable and nowhere else


class AnthropicClient:
    """Claude through the Anthropic API: one Messages call per model turn.  The conversation goes up as the API's own
    shapes -- an assistant turn as the content Claude sent (its thinking blocks verbatim, as Opus 5.5 requires), the tool
    results that answer it as ONE user message.  The prompt prefix (tools, system, conversation so far) is cached, so a
    long chat re-sends cheaply.  Refusal fallback is on: a request Claude's safeguards decline is retried server-side on
    another model (the API's "default" routing) instead of ending the turn."""
    leaves_box = True

    def __init__(self, model: str = MODEL, effort: str = "medium", sdk=None):
        """`sdk`: an anthropic.Anthropic-shaped client (tests); None builds one from ANTHROPIC_API_KEY."""
        if sdk is None:
            key = os.environ.get(KEY_ENV)
            if not key:
                raise RuntimeError(f"{KEY_ENV} is not set: the agent's key comes from that environment variable only")
            import anthropic
            sdk = anthropic.Anthropic(api_key=key)
        self._sdk, self.model, self.effort = sdk, model, effort
        self.name = f"Claude ({model})"

    @staticmethod
    def why_absent() -> Optional[str]:
        """Why no AnthropicClient can be made here (None: it can)."""
        if not os.environ.get(KEY_ENV):
            return f"{KEY_ENV} is not set"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return "the anthropic package is not installed (pip install -r requirements.txt)"
        return None

    def respond(self, system, messages, tools) -> ModelTurn:
        r = self._sdk.beta.messages.create(
            model=self.model, max_tokens=16000, system=system, messages=_to_api(messages),
            tools=[{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools],
            output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        if r.stop_reason == "refusal":
            why = getattr(r.stop_details, "explanation", None) if r.stop_details else None
            return ModelTurn(f"Claude declined this request{': ' + why if why else ''}.")
        text = "\n".join(b.text for b in r.content if b.type == "text").strip()
        calls = tuple(ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use")
        if r.stop_reason == "max_tokens":
            text = (text + "\n" if text else "") + "(the reply hit its length limit and was cut off)"
            calls = ()                                           # a truncated tool call's input is not trustworthy
        return ModelTurn(text, calls, raw=[b.model_dump(exclude_none=True) for b in r.content])


def _to_api(messages: Sequence[Message]) -> list[dict]:
    """The session's conversation as Messages-API turns: consecutive tool results fold into ONE user message."""
    out: list[dict] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            content = m.raw if m.raw is not None else (
                ([{"type": "text", "text": m.text}] if m.text else [])
                + [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args} for c in m.tool_calls])
            out.append({"role": "assistant", "content": content})
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": json.dumps(m.content, default=str),
                     **({"is_error": True} if isinstance(m.content, dict) and "error" in m.content else {})}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def _offline_help() -> str:
    """What the offline planner answers to anything it does not understand: plain words, one command per line."""
    from ..molecules import MOLECULES
    names = ", ".join(n.split(" (")[0].lower() for n in MOLECULES)
    return ("No language model is connected, so I cannot read requests written in plain English.  To connect Claude, set "
            f"{KEY_ENV} in the environment and restart the app.\n"
            "Until one is, I understand only these short commands, one per message:\n\n"
            "  molecule water          load a preset molecule (" + names + ")\n"
            "  set accuracy.scf.accelerator=\"Ladder\"   change one setting of the draft\n"
            "  check                   say which codes can run the draft\n"
            "  run                     ask to run the draft\n"
            "  undo                    take back the last change\n\n"
            "Most settings are quicker to change in the Setup panel.")


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
