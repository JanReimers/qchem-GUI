"""
qviz.agent.consent -- what may leave the box, decided in ONE place on the call path (doc/RunSpecPlan.md §5.3, Q4).

Every model call goes through ConsentGate.outbound before the client sees the conversation:
  * a client that does not leave the box (scripted, offline) passes untouched;
  * a client that does needs an OutboundPolicy the USER granted.  The default is Q4_UNANSWERED, which refuses every
    off-box call: until the user rules on Q4 nothing is sent, whatever client is plugged in;
  * even with a policy granted, a conversation holding a LICENSED tool result (VASP inputs/outputs) is refused --
    the licence forbids redistribution, and a model provider is a third party;
  * a policy may also exclude results (`send_results=False`: structures and specs go, energies and fields do not).

Consent for an external database (COD, Materials Project; Q5) is `ask`, a callback the UI answers (to be cached per
database for the app run, preferably persisted -- §10 Q5; not built yet, so today it asks every time); the default
answers no.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .client import Message, ModelClient


class ConsentRefused(PermissionError):
    """A call that would send something the user has not agreed to send."""


@dataclass(frozen=True)
class OutboundPolicy:
    granted: bool
    send_results: bool = False                 # tool results carrying run outputs (energies, fields) may be sent
    note: str = ""


#: RunSpec Q4 as the user ruled it (2026-11-01): the Anthropic API, key from ANTHROPIC_API_KEY only, and specs, structures
#: AND results may be sent (licensed VASP results still never are: that check is not a policy, it is the licence)
Q4_RULED = OutboundPolicy(True, send_results=True, note="RunSpec Q4 (ruled 2026-11-01): specs, structures and results go "
                                                        "to the Anthropic API; licensed (VASP) results never leave the box")

Q4_UNANSWERED = OutboundPolicy(False, note="RunSpec Q4 (the agent's model, key and what may leave the box) is the user's "
                                          "decision and is unanswered: nothing is sent off this machine")


class ConsentGate:
    def __init__(self, policy: OutboundPolicy = Q4_UNANSWERED, ask: Optional[Callable[[str], bool]] = None):
        self.policy = policy
        self._ask = ask or (lambda _q: False)

    def outbound(self, client: ModelClient, messages: Sequence[Message]) -> Sequence[Message]:
        """The conversation `client` may be handed, or ConsentRefused naming why not."""
        if not client.leaves_box:
            return messages
        if not self.policy.granted:
            raise ConsentRefused(f"{client.name} would send the conversation off this machine: {self.policy.note}")
        if any(m.licensed for m in messages):
            raise ConsentRefused(f"{client.name}: the conversation holds a licensed (VASP) result, which never leaves the box")
        if not self.policy.send_results and any(m.role == "tool" and isinstance(m.content, dict) and m.content.get("is_result")
                                                for m in messages):
            raise ConsentRefused(f"{client.name}: the conversation holds run results, and the policy sends specs and structures only")
        return messages

    def external_query(self, what: str) -> bool:
        """Per-query consent for an external service (Q5).  False unless the user says yes to THIS query."""
        return bool(self._ask(what))
