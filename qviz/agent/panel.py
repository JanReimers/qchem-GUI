"""
qviz.agent.panel -- the Agent dock: chat and run-request cards (doc/RunSpecPlan.md §5.1).

The shared draft the agent edits is shown and edited in the Setup panel (qviz.setup_panel: its tabs and its Spec JSON),
one SpecDraft and one undo stack; this panel reports the agent's draft edits in the chat.  A card's Run button is the only thing here that starts a run (it calls the app's start hook).  The header says which
client is connected and that nothing leaves the machine while Q4 is unanswered.
"""
from __future__ import annotations

import html
import json

from PySide6 import QtCore, QtGui, QtWidgets

from typing import Callable

from ..draft import Edit, SpecDraft
from .session import AgentSession, Event
from .tools import RunRequest

_STYLE = {"user": "#1f6feb", "assistant": "#2da44e", "tool_call": "#8b949e", "tool_result": "#8b949e",
          "refused": "#cf222e", "stopped": "#cf222e", "error": "#cf222e"}


class _Done(QtCore.QObject):
    """Brings a finished model call back to the GUI thread (queued across threads)."""
    done = QtCore.Signal(object, object)          # (the continuation, the ModelTurn or the exception)


class _Call(QtCore.QRunnable):
    def __init__(self, call, done, bridge):
        super().__init__(); self._call, self._done, self._bridge = call, done, bridge

    def run(self):
        try:
            got = self._call()
        except Exception as e:                        # noqa: BLE001 -- the session reports it in the chat
            got = e
        self._bridge.done.emit(self._done, got)


class AgentPanel(QtWidgets.QWidget):
    def __init__(self, session: AgentSession, draft: SpecDraft, parent=None):
        super().__init__(parent)
        self.session, self.draft = session, draft
        lay = QtWidgets.QVBoxLayout(self)
        gate = session.gate.policy
        off = session.client.leaves_box
        head = QtWidgets.QLabel(f"<b>{html.escape(session.client.name)}</b> · "
                                + ("runs on this machine" if not off else
                                   "sends the conversation (specs, structures, results) to the Anthropic API"
                                   if gate.granted else "would send off the machine, but nothing is sent: Q4 unanswered"))
        head.setWordWrap(True)
        lay.addWidget(head)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.chat = QtWidgets.QTextBrowser()
        split.addWidget(self.chat)
        self._cards_box = QtWidgets.QWidget()
        self._cards = QtWidgets.QVBoxLayout(self._cards_box); self._cards.setContentsMargins(0, 0, 0, 0)
        split.addWidget(self._cards_box)
        lay.addWidget(split, 1)

        self.entry = QtWidgets.QLineEdit()
        self.entry.setPlaceholderText("Ask the agent…" if off else "Ask the agent…  (offline: molecule water, set …, check, run)")
        self._bridge = _Done(); self._bridge.done.connect(lambda k, got: k(got))
        self._pool = QtCore.QThreadPool(); self._pool.setMaxThreadCount(1)   # model calls only: never the library
        self._runner: Callable = ((lambda call, done: self._pool.start(_Call(call, done, self._bridge))) if off else
                                  (lambda call, done: done(call())))
        self.entry.returnPressed.connect(self._send)
        lay.addWidget(self.entry)

        session.on_event(self._on_event)
        session.tools.on_card(self._on_card)
        draft.on_change(self._on_draft)
        self._on_draft(draft, None)

    # -- chat -----------------------------------------------------------------
    def _send(self):
        text = self.entry.text().strip()
        if text and not self.session.busy:
            self.entry.clear()
            self.entry.setEnabled(False)                      # one message at a time; "idle" re-enables
            self.session.start(text, self._runner)

    def _on_event(self, e: Event):
        if e.kind == "idle":
            self.entry.setEnabled(True); self.entry.setFocus()
            return
        if e.kind == "user" and self.session.client.leaves_box:
            QtCore.QTimer.singleShot(0, lambda: self.chat.append('<span style="color:#8b949e"><i>thinking…</i></span>'))
        who = {"user": "you", "assistant": "agent", "tool_call": "→", "tool_result": "←"}.get(e.kind, e.kind)
        body = html.escape(e.text).replace("\n", "<br>")
        self.chat.append(f'<span style="color:{_STYLE[e.kind]}"><b>{who}</b></span> {body}')

    # -- Claude Code, through the MCP bridge --------------------------------------
    def note_bridge(self, text: str) -> None:
        self.chat.append(f'<span style="color:#8b949e"><i>{html.escape(text)}</i></span>')

    def note_said(self, text: str) -> None:
        """Claude Code's reply, posted through the bridge's tell_user."""
        body = html.escape(text).replace("\n", "<br>")
        self.chat.append(f'<span style="color:#8250df"><b>Claude Code</b></span> {body}')

    def note_external(self, name: str, args: dict, result: dict) -> None:
        """A tool call Claude Code made through the bridge: logged here like the in-app agent's."""
        brief = ", ".join(f"{k}={json.dumps(v)[:60]}" for k, v in args.items())
        said = result.get("summary") or result.get("error") or ""
        self.chat.append(f'<span style="color:#8250df"><b>Claude Code →</b></span> {html.escape(name)}'
                         f'<span style="color:#8b949e">({html.escape(brief)})</span>'
                         + (f'<br><span style="color:#8b949e">← {html.escape(str(said))}</span>' if said else ""))

    # -- cards ----------------------------------------------------------------
    def _on_card(self, card: RunRequest):
        box = QtWidgets.QGroupBox(f"Run request #{card.id}")
        bl = QtWidgets.QVBoxLayout(box)
        bl.addWidget(QtWidgets.QLabel(f"{html.escape(card.spec.summary)}<br>on {', '.join(card.backends)} — {html.escape(card.purpose)}"))
        row = QtWidgets.QHBoxLayout()
        run_b, no_b = QtWidgets.QPushButton("Run"), QtWidgets.QPushButton("Decline")

        def settle(accepted: bool):
            card.accept() if accepted else card.decline()
            run_b.setEnabled(False); no_b.setEnabled(False)
            box.setTitle(f"Run request #{card.id} — {card.state}")
        run_b.clicked.connect(lambda: settle(True)); no_b.clicked.connect(lambda: settle(False))
        row.addWidget(run_b); row.addWidget(no_b); row.addStretch(1)
        bl.addLayout(row)
        self._cards.addWidget(box)

    # -- draft ----------------------------------------------------------------
    def _on_draft(self, draft: SpecDraft, edit: Edit | None):
        if edit is not None and edit.author == "agent":
            lines = "<br>".join(html.escape(f"{p}: {o!r} → {n!r}") for p, o, n in edit.diff()[:20])
            self.chat.append(f'<span style="color:#9a6700"><b>draft edit</b></span> {html.escape(edit.rationale)}<br><tt>{lines}</tt>')
