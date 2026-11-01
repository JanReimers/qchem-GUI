"""
qviz.agent.panel -- the Agent dock: chat and run-request cards (doc/RunSpecPlan.md §5.1).

The shared draft the agent edits is shown and edited in the Setup panel (qviz.setup_panel: its tabs and its Spec JSON),
one SpecDraft and one undo stack; this panel reports the agent's draft edits in the chat.  A card's Run button is the only thing here that starts a run (it calls the app's start hook).  The header says which
client is connected and that nothing leaves the machine while Q4 is unanswered.
"""
from __future__ import annotations

import html

from PySide6 import QtCore, QtGui, QtWidgets

from ..draft import Edit, SpecDraft
from .session import AgentSession, Event
from .tools import RunRequest

_STYLE = {"user": "#1f6feb", "assistant": "#2da44e", "tool_call": "#8b949e", "tool_result": "#8b949e",
          "refused": "#cf222e", "stopped": "#cf222e"}


class AgentPanel(QtWidgets.QWidget):
    def __init__(self, session: AgentSession, draft: SpecDraft, parent=None):
        super().__init__(parent)
        self.session, self.draft = session, draft
        lay = QtWidgets.QVBoxLayout(self)
        gate = session.gate.policy
        head = QtWidgets.QLabel(f"<b>{html.escape(session.client.name)}</b> · "
                                + ("runs on this machine" if not session.client.leaves_box else "off-box client")
                                + ("" if gate.granted else " · nothing leaves this machine (RunSpec Q4 unanswered)"))
        head.setWordWrap(True)
        lay.addWidget(head)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.chat = QtWidgets.QTextBrowser()
        split.addWidget(self.chat)
        self._cards_box = QtWidgets.QWidget()
        self._cards = QtWidgets.QVBoxLayout(self._cards_box); self._cards.setContentsMargins(0, 0, 0, 0)
        split.addWidget(self._cards_box)
        lay.addWidget(split, 1)

        self.entry = QtWidgets.QLineEdit(); self.entry.setPlaceholderText("Ask the agent…  (offline planner: molecule water, set …, check, run)")
        self.entry.returnPressed.connect(self._send)
        lay.addWidget(self.entry)

        session.on_event(self._on_event)
        session.tools.on_card(self._on_card)
        draft.on_change(self._on_draft)
        self._on_draft(draft, None)

    # -- chat -----------------------------------------------------------------
    def _send(self):
        text = self.entry.text().strip()
        if text:
            self.entry.clear()
            self.session.send(text)

    def _on_event(self, e: Event):
        who = {"user": "you", "assistant": "agent", "tool_call": "→", "tool_result": "←"}.get(e.kind, e.kind)
        body = html.escape(e.text).replace("\n", "<br>")
        self.chat.append(f'<span style="color:{_STYLE[e.kind]}"><b>{who}</b></span> {body}')

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
