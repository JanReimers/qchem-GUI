"""qviz.bse_dialog -- pick an orbital basis from the Basis Set Exchange for the draft's elements and download it (qviz.bse).

Lists the Exchange's orbital sets that cover every element of the structure (ECP sets left out: no ECP route), filtered by
a search box.  Download fetches and translates on the panel's serial library thread (`serial`), so the window stays
responsive and the library's import never overlaps a run; the finished set is announced through `downloaded`.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets

from . import bse


class BSEDialog(QtWidgets.QDialog):
    downloaded = QtCore.Signal(object)       # the saved set's sidecar (qviz.bse.download), on the GUI thread
    _done = QtCore.Signal(object)            # sidecar | the exception, from the serial thread

    def __init__(self, Zs: set[int], serial: Optional[Callable[[Callable[[], None]], None]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Basis Set Exchange")
        self.Zs = set(Zs)
        self._serial = serial or (lambda fn: fn())
        self._done.connect(self._on_done)
        lay = QtWidgets.QVBoxLayout(self)
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText("search: name or description (cc-pV, def2, 6-31 ...)")
        self.search.textChanged.connect(self._fill)
        lay.addWidget(self.search)
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._show)
        self.list.itemDoubleClicked.connect(lambda _i: self._download())
        lay.addWidget(self.list, 1)
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel(); self.status.setWordWrap(True)
        self.download_btn = QtWidgets.QPushButton("Download"); self.download_btn.clicked.connect(self._download)
        close = QtWidgets.QPushButton("Close"); close.clicked.connect(self.reject)
        row.addWidget(self.status, 1); row.addWidget(self.download_btn); row.addWidget(close)
        lay.addLayout(row)
        self._cands: list[dict] = []
        try:
            bse.metadata()
            self._fill()
        except bse.BSEError as e:
            self.status.setText(str(e)); self.download_btn.setEnabled(False)
        self.resize(560, 520)

    def _fill(self, *_):
        self._cands = bse.candidates(self.Zs, self.search.text())
        have = {(s["key"], s["version"]) for s in bse.installed()}
        self.list.clear()
        for c in self._cands:
            self.list.addItem(f"{c['name']}   (v{c['version']}, {c['family']})" + ("   ✓ downloaded" if (c["key"], c["version"]) in have else ""))
        self.status.setText(f"{len(self._cands)} orbital sets cover every element here (ECP sets are not offered: no ECP route yet)")
        self.download_btn.setEnabled(bool(self._cands))
        if self._cands:
            self.list.setCurrentRow(0)

    def _show(self, row: int):
        c = self._cands[row] if 0 <= row < len(self._cands) else None
        self.info.setText("" if c is None else f"{c['name']}: {c['description']}"
                          + ("\nspherical d/f by design; the molecular facade builds Cartesian ones" if c["spherical"] else ""))

    def _download(self):
        row = self.list.currentRow()
        if not 0 <= row < len(self._cands):
            return
        key = self._cands[row]["key"]
        self.download_btn.setEnabled(False)
        self.status.setText(f"downloading {self._cands[row]['name']} …")

        def job():
            try:
                got = bse.download(key)
            except Exception as e:                    # noqa: BLE001 -- the Exchange's or the library's words, shown
                got = e
            self._done.emit(got)
        self._serial(job)

    def _on_done(self, got):
        self.download_btn.setEnabled(True)
        if isinstance(got, Exception):
            self.status.setText(f"not downloaded: {got}"); return
        self.status.setText(f"saved {got['name']} v{got['version']} → {got['path']}")
        self._fill()
        self.downloaded.emit(got)
