"""
qviz.scfview -- the live SCF view: one plot TAB per column of the code's iteration table (`tabs`), and the table itself
(`table`) -- two widgets the Output panel shows as its "SCF graphs" and "Iteration log" tabs, fed by one object.

The columns are whatever the code prints, under the code's own titles (qviz.data.SCFCell): SOLID²'s console headers,
CP2K's `Convergence` / `Change`, pw.x's `estimated scf accuracy`, ABINIT's `residm`.  Nothing here knows which code
ran.  A column with a number gets a tab, created the first time a number arrives; a text-only column (CP2K's
`Update method`, SOLID²'s `XC ρ`) lives in the table only.  A convergence measure (scale "log") is drawn as |value| on
a log axis; everything else (energies, moments, times) on a linear one.  Only the selected tab is visible, and the
selection survives a new run when the new run has a column of that title.

A column the code GATES on (SCFCell.tolerance: SOLID²'s [F,D], Δρ, ΔE/E when set) shows its gate under the header, and a
value that meets it (|value| < gate, the iterator's own test) turns green: the log reads off which gate is still holding
the run.
"""
from __future__ import annotations

import math

import matplotlib.ft2font  # noqa: F401 -- BEFORE Qt draws text: see export_column
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .data import SCFCell, SCFRow


def _axis_label(c: SCFCell) -> str:
    return f"|{c.title}|" if c.scale == "log" else c.title


def column_xy(rows: list[SCFRow], title: str, log: bool) -> tuple[list[int], list[float]]:
    """The (iteration, value) points of one column; a log column drops the points a log axis cannot draw."""
    pts = [(r.iteration, abs(v) if log else v) for r in rows if not math.isnan(v := r.value(title))]
    if log:
        pts = [(i, v) for i, v in pts if v > 0]
    return [i for i, _ in pts], [v for _, v in pts]


PASS_GREEN = "#3fb950"


def meets(c: SCFCell) -> bool:
    """Does this cell's value pass its column's gate?  The iterator's own test, |value| < gate (an ungated column never does)."""
    return c.tolerance is not None and c.tolerance > 0 and not math.isnan(c.value) and abs(c.value) < c.tolerance


class SCFView(QtCore.QObject):
    """The plot tabs (`tabs`) and the iteration table (`table`).  add_row() is the live feed; set_rows() shows a finished trace."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QtWidgets.QTabWidget(); self.tabs.setDocumentMode(True)
        self.table = QtWidgets.QTableWidget(0, 1)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setFont(QtGui.QFont("monospace"))
        self.table.verticalHeader().setVisible(False)
        self.rows: list[SCFRow] = []
        self._cells: dict[str, SCFCell] = {}      # first cell seen per title: its scale and unit
        self._titles: list[str] = []               # the table's columns after "#", by title (a header may carry its gate too)
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._wanted: str | None = None           # the tab the user last picked: kept across runs
        self._auto = False                         # this run's tab was chosen for the user (a convergence measure)
        self.tabs.currentChanged.connect(self._picked)
        self.clear()

    # -- feed ----------------------------------------------------------------
    def clear(self) -> None:
        self.rows.clear(); self._cells.clear(); self._curves.clear(); self._titles.clear(); self._auto = False
        self.tabs.blockSignals(True); self.tabs.clear(); self.tabs.blockSignals(False)
        self.table.setRowCount(0); self.table.setColumnCount(1); self.table.setHorizontalHeaderLabels(["#"])

    def set_rows(self, rows: list[SCFRow]) -> None:
        self.clear()
        for r in rows:
            self.add_row(r)

    def add_row(self, row: SCFRow) -> None:
        self.rows.append(row)
        for c in row.cells:
            if c.title not in self._cells:
                self._cells[c.title] = c
                self._add_column(c)
            if c.title not in self._curves and not math.isnan(c.value):
                self._add_tab(self._cells[c.title])
        for title, curve in self._curves.items():
            curve.setData(*column_xy(self.rows, title, self._cells[title].scale == "log"))
        self._append_table_row(row)

    @property
    def current_title(self) -> str | None:
        return self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else None

    # -- internals -----------------------------------------------------------
    def _add_column(self, c: SCFCell) -> None:
        n = self.table.columnCount()
        self.table.insertColumn(n)
        self._titles.append(c.title)
        gated = c.tolerance is not None and c.tolerance > 0
        item = QtWidgets.QTableWidgetItem(f"{c.title}\n(< {c.tolerance:g})" if gated else c.title)
        if gated:
            item.setToolTip(f"the run converges on this column once |{c.title}| < {c.tolerance:g}: green rows meet it")
        self.table.setHorizontalHeaderItem(n, item)

    def _add_tab(self, c: SCFCell) -> None:
        log = c.scale == "log"
        w = pg.PlotWidget()
        w.setLogMode(y=log); w.showGrid(x=True, y=True, alpha=0.2)
        w.setLabel("bottom", "iteration")
        w.setLabel("left", _axis_label(c) + (f" ({c.unit})" if c.unit else ""))   # no units=: pyqtgraph would SI-rescale
        w.getAxis("left").enableAutoSIPrefix(False)
        if c.tolerance is not None and c.tolerance > 0:  # the gate the code holds this column to
            w.addItem(pg.InfiniteLine(pos=math.log10(c.tolerance) if log else c.tolerance, angle=0,
                                      pen=pg.mkPen("#888", style=QtCore.Qt.DashLine)))
        self._curves[c.title] = w.plot(pen="#4dd0e1", symbol="o", symbolSize=5)
        self.tabs.blockSignals(True)                 # the first addTab selects itself: not the user's pick
        i = self.tabs.addTab(w, c.title)
        if c.title == self._wanted:
            self.tabs.setCurrentIndex(i); self._auto = True
        elif not self._auto and (self._wanted not in self._cells) and log:   # no pick to keep: the first convergence measure
            self.tabs.setCurrentIndex(i); self._auto = True
        self.tabs.blockSignals(False)

    def _picked(self, i: int) -> None:
        if i >= 0:
            self._wanted = self.tabs.tabText(i)

    def _append_table_row(self, row: SCFRow) -> None:
        n = self.table.rowCount()
        self.table.insertRow(n)
        self.table.setItem(n, 0, QtWidgets.QTableWidgetItem(str(row.iteration)))
        cells = {c.title: c for c in row.cells}
        for k, title in enumerate(self._titles, start=1):
            c = cells.get(title)
            item = QtWidgets.QTableWidgetItem(c.shown() if c else "")
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if c is not None and meets(c):
                item.setForeground(QtGui.QColor(PASS_GREEN))
            self.table.setItem(n, k, item)
        self.table.resizeColumnsToContents()
        self.table.scrollToBottom()


def export_column(path: str, rows: list[SCFRow], title: str, heading: str) -> None:
    """One column of a finished trace as a publication figure (format from the extension).

    matplotlib's ft2font carries its own FreeType and exports its FT_ symbols; loaded AFTER Qt has drawn text through
    the system libfreetype, its glyph rendering fails ("FT_Render_Glyph ... raster overflow", measured 2026-10-31 in the
    offscreen test run).  Hence the module-level `import matplotlib.ft2font` ahead of pyqtgraph."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure          # not pyplot: its global backend is Qt's once a QApplication exists
    c = next(c for r in rows for c in r.cells if c.title == title)
    log = c.scale == "log"
    fig = Figure(figsize=(6.4, 4.2)); FigureCanvasAgg(fig); ax = fig.add_subplot()
    (ax.semilogy if log else ax.plot)(*column_xy(rows, title, log), "o-")
    ax.set_xlabel("SCF iteration"); ax.set_ylabel(_axis_label(c) + (f" ({c.unit})" if c.unit else ""))
    ax.set_title(heading); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(path)
