"""qviz.basis_shells -- the crystal basis's SHELL WHAT-IF table (the user, 2026-11-11): which shell to switch off when the basis is
near-singular, judged by what switching it off DOES.

The library answers (qchem_py.BasisShells, deck::MakeBasisShells): it resolves the deck's setup as Build does and builds its UNTRIMMED
basis once -- the analytic overlap S of every k-block, no grids, no SCF.  Each query is a library run, `basisTrimAnalysis`, and the
GUI shows ITS RECORD (never a parallel computation): the verdict on the vet stage's own metric (healthy = no AO drops at the floor),
a row per k, and a row per shell with what toggling it ALONE would do (`lambdaMinToggled`, `condToggled`, `aoDropsToggled`) and its
share of the near-null space at the worst k (`participation`); beside them the deck's trim (`deckKept`) and the vet stage's own pick
at the floor (`autoTrimKept`).  A shell is a basis fact (element, l, exponents; pin 22): it goes from every site of its element and
every k.

The result is a STATED trim (deck basis.trim, one {Z, l, alpha} per removed shell), so the run is reproducible and the choice is
visible in the spec.  A stated trim names ONE exponent, so a contracted shell cannot be switched off here (the shipped periodic
valence sets are single-primitive).
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .backends import solid2_deck
from .spec import RunSpec

L_NAMES = "spdfghi"
_SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc "
            "Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po "
            "At Rn Fr Ra Ac Th Pa U").split()


def symbol(Z: int) -> str:
    return _SYMBOLS[Z - 1] if 0 < Z <= len(_SYMBOLS) else f"Z{Z}"


class ShellTable:
    """One deck's what-if table: the library's handle, and the records its queries file."""

    def __init__(self, qc, spec: RunSpec):
        deck = solid2_deck.to_deck(spec)
        self._h = qc.BasisShells(json.dumps(deck))
        self.first = self.query(None)                  # the deck's own trim (stated, or its autoTrim's frozen choice)
        self.shells = self.first["shells"]
        self.deck_kept = list(self.first["deckKept"])
        self.auto_kept = list(self.first["autoTrimKept"])

    def query(self, kept: Optional[list[bool]]) -> dict:
        out = json.loads(self._h.query(json.dumps({} if kept is None else {"kept": kept})))
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def stated_trim(self, kept: list[bool]) -> list[dict]:
        """The deck's basis.trim for `kept`: one {Z, l, alpha} per switched-off shell (single-primitive shells only)."""
        out = []
        for s, on in zip(self.shells, kept):
            if not on:
                if len(s["exponents"]) != 1:
                    raise ValueError(f"{symbol(s['Z'])} {L_NAMES[s['l']]} {s['exponents']}: a contracted shell -- a stated trim names one exponent")
                out.append({"Z": s["Z"], "l": s["l"], "alpha": s["exponents"][0]})
        return out


def _g(v, fmt: str) -> str:
    """A record number; null is the library's infinity (no function left at some k)."""
    return "—" if v is None else format(v, fmt)


def summary(a: dict) -> str:
    return (f"{'✓ good to go' if a['healthy'] else '⚠ near-singular'}: S min eigenvalue {_g(a['lambdaMin'], '.3e')} at k {a['worstK']}, "
            f"condition number {_g(a['cond'], '.3g')}; {a['aoDrops']} function(s) would drop at the floor {a['floor']:g} "
            f"(summed over k); {a['nFunctions']} functions per k-block")


class _Num(QtWidgets.QTableWidgetItem):
    """A cell that sorts by its value, not its text."""

    def __lt__(self, other):
        a, b = self.data(QtCore.Qt.UserRole + 1), other.data(QtCore.Qt.UserRole + 1)
        return a < b if a is not None and b is not None else super().__lt__(other)


class ShellDialog(QtWidgets.QDialog):
    """The pop-out table (non-modal: it can sit beside the Setup panel).  Untick a shell: the verdict and every what-if column update
    at once.  'Use as the draft's trim' hands the stated trim back through `apply(trim)`."""

    COLUMNS = ("on", "element", "l", "exponent(s)", "functions", "autoTrim", "λ_min if toggled", "cond if toggled",
               "drops if toggled", "near-null share")
    #: what each column says, for a reader who has not met it (the header tooltips; the user, 2026-11-12: "explain the near null share")
    TIPS = ("Tick to keep the shell, untick to switch it off (from every site of its element, at every k).  A contracted shell "
            "cannot be switched off here",
            "The element whose sites carry the shell",
            "The shell's angular momentum: s, p, d, f, ...",
            "The Gaussian exponent(s) α in a.u.: a SMALL α is a DIFFUSE function, the usual cause of near-dependence in a crystal",
            "Functions per site this shell adds (2l+1 spherical; Cartesian d has 6, f 10)",
            "✂: the vet stage's own trim (autoTrim at the floor) would remove this shell",
            "The overlap matrix S's lowest eigenvalue, at the worst k, if this ONE shell were switched (off if on, on if off).  "
            "Near 0 means the basis is nearly linearly dependent; bigger is healthier",
            "S's condition number (largest / smallest eigenvalue) at the worst k if this ONE shell were switched.  Smaller is healthier",
            "How many functions would still fall under the floor (summed over k) if this ONE shell were switched.  0 = good to go "
            "(green: switching this shell alone fixes the basis)",
            "NEAR-NULL SHARE.  When the basis is near-singular, S has eigenvectors with eigenvalues under the floor: directions in "
            "which some combination of functions almost cancels (a function nearly equal to a sum of its neighbours).  This column "
            "is the fraction of those vectors' weight (at the worst k) that sits on this shell's functions.  The shares add up to 1, "
            "so the shell with the largest share is the one most responsible for the near-dependence: switching it off usually "
            "helps most.  0 when the basis is healthy (there are no such vectors)")

    def __init__(self, table: ShellTable, apply: Callable[[list[dict]], None], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Basis shells: conditioning what-if")
        self.resize(940, 580)
        self.table, self._apply = table, apply
        self.kept = list(table.deck_kept)
        lay = QtWidgets.QVBoxLayout(self)
        self.head = QtWidgets.QLabel(); self.head.setWordWrap(True)
        self.head.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.head)
        note = QtWidgets.QLabel("A shell goes from every site of its element and every k.  'if toggled': S with that ONE shell "
                                "switched off (or back on) -- lowest eigenvalue and condition number at the worst k, and the "
                                "functions that would still drop at the floor (0 = good to go).  'autoTrim': ✂ where the vet "
                                "stage's own trim at the floor removes the shell.  'near-null share': how much of the basis's "
                                "near-dependence sits on this shell -- the share of S's eigenvectors under the floor (the "
                                "nearly cancelling combinations of functions) carried by its functions at the worst k; the "
                                "shares add up to 1, and the largest is the likeliest culprit.  Hover a column header for more.  "
                                "Sort by any column.")
        note.setWordWrap(True); note.setStyleSheet("color: palette(mid)")
        lay.addWidget(note)
        self.grid = QtWidgets.QTableWidget(len(table.shells), len(self.COLUMNS))
        self.grid.setHorizontalHeaderLabels(self.COLUMNS)
        for c, tip in enumerate(self.TIPS):
            self.grid.horizontalHeaderItem(c).setToolTip(tip)
        self.grid.verticalHeader().setVisible(False)
        self.grid.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.grid.itemChanged.connect(self._on_item)
        lay.addWidget(self.grid, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset to the draft's trim")
        self.auto_btn = QtWidgets.QPushButton("autoTrim's pick")
        self.auto_btn.setToolTip("Switch off exactly the shells the vet stage's trim removes at the floor")
        self.apply_btn = QtWidgets.QPushButton("Use as the draft's trim")
        self.apply_btn.setToolTip("Writes the switched-off shells as the deck's STATED trim (basis.trim), replacing a trim preset")
        close = QtWidgets.QPushButton("Close")
        for w in (self.reset_btn, self.auto_btn):
            buttons.addWidget(w)
        buttons.addStretch(1); buttons.addWidget(self.apply_btn); buttons.addWidget(close)
        lay.addLayout(buttons)
        self.reset_btn.clicked.connect(lambda: self._set(table.deck_kept))
        self.auto_btn.clicked.connect(lambda: self._set(table.auto_kept))
        self.apply_btn.clicked.connect(self._use)
        close.clicked.connect(self.close)
        self._show(table.first)

    # -- the rows ---------------------------------------------------------------------------------------------------------
    def _num(self, v, text: str) -> QtWidgets.QTableWidgetItem:
        it = _Num(text)
        it.setData(QtCore.Qt.UserRole + 1, float("inf") if v is None else float(v))   # sorts by value, not text
        return it

    def _show(self, a: dict) -> None:
        self.record = a
        self.head.setText(summary(a))
        self.grid.blockSignals(True)
        self.grid.setSortingEnabled(False)          # rows stay put while they are refilled
        rows = a["shells"]
        top = max(r["participation"] for r in rows) or 1.0
        for i, r in enumerate(rows):
            row = next((j for j in range(self.grid.rowCount()) if self.grid.item(j, 0) and self.grid.item(j, 0).data(QtCore.Qt.UserRole) == i), i)
            on = QtWidgets.QTableWidgetItem()
            on.setData(QtCore.Qt.UserRole, i)
            on.setCheckState(QtCore.Qt.Checked if self.kept[i] else QtCore.Qt.Unchecked)
            contracted = len(r["exponents"]) != 1
            on.setFlags(QtCore.Qt.ItemIsEnabled | (QtCore.Qt.NoItemFlags if contracted else QtCore.Qt.ItemIsUserCheckable))
            if contracted:
                on.setToolTip("a contracted shell: a stated trim names one exponent")
            self.grid.setItem(row, 0, on)
            self.grid.setItem(row, 1, QtWidgets.QTableWidgetItem(symbol(r["Z"])))
            self.grid.setItem(row, 2, QtWidgets.QTableWidgetItem(L_NAMES[r["l"]] if r["l"] < len(L_NAMES) else str(r["l"])))
            self.grid.setItem(row, 3, self._num(min(r["exponents"]), ", ".join(f"{e:g}" for e in r["exponents"])))
            self.grid.setItem(row, 4, self._num(r["nAO"], str(r["nAO"])))
            cut = not self.table.auto_kept[i]
            self.grid.setItem(row, 5, self._num(int(cut), "✂" if cut else ""))
            self.grid.setItem(row, 6, self._num(r["lambdaMinToggled"], _g(r["lambdaMinToggled"], ".3e")))
            self.grid.setItem(row, 7, self._num(r["condToggled"], _g(r["condToggled"], ".3g")))
            drops = self._num(r["aoDropsToggled"], str(r["aoDropsToggled"]))
            if r["aoDropsToggled"] == 0 and not a["healthy"]:
                drops.setBackground(QtGui.QColor(26, 127, 55, 90))   # toggling this ONE shell alone makes the basis good to go
            self.grid.setItem(row, 8, drops)
            p = r["participation"]
            share = self._num(p, f"{p:.3f}" if self.kept[i] else "—")
            if self.kept[i] and p > 0:                 # the heat map: red in proportion to the share
                share.setBackground(QtGui.QColor(207, 34, 46, int(200 * p / top)))
            self.grid.setItem(row, 9, share)
            for c in range(1, len(self.COLUMNS)):
                self.grid.item(row, c).setForeground(QtGui.QBrush() if self.kept[i] else QtGui.QColor("gray"))
        self.grid.setSortingEnabled(True)
        self.grid.resizeColumnsToContents()
        self.grid.blockSignals(False)

    def _on_item(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        i = item.data(QtCore.Qt.UserRole)
        on = item.checkState() == QtCore.Qt.Checked
        if self.kept[i] != on:
            self.kept[i] = on
            self._show(self.table.query(self.kept))

    def _set(self, kept: list[bool]) -> None:
        self.kept = list(kept)
        self._show(self.table.query(self.kept))

    def _use(self) -> None:
        try:
            self._apply(self.table.stated_trim(self.kept))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Basis shells", str(e))
