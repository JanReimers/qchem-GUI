"""
qviz.inspector -- the Inspector: a VIEW of the shared SpecDraft (doc/RunSpecPlan.md §5.1), not a second home for the spec.

Every widget edit is a user-authored patch to the draft, on the same undo stack and through the same refusal path as
the agent's edits: a change the spec refuses (a multiplicity of the wrong parity) leaves the draft as it was, the
widgets re-render from it, and the reason shows.  Every draft change, whoever made it, re-renders the widgets and
re-judges the back-end strip (option 2.2).  The Add button asks for the DRAFT on the ticked back ends.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from . import molecules
from .draft import Edit, SpecDraft
from .backends.base import BackendRegistry
from .spec import RunSpec, SpecError

#: the method box's names -> (theory, functional)
METHODS = {"HF": ("HF", None), "LDA": ("DFT", "LDA"), "Xalpha": ("DFT", "Xalpha")}
BASES = ("dzvp", "orb1", "dzvp2", "tzvp", "valence_lowq_vb", "sipp_sr")
_NOT_PRESET = "— draft —"


class Inspector(QtWidgets.QWidget):
    add_requested = QtCore.Signal(object, list)        # (spec, back-end names): the user's Add click
    refused = QtCore.Signal(str)                       # an edit the spec refused, and why

    def __init__(self, draft: SpecDraft, registry: BackendRegistry, parent=None):
        super().__init__(parent)
        self.draft, self.registry = draft, registry
        form = QtWidgets.QFormLayout(self)
        self.mol_box = QtWidgets.QComboBox(); self.mol_box.addItems(molecules.MOLECULES.keys())
        self.method_box = QtWidgets.QComboBox(); self.method_box.addItems(METHODS)
        self.add_btn = QtWidgets.QPushButton()
        row = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.mol_box); h.addWidget(self.method_box)
        form.addRow("Molecule", row)
        self.basis_box = QtWidgets.QComboBox(); self.basis_box.addItems(BASES)
        self.basis_box.setToolTip("orb1 is the smallest/fastest; valence_lowq_vb / sipp_sr are valence (GTH) bases -- "
                                  "with the pseudopotential ticked, the same spec runs on SOLID² and CP2K")
        form.addRow("Basis", self.basis_box)
        self.pp_check = QtWidgets.QCheckBox("pseudopotential (GTH; LDA + a valence basis)")
        form.addRow(self.pp_check)
        self.mult_spin = QtWidgets.QSpinBox(); self.mult_spin.setRange(0, 7)
        self.mult_spin.setSpecialValueText("auto")     # 0 shows as 'auto' (minimal spin)
        self.mult_spin.setToolTip("Spin multiplicity 2S+1 (auto = minimal). >1 → open-shell, spin-polarized.")
        form.addRow("Multiplicity", self.mult_spin)
        self.sym_check = QtWidgets.QCheckBox("point-group symmetry (SALC → irrep labels)")
        form.addRow(self.sym_check)
        strip = QtWidgets.QWidget(); sh = QtWidgets.QHBoxLayout(strip); sh.setContentsMargins(0, 0, 0, 0)
        self.be_checks: dict[str, QtWidgets.QCheckBox] = {}
        for name in registry.names():
            cb = QtWidgets.QCheckBox(name + (" (illustration)" if registry.get(name).capabilities().synthetic else ""))
            cb.toggled.connect(self._update_add_label); sh.addWidget(cb); self.be_checks[name] = cb
        sh.addStretch(1)
        form.addRow("Run on", strip)
        self.undo_btn = QtWidgets.QPushButton("Undo last edit")
        self.reason = QtWidgets.QLabel(); self.reason.setWordWrap(True); self.reason.setStyleSheet("color:#cf222e")
        form.addRow(self.undo_btn, self.reason)
        # the ONLY thing that starts a calculation: nothing runs on its own (a crystal can be hours)
        self.add_btn.setStyleSheet("font-weight: bold; padding: 4px 12px")
        form.addRow(self.add_btn)

        self.mol_box.activated.connect(lambda _i: self._pick_preset(self.mol_box.currentText()))
        self.method_box.activated.connect(lambda _i: self._edit({"physics": dict(zip(("theory", "functional"),
                                                                                     METHODS[self.method_box.currentText()]))}, "method"))
        self.basis_box.activated.connect(lambda _i: self._edit({"accuracy": {"representation": {"basis": self.basis_box.currentText()}}}, "basis"))
        self.pp_check.clicked.connect(lambda on: self._edit({"physics": {"core": "pseudopotential" if on else "all-electron"}}, "core"))
        self.mult_spin.valueChanged.connect(self._on_mult)
        self.sym_check.clicked.connect(lambda on: self._edit({"accuracy": {"scf": {"impose_symmetry": on}}}, "symmetry"))
        self.undo_btn.clicked.connect(self.draft.undo)
        self.add_btn.clicked.connect(lambda: self.draft.spec and self.add_requested.emit(self.draft.spec, self.ticked()))
        draft.on_change(self._on_draft)
        self._render()

    # -- edits: every one goes through the draft --------------------------------
    def _edit(self, patch: dict, what: str) -> None:
        try:
            self.draft.patch(patch, "user", f"Inspector: {what}")
        except SpecError as e:
            self.reason.setText(f"not applied: {e}")
            self.refused.emit(str(e))
            self._render()                               # the widgets show the draft, which did not change

    def _on_mult(self, v: int) -> None:
        if not self._rendering:
            self._edit({"physics": {"spin": {"multiplicity": v or None, "polarized": v > 1}}}, "multiplicity")

    def _pick_preset(self, name: str) -> None:
        if name in molecules.MOLECULES:
            self.draft.set(molecules.preset_spec(name), "user", f"Inspector: preset {name}")

    def set_structure(self, label: str, numbers, positions) -> None:
        """A molecule from outside (an .xyz import) into the draft: the draft's physics kept, its structure replaced."""
        from .spec import molecule
        base = self.draft.spec
        if base is None or base.structure.kind != "molecule":
            self.draft.set(molecule(label, numbers, positions), "user", f"import {label}")
            return
        new = molecule(label, numbers, positions)
        self._edit({"label": label, "structure": new.to_dict()["structure"]}, f"import {label}")

    # -- the draft -> the widgets ---------------------------------------------
    def _on_draft(self, _draft: SpecDraft, _edit: Optional[Edit]) -> None:
        self.reason.clear()
        self._render()

    _rendering = False

    def _render(self) -> None:
        s: Optional[RunSpec] = self.draft.spec
        self._rendering = True
        try:
            for w in (self.mol_box, self.method_box, self.basis_box, self.pp_check, self.mult_spin, self.sym_check):
                w.blockSignals(True)
            mol = s is not None and s.structure.kind == "molecule"
            preset = molecules.preset_of(s) if mol else None
            i = self.mol_box.findText(_NOT_PRESET)
            if preset:
                if i >= 0:
                    self.mol_box.removeItem(i)
                self.mol_box.setCurrentText(preset)
            else:
                if i < 0:
                    self.mol_box.addItem(_NOT_PRESET); i = self.mol_box.count() - 1
                self.mol_box.setItemText(i, _NOT_PRESET if s is None else f"draft: {s.label}")
                self.mol_box.setCurrentIndex(i)
            for w in (self.method_box, self.basis_box, self.pp_check, self.mult_spin, self.sym_check):
                w.setEnabled(mol)
            if s is not None:
                p = s.physics
                name = next((k for k, v in METHODS.items() if v == (p.theory, p.functional)), None)
                if name:
                    self.method_box.setCurrentText(name)
                basis = (s.accuracy.representation or {}).get("basis")
                if basis and self.basis_box.findText(basis) < 0:
                    self.basis_box.addItem(basis)
                if basis:
                    self.basis_box.setCurrentText(basis)
                self.pp_check.setChecked(p.core == "pseudopotential")
                self.mult_spin.setValue(p.spin.multiplicity or 0)
                self.sym_check.setChecked(bool((s.accuracy.scf or {}).get("impose_symmetry")))
        finally:
            for w in (self.mol_box, self.method_box, self.basis_box, self.pp_check, self.mult_spin, self.sym_check):
                w.blockSignals(False)
            self._rendering = False
        self._refresh_strip()

    def ticked(self) -> list[str]:
        return [n for n, cb in self.be_checks.items() if cb.isEnabled() and cb.isChecked()]

    def _refresh_strip(self) -> None:
        s = self.draft.spec
        if s is None:
            for cb in self.be_checks.values():
                cb.setEnabled(False); cb.setToolTip("the draft is empty")
        else:
            for n, (able, ticked, tip) in self.registry.strip(s, self.ticked()).items():
                cb = self.be_checks[n]
                cb.blockSignals(True); cb.setEnabled(able); cb.setChecked(ticked); cb.setToolTip(tip); cb.blockSignals(False)
        self._update_add_label()

    def _update_add_label(self, *_):
        on, s = self.ticked(), self.draft.spec
        what = s.summary if s else "nothing drafted"
        self.add_btn.setText("▶ Run" + (f" on {' + '.join(on)}" if on else " (no back end ticked)"))
        self.add_btn.setToolTip(f"Start {what}" + (f" on {' + '.join(on)}" if on else ""))
        self.add_btn.setEnabled(bool(on) and s is not None)
