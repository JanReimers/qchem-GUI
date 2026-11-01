"""
qviz.setup_panel -- the Setup panel: everything that DEFINES a run, in tabs (Structure, Hamiltonian, Basis, SCF, Spec,
Codes), with the back-end strip and ▶ Run underneath (doc/RunSpecPlan.md §5.1).

It shows ONE spec at a time:
  * the shared DRAFT (editable).  Every widget edit is a user-authored patch to the SpecDraft, on the same undo stack
    and through the same refusal path as the agent's edits: a change the spec refuses leaves the draft as it was, the
    widgets re-render from it, and the reason shows.  The Spec tab is the same draft as JSON (the back-end-NEUTRAL run
    spec; each code's native input -- qchem's deck, CP2K's .inp -- is in the Codes tab).
  * a RUN's spec (read-only: a finished run's results must keep matching what it ran).  "Copy to new draft" makes it
    the draft.
▶ Run is the only thing that starts a calculation; nothing runs on its own (a crystal can be hours).
"""
from __future__ import annotations

import json
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from . import geometry, molecules, symmetry
from .backends.base import BackendRegistry, data_structure
from .draft import Edit, SpecDraft
from .spec import RunSpec, SpecError

#: the method box's names -> (theory, functional)
METHODS = {"HF": ("HF", None), "LDA": ("DFT", "LDA"), "Xalpha": ("DFT", "Xalpha")}
BASES = ("dzvp", "orb1", "dzvp2", "tzvp", "valence_lowq_vb", "sipp_sr")
ACCELERATORS = ("DIIS", "GDM", "Ladder")
MIXERS = ("Auto", "Linear", "Kerker", "Pulay")         # SOLID²'s SCFParams::Mixing; Kerker/Pulay mix in G space (crystals)
EPSILONS = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
_NOT_PRESET = "— draft —"


def _mono() -> QtGui.QFont:
    return QtGui.QFont("monospace")


class SetupPanel(QtWidgets.QWidget):
    run_requested = QtCore.Signal(object, list)        # (spec, back-end names): the user's ▶ Run click
    refused = QtCore.Signal(str)                       # an edit the spec refused, and why

    def __init__(self, draft: SpecDraft, registry: BackendRegistry, qc=None,
                 legs: Optional[Callable[[object], tuple[list[str], list[tuple[str, list[str]]]]]] = None, parent=None):
        """`qc`: the qchem_py module for the detected symmetry (None: the readout says so).  `legs(run)`: the Codes table
        for a run and its sibling runs of the same spec (compare.code_table), supplied by the app that owns the workspace."""
        super().__init__(parent)
        self.draft, self.registry, self._qc, self._legs = draft, registry, qc, legs
        self._run = None                               # the run shown read-only; None = the draft
        outer = QtWidgets.QVBoxLayout(self); outer.setContentsMargins(4, 4, 4, 4)

        head = QtWidgets.QHBoxLayout()
        self.mode_label = QtWidgets.QLabel(); self.mode_label.setWordWrap(True)
        self.copy_btn = QtWidgets.QPushButton("Copy to new draft")
        self.draft_btn = QtWidgets.QPushButton("Back to draft")
        head.addWidget(self.mode_label, 1); head.addWidget(self.copy_btn); head.addWidget(self.draft_btn)
        outer.addLayout(head)

        self.tabs = QtWidgets.QTabWidget(); self.tabs.setDocumentMode(True)
        outer.addWidget(self.tabs, 1)
        self._build_structure(); self._build_hamiltonian(); self._build_basis(); self._build_scf()
        self._build_spec(); self._build_codes()

        foot = QtWidgets.QFormLayout()
        strip = QtWidgets.QWidget(); sh = QtWidgets.QHBoxLayout(strip); sh.setContentsMargins(0, 0, 0, 0)
        self.be_checks: dict[str, QtWidgets.QCheckBox] = {}
        for name in registry.names():
            cb = QtWidgets.QCheckBox(name + (" (illustration)" if registry.get(name).capabilities().synthetic else ""))
            cb.toggled.connect(self._on_ticks); sh.addWidget(cb); self.be_checks[name] = cb
        sh.addStretch(1)
        foot.addRow("Run on", strip)
        self.undo_btn = QtWidgets.QPushButton("Undo last edit")
        self.reason = QtWidgets.QLabel(); self.reason.setWordWrap(True); self.reason.setStyleSheet("color:#cf222e")
        foot.addRow(self.undo_btn, self.reason)
        self.add_btn = QtWidgets.QPushButton()
        self.add_btn.setStyleSheet("font-weight: bold; padding: 4px 12px")
        foot.addRow(self.add_btn)
        self._foot = QtWidgets.QWidget(); self._foot.setLayout(foot)
        outer.addWidget(self._foot)

        self.copy_btn.clicked.connect(self._copy_to_draft)
        self.draft_btn.clicked.connect(lambda: self.show_run(None))
        self.undo_btn.clicked.connect(self.draft.undo)
        self.add_btn.clicked.connect(lambda: self.draft.spec and self.run_requested.emit(self.draft.spec, self.ticked()))
        draft.on_change(self._on_draft)
        self._render()

    # -- the tabs ---------------------------------------------------------------
    def _page(self, title: str) -> QtWidgets.QFormLayout:
        w = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(w)
        sc = QtWidgets.QScrollArea(); sc.setWidget(w); sc.setWidgetResizable(True); sc.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tabs.addTab(sc, title)
        return form

    def _build_structure(self):
        f = self._page("Structure")
        self.mol_box = QtWidgets.QComboBox(); self.mol_box.addItems(molecules.MOLECULES.keys())
        self.mol_box.activated.connect(lambda _i: self._pick_preset(self.mol_box.currentText()))
        f.addRow("Preset", self.mol_box)
        self.sites = QtWidgets.QTableWidget(0, 5); self.sites.setFont(_mono())
        self.sites.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sites.verticalHeader().setVisible(False)
        f.addRow(self.sites)
        self.lattice = QtWidgets.QLabel(); self.lattice.setFont(_mono())
        self.lattice.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow("Lattice (bohr)", self.lattice)
        self.sym_detected = QtWidgets.QLabel(); self.sym_detected.setWordWrap(True)
        self.sym_detected.setToolTip("DETECTED by the library from the positions (and a crystal's site moments) -- what the "
                                     "structure allows; whether the SCF uses it is the box below")
        f.addRow("Symmetry (detected)", self.sym_detected)
        self.sym_check = QtWidgets.QCheckBox("impose: block the SCF by irreps (SALC → irrep labels)")
        self.sym_check.clicked.connect(lambda on: self._edit({"accuracy": {"scf": {"impose_symmetry": on}}}, "symmetry"))
        f.addRow("Symmetry (imposed)", self.sym_check)
        self.geom = QtWidgets.QPlainTextEdit(); self.geom.setReadOnly(True); self.geom.setFont(_mono())
        self.geom.setMinimumHeight(100)
        f.addRow("Geometry", self.geom)

    def _build_hamiltonian(self):
        f = self._page("Hamiltonian")
        self.method_box = QtWidgets.QComboBox(); self.method_box.addItems(METHODS)
        self.method_box.activated.connect(lambda _i: self._edit({"physics": dict(zip(("theory", "functional"),
                                                                METHODS[self.method_box.currentText()]))}, "method"))
        f.addRow("Method", self.method_box)
        self.pp_check = QtWidgets.QCheckBox("pseudopotential (GTH; LDA + a valence basis)")
        self.pp_check.clicked.connect(lambda on: self._edit({"physics": {"core": "pseudopotential" if on else "all-electron"}}, "core"))
        f.addRow("Core", self.pp_check)
        self.pol_check = QtWidgets.QCheckBox("spin-polarized (two channels)")
        self.pol_check.clicked.connect(lambda on: self._edit({"physics": {"spin": {"polarized": on}}}, "spin"))
        f.addRow("Spin", self.pol_check)
        self.mult_spin = QtWidgets.QSpinBox(); self.mult_spin.setRange(0, 7)
        self.mult_spin.setSpecialValueText("auto")     # 0 shows as 'auto' (minimal spin)
        self.mult_spin.setToolTip("Spin multiplicity 2S+1 (auto = minimal). >1 → open-shell, spin-polarized.")
        self.mult_spin.valueChanged.connect(self._on_mult)
        f.addRow("Multiplicity", self.mult_spin)
        self.smearing = QtWidgets.QLabel(); f.addRow("Occupations", self.smearing)
        self.hubbard = QtWidgets.QLabel(); self.hubbard.setWordWrap(True); f.addRow("DFT+U", self.hubbard)
        self.relativity = QtWidgets.QLabel(); f.addRow("Relativity", self.relativity)

    def _build_basis(self):
        f = self._page("Basis")
        self.basis_box = QtWidgets.QComboBox(); self.basis_box.addItems(BASES)
        self.basis_box.setToolTip("orb1 is the smallest/fastest; valence_lowq_vb / sipp_sr are valence (GTH) bases -- "
                                  "with the pseudopotential ticked, the same spec runs on SOLID² and CP2K")
        self.basis_box.activated.connect(lambda _i: self._edit({"accuracy": {"representation": {"basis": self.basis_box.currentText()}}}, "basis"))
        f.addRow("Basis", self.basis_box)
        self.eps_box = QtWidgets.QComboBox(); self.eps_box.addItems(EPSILONS)
        self.eps_box.setToolTip("ε, Ha/atom: the ONE accuracy target; each code's resolved settings say what it became")
        self.eps_box.activated.connect(lambda _i: self._edit({"accuracy": {"epsilon_Ha_per_atom": float(self.eps_box.currentText())}}, "ε"))
        f.addRow("Accuracy ε (Ha/atom)", self.eps_box)
        self.kmesh = QtWidgets.QLabel(); f.addRow("k-mesh", self.kmesh)

    def _build_scf(self):
        f = self._page("SCF")
        self.acc_box = QtWidgets.QComboBox(); self.acc_box.addItems(ACCELERATORS)
        self.acc_box.activated.connect(lambda _i: self._edit({"accuracy": {"scf": {"accelerator": self.acc_box.currentText()}}}, "accelerator"))
        f.addRow("Accelerator", self.acc_box)
        self.mix_box = QtWidgets.QComboBox(); self.mix_box.addItems(MIXERS)
        self.mix_box.setToolTip("The density mixer.  Auto: Pulay on D for a crystal without a Fock history, else linear; "
                                "Kerker and Pulay mix in G space, so a molecule takes Auto or Linear")
        self.mix_box.activated.connect(lambda _i: self._edit({"accuracy": {"scf": {"mixer": self.mix_box.currentText()}}}, "mixer"))
        f.addRow("ρ mixer", self.mix_box)
        self.maxit = QtWidgets.QSpinBox(); self.maxit.setRange(1, 10000); self.maxit.setValue(100)
        self.maxit.setToolTip("A safety net, never the recipe: a run that hits it FAILS")
        self.maxit.editingFinished.connect(lambda: self._edit({"accuracy": {"scf": {"max_iter": self.maxit.value()}}}, "max_iter"))
        f.addRow("Max iterations", self.maxit)

    def _build_spec(self):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("The run spec (back-end neutral; yours and the agent's)"))
        self.spec_view = QtWidgets.QPlainTextEdit(); self.spec_view.setFont(_mono())
        lay.addWidget(self.spec_view, 1)
        row = QtWidgets.QHBoxLayout()
        self.apply_btn = QtWidgets.QPushButton("Apply my edit"); self.apply_btn.clicked.connect(self._apply_json)
        row.addWidget(self.apply_btn); row.addStretch(1)
        lay.addLayout(row)
        self.tabs.addTab(w, "Spec")

    def _build_codes(self):
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.codes = QtWidgets.QTableWidget(); self.codes.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.codes.setFont(_mono())
        split.addWidget(self.codes)
        nat = QtWidgets.QWidget(); nl = QtWidgets.QVBoxLayout(nat); nl.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("Native input of"))
        self.code_pick = QtWidgets.QComboBox(); self.code_pick.currentTextChanged.connect(self._render_native)
        row.addWidget(self.code_pick, 1); nl.addLayout(row)
        self.native = QtWidgets.QPlainTextEdit(); self.native.setReadOnly(True); self.native.setFont(_mono())
        nl.addWidget(self.native, 1)
        split.addWidget(nat)
        self.tabs.addTab(split, "Codes")

    # -- which spec: the draft or a run ---------------------------------------------
    def show_run(self, run) -> None:
        """Show `run`'s spec read-only (None: back to the editable draft)."""
        self._run = run
        self.reason.clear()
        self._render()

    @property
    def viewing_run(self):
        return self._run

    def _spec(self) -> Optional[RunSpec]:
        return self._run.spec if self._run is not None else self.draft.spec

    def _copy_to_draft(self) -> None:
        if self._run is not None:
            self.draft.set(self._run.spec, "user", f"copied from run {self._run.summary}")
            self.show_run(None)

    # -- edits: every one goes through the draft --------------------------------
    def _edit(self, patch: dict, what: str) -> None:
        if self._run is not None or self._rendering:
            return
        try:
            self.draft.patch(patch, "user", f"Setup: {what}")
        except SpecError as e:
            self.reason.setText(f"not applied: {e}")
            self.refused.emit(str(e))
            self._render()                               # the widgets show the draft, which did not change

    def _on_mult(self, v: int) -> None:
        self._edit({"physics": {"spin": {"multiplicity": v or None, "polarized": v > 1}}}, "multiplicity")

    def _pick_preset(self, name: str) -> None:
        if self._run is None and name in molecules.MOLECULES:
            self.draft.set(molecules.preset_spec(name), "user", f"Setup: preset {name}")

    def _apply_json(self) -> None:
        try:
            self.draft.set(RunSpec.from_json(self.spec_view.toPlainText()), "user", "edited in the Spec tab")
        except (SpecError, json.JSONDecodeError) as e:
            self.reason.setText(f"not applied: {e}")
            self.refused.emit(str(e))

    def set_structure(self, label: str, numbers, positions) -> None:
        """A molecule from outside (an .xyz import) into the draft: the draft's physics kept, its structure replaced."""
        from .spec import molecule
        self.show_run(None)
        base = self.draft.spec
        if base is None or base.structure.kind != "molecule":
            self.draft.set(molecule(label, numbers, positions), "user", f"import {label}")
            return
        new = molecule(label, numbers, positions)
        self._edit({"label": label, "structure": new.to_dict()["structure"]}, f"import {label}")

    # -- the spec -> the widgets ------------------------------------------------
    def _on_draft(self, _draft: SpecDraft, _edit: Optional[Edit]) -> None:
        if self._run is None:
            self.reason.clear()
            self._render()

    _rendering = False
    _EDITORS = ("mix_box", "mol_box", "method_box", "basis_box", "pp_check", "pol_check", "mult_spin", "sym_check", "eps_box",
                "acc_box", "maxit")

    def _render(self) -> None:
        s = self._spec()
        ro = self._run is not None
        self._rendering = True
        editors = [getattr(self, n) for n in self._EDITORS]
        try:
            for w in editors:
                w.blockSignals(True)
            self.mode_label.setText(f"<b>Run</b> {self._run.summary} <i>(read-only)</i>" if ro
                                    else "<b>Draft</b> " + (s.summary if s else "(empty)"))
            self.copy_btn.setVisible(ro); self.draft_btn.setVisible(ro)
            self._foot.setVisible(not ro)
            self.spec_view.setReadOnly(ro); self.apply_btn.setVisible(not ro)
            self.spec_view.setPlainText(s.to_json(indent=1) if s else "")
            mol = s is not None and s.structure.kind == "molecule"
            self._render_preset(s, mol)
            for w in editors:
                w.setEnabled(not ro and s is not None)
            for w in (self.method_box, self.pp_check, self.mult_spin, self.sym_check, self.basis_box):
                w.setEnabled(not ro and mol)          # a crystal's are stated through its deck (Spec / representation)
            self.mult_spin.setVisible(s is None or mol)
            if s is not None:
                self._render_values(s, mol)
            else:
                self.sites.setRowCount(0); self.lattice.clear(); self.sym_detected.clear(); self.geom.clear()
        finally:
            for w in editors:
                w.blockSignals(False)
            self._rendering = False
        self._refresh_strip()
        self._render_codes()

    def _render_preset(self, s, mol: bool) -> None:
        preset = molecules.preset_of(s) if mol else None
        i = self.mol_box.findText(_NOT_PRESET)
        if preset:
            if i >= 0:
                self.mol_box.removeItem(i)
            self.mol_box.setCurrentText(preset)
            return
        if i < 0:
            self.mol_box.addItem(_NOT_PRESET); i = self.mol_box.count() - 1
        self.mol_box.setItemText(i, _NOT_PRESET if s is None else f"draft: {s.label}")
        self.mol_box.setCurrentIndex(i)

    def _render_values(self, s: RunSpec, mol: bool) -> None:
        p, ac = s.physics, s.accuracy
        st = s.structure
        moments = p.spin.site_moments or ()
        heads = ["site", "el", *(("x", "y", "z") if mol else ("a", "b", "c"))] + ([] if mol else ["μB"])
        self.sites.setColumnCount(len(heads)); self.sites.setHorizontalHeaderLabels(heads)
        self.sites.setRowCount(len(st.sites))
        for i, site in enumerate(st.sites):
            r = site.xyz if mol else site.frac
            cells = [site.tag or str(i + 1), site.element, *(f"{x:.5f}" for x in r)]
            if not mol:
                cells.append(f"{moments[i]:+.1f}" if i < len(moments) else "")
            for j, c in enumerate(cells):
                self.sites.setItem(i, j, QtWidgets.QTableWidgetItem(c))
        self.sites.resizeColumnsToContents()
        rows = min(len(st.sites), 12)                  # as tall as the sites (up to 12), then it scrolls
        self.sites.setFixedHeight(self.sites.horizontalHeader().height() + rows * self.sites.verticalHeader().defaultSectionSize() + 6)
        self.lattice.setText("\n".join("  ".join(f"{x:9.4f}" for x in row) for row in st.lattice_bohr) if not mol else "—")
        self.sym_detected.setText("\n".join(symmetry.detected(s, self._qc)))
        self.sym_check.setChecked(bool((ac.scf or {}).get("impose_symmetry")))
        try:
            self.geom.setPlainText(geometry.report(data_structure(s)))
        except Exception as e:                        # noqa: BLE001 -- a readout
            self.geom.setPlainText(f"(no geometry report: {e})")
        name = next((k for k, v in METHODS.items() if v == (p.theory, p.functional)), None)
        if name:
            self.method_box.setCurrentText(name)
        self.pp_check.setChecked(p.core == "pseudopotential")
        self.pol_check.setChecked(p.spin.polarized)
        self.mult_spin.setValue(p.spin.multiplicity or 0)
        o = p.occupations
        self.smearing.setText("integer (no smearing)" if o.smearing == "none" else f"{o.smearing}, kT = {o.kT_Ha} Ha")
        self.hubbard.setText("; ".join(f"U = {h.U_eV} eV on {', '.join(h.sites)} (l = {h.l}, {h.projector})"
                                       for h in p.hubbard) or "none")
        self.relativity.setText(p.relativity)
        basis = (ac.representation or {}).get("basis")
        if basis and self.basis_box.findText(basis) < 0:
            self.basis_box.addItem(basis)
        if basis:
            self.basis_box.setCurrentText(basis)
        eps = f"{ac.epsilon_Ha_per_atom:.0e}".replace("e-0", "e-")
        if self.eps_box.findText(eps) < 0:
            self.eps_box.addItem(eps)
        self.eps_box.setCurrentText(eps)
        self.kmesh.setText("×".join(map(str, ac.kmesh)) if ac.kmesh else ("—" if mol else "not stated"))
        scf = ac.scf or {}
        self.acc_box.setCurrentText(str(scf.get("accelerator", "DIIS")))
        self.mix_box.setCurrentText(str(scf.get("mixer", "Auto")))
        self.maxit.setValue(int(scf.get("max_iter", 100)))

    # -- back ends ----------------------------------------------------------------
    def ticked(self) -> list[str]:
        return [n for n, cb in self.be_checks.items() if cb.isEnabled() and cb.isChecked()]

    def _on_ticks(self, *_):
        self._update_add_label()
        self._render_codes()

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

    # -- the Codes tab: resolved settings side by side, and each code's native input --------------------
    def _jobs(self) -> dict:
        """{code: Job or the reason it cannot translate} for what is shown: the run's own code, or the ticked ones."""
        s = self._spec()
        if s is None:
            return {}
        names = [self._run.backend_name] if self._run is not None else self.ticked()
        out = {}
        for n in names:
            try:
                out[n] = self.registry.get(n).translate(s)
            except Exception as e:                    # noqa: BLE001 -- shown, not raised
                out[n] = str(e)
        return out

    _job_cache: dict = {}

    def _render_codes(self) -> None:
        self._job_cache = jobs = self._jobs()          # one translate per code per render
        if self._run is not None and self._legs is not None:
            heads, rows = self._legs(self._run)
        else:
            heads = list(jobs)
            keys: list[str] = []
            for j in jobs.values():
                for k in (j.resolved if not isinstance(j, str) else {"cannot translate": j}):
                    if k not in keys:
                        keys.append(k)
            rows = [(k, [str((j.resolved if not isinstance(j, str) else {"cannot translate": j}).get(k, ""))
                         for j in jobs.values()]) for k in keys]
        self.codes.clear()
        self.codes.setColumnCount(len(heads)); self.codes.setRowCount(len(rows))
        self.codes.setHorizontalHeaderLabels(heads); self.codes.setVerticalHeaderLabels([k for k, _ in rows])
        for i, (_k, vals) in enumerate(rows):
            for j, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v); item.setToolTip(v)
                self.codes.setItem(i, j, item)
        self.codes.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        names = list(jobs)
        cur = self.code_pick.currentText()
        self.code_pick.blockSignals(True); self.code_pick.clear(); self.code_pick.addItems(names)
        if cur in names:
            self.code_pick.setCurrentText(cur)
        self.code_pick.blockSignals(False)
        self._render_native(self.code_pick.currentText())

    def _render_native(self, name: str) -> None:
        j = self._job_cache.get(name)
        if j is None:
            self.native.setPlainText(""); return
        if isinstance(j, str):
            self.native.setPlainText(f"cannot translate: {j}"); return
        if "deck" in j.files:
            self.native.setPlainText(j.files["deck"].decode(errors="replace"))
        else:                                          # an in-process code's arguments: one per line, values compact
            self.native.setPlainText("{\n" + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, default=str)}"
                                                         for k, v in j.native.items()) + "\n}")
