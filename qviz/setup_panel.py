"""
qviz.setup_panel -- the Setup panel: everything that DEFINES a run, in tabs (Structure, Hamiltonian, Basis, SCF, Grids,
Spec, Codes), with the back-end strip and ▶ Run underneath (doc/RunSpecPlan.md §5.1).

Build (beside ▶ Run) does every pre-SCF step on SOLID² -- basis, grids, pseudopotentials -- on the app's serial library
thread, and the Basis and Grids tabs then read the build run's record.  Run without Build is fine: SOLID² converges a
kept Build of the same deck, and builds afresh otherwise (after an edit).  The Grids tab shows the grids ε resolves to:
PLANNED from the spec, BUILT from the library's own run record (a run's, or this spec's Build).

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

from . import basisinfo, geometry, gridinfo, molecules, symmetry
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
_SOLID2 = "SOLID²"
SV_TOLS = ("default (5e-9)", "1e-10", "1e-9", "1e-8", "1e-7", "1e-6", "1e-5")
ENGAGE_FDS = ("default (Z-scaled)", "10", "1", "0.1", "0.01", "0.001")
LADDER_INFO = ("Ladder, as the library runs it for a molecule today: DIIS from the first iteration, then GDM only if DIIS "
               "stalls (out of history and no progress for 5 iterations while |ΔE/E| > 1e-8).  A converging DIIS never hands "
               "over, so on most molecules this is plain DIIS.  The ΔE/E switch-over (to GDM at ΔE/E < 1e-6) is off and "
               "cannot be set yet: a library fix, logged.  The DIIS rows below set its DIIS rung.")                                     # the back end Build runs on (the library's own pre-SCF steps)


def _mono() -> QtGui.QFont:
    return QtGui.QFont("monospace")


class SetupPanel(QtWidgets.QWidget):
    run_requested = QtCore.Signal(object, list)        # (spec, back-end names): the user's ▶ Run click
    refused = QtCore.Signal(str)                       # an edit the spec refused, and why
    _built = QtCore.Signal(str, object)                # (spec JSON, record | the exception): a Build finished

    def __init__(self, draft: SpecDraft, registry: BackendRegistry, qc=None,
                 legs: Optional[Callable[[object], tuple[list[str], list[tuple[str, list[str]]]]]] = None,
                 serial: Optional[Callable[[Callable[[], None]], None]] = None,
                 runs: Optional[Callable[[], list]] = None, parent=None):
        """`qc`: the qchem_py module for the detected symmetry (None: the readout says so).  `legs(run)`: the Codes table
        for a run and its sibling runs of the same spec (compare.code_table), supplied by the app that owns the workspace.
        `serial(fn)`: runs `fn` on the app's ONE library thread (the library's caches and run record are process globals,
        so a Build must not overlap a run); None runs it here, synchronously.  `runs()`: the workspace's runs, newest last --
        the draft's Basis and Grids tabs read the newest run of the SAME spec when it has no Build of its own."""
        super().__init__(parent)
        self.draft, self.registry, self._qc, self._legs = draft, registry, qc, legs
        self._serial = serial or (lambda fn: fn())
        self._runs = runs or (lambda: [])
        self._builds: dict[str, object] = {}           # spec JSON -> its Build's record, "pending", or the exception it raised
        self._built.connect(self._on_built)
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
        self._build_structure(); self._build_hamiltonian(); self._build_basis(); self._build_scf(); self._build_grids()
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
        self.build_btn = QtWidgets.QPushButton("Build")
        self.build_btn.setStyleSheet("padding: 4px 12px")
        buttons = QtWidgets.QHBoxLayout(); buttons.addWidget(self.build_btn); buttons.addWidget(self.add_btn, 1)
        foot.addRow(buttons)
        self._foot = QtWidgets.QWidget(); self._foot.setLayout(foot)
        outer.addWidget(self._foot)

        self.copy_btn.clicked.connect(self._copy_to_draft)
        self.draft_btn.clicked.connect(lambda: self.show_run(None))
        self.undo_btn.clicked.connect(self.draft.undo)
        self.build_btn.clicked.connect(self._build)
        self.add_btn.clicked.connect(lambda: self.draft.spec and self.run_requested.emit(self.draft.spec, self.ticked()))
        draft.on_change(self._on_draft)
        self._render()

    # -- the tabs ---------------------------------------------------------------
    def _page(self, title: str) -> QtWidgets.QFormLayout:
        w = QtWidgets.QWidget(); outer = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout(); outer.addLayout(form); outer.addStretch(1)   # rows top-aligned, spare height below
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
        self.basis_summary = QtWidgets.QLabel(); self.basis_summary.setWordWrap(True)
        self.basis_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow("Size / overlap", self.basis_summary)
        self.basis_table = QtWidgets.QTableWidget(0, 4)
        self.basis_table.setHorizontalHeaderLabels(["Element", "Atoms", "Shells", "Functions/atom"])
        self.basis_table.verticalHeader().setVisible(False)
        self.basis_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.basis_table.horizontalHeader().setStretchLastSection(True)
        f.addRow(self.basis_table)

    def _build_scf(self):
        f = self._page("SCF")
        self.acc_box = QtWidgets.QComboBox(); self.acc_box.addItems(ACCELERATORS)
        self.acc_box.activated.connect(lambda _i: self._edit({"accuracy": {"scf": {"accelerator": self.acc_box.currentText()}}}, "accelerator"))
        f.addRow("Accelerator", self.acc_box)
        self.ladder_info = QtWidgets.QLabel(LADDER_INFO); self.ladder_info.setWordWrap(True)
        self.ladder_info.setStyleSheet("color: palette(mid)")
        f.addRow("", self.ladder_info)
        self.diis_depth = QtWidgets.QSpinBox(); self.diis_depth.setRange(0, 30); self.diis_depth.setSpecialValueText("default (4)")
        self.diis_depth.setToolTip("How many past Fock/error pairs DIIS extrapolates over")
        self.diis_depth.editingFinished.connect(lambda: self._edit({"accuracy": {"scf": {"diis_depth": self.diis_depth.value() or None}}}, "DIIS depth"))
        f.addRow("DIIS depth", self.diis_depth)
        self.diis_sv, self.diis_engage = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for box, key, choices, tip in ((self.diis_sv, "diis_sv_tol", SV_TOLS, "DIIS drops history directions whose singular "
                                        "value falls below this: raise it if DIIS goes wild on a near-singular subspace"),
                                       (self.diis_engage, "diis_engage_fd", ENGAGE_FDS, "DIIS sits out (plain mixing) until "
                                        "[F,D] falls below this, then engages")):
            box.addItems(choices); box.setToolTip(tip)
            box.activated.connect(lambda _i, b=box, k=key: self._edit(
                {"accuracy": {"scf": {k: None if b.currentIndex() == 0 else float(b.currentText())}}}, k))
        f.addRow("DIIS SV floor", self.diis_sv)
        f.addRow("DIIS engages below [F,D]", self.diis_engage)
        self.mix_box = QtWidgets.QComboBox(); self.mix_box.addItems(MIXERS)
        self.mix_box.setToolTip("The density mixer.  Auto: Pulay on D for a crystal without a Fock history, else linear; "
                                "Kerker and Pulay mix in G space, so a molecule takes Auto or Linear")
        self.mix_box.activated.connect(lambda _i: self._edit({"accuracy": {"scf": {"mixer": self.mix_box.currentText()}}}, "mixer"))
        f.addRow("ρ mixer", self.mix_box)
        self._scf_form = f
        self.alpha, self.alpha_max = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for box, key, tip in ((self.alpha, "mixing_alpha", "where linear mixing's step α starts"),
                              (self.alpha_max, "mixing_alpha_max", "the ceiling the adaptive α may grow to; lower it to "
                                                                   "tamp down an oscillating SCF")):
            box.setRange(0.01, 1.0); box.setSingleStep(0.05); box.setDecimals(2); box.setValue(1.0); box.setToolTip(tip)
            box.editingFinished.connect(lambda b=box, k=key: self._edit({"accuracy": {"scf": {k: round(b.value(), 4)}}}, k))
        f.addRow("α start", self.alpha)
        f.addRow("α max", self.alpha_max)
        self.mom_check = QtWidgets.QCheckBox("Maximum Overlap Method")
        self.mom_check.setToolTip("Occupy the orbitals that overlap most with a fixed reference occupied set, instead of the "
                                  "lowest eigenvalues: the cure when the occupied configuration keeps changing (the '*'s in "
                                  "the iteration log's cfg column)")
        self.mom_check.toggled.connect(lambda on: self._edit({"accuracy": {"scf": {"mom": on or None}}}, "MOM"))
        f.addRow("Occupations", self.mom_check)
        self.mom_start = QtWidgets.QSpinBox(); self.mom_start.setRange(0, 1000); self.mom_start.setSpecialValueText("default (10)")
        self.mom_start.setToolTip("Plain aufbau for this many iterations, then capture the reference once and hold it")
        self.mom_start.editingFinished.connect(lambda: self._edit({"accuracy": {"scf": {"mom_start": self.mom_start.value() or None}}}, "MOM start"))
        f.addRow("MOM reference at iteration", self.mom_start)
        self.maxit = QtWidgets.QSpinBox(); self.maxit.setRange(1, 10000); self.maxit.setValue(100)
        self.maxit.setToolTip("A safety net, never the recipe: a run that hits it FAILS")
        self.maxit.editingFinished.connect(lambda: self._edit({"accuracy": {"scf": {"max_iter": self.maxit.value()}}}, "max_iter"))
        f.addRow("Max iterations", self.maxit)

    def _build_grids(self):
        f = self._page("Grids")
        self.grids_planned = QtWidgets.QLabel(); self.grids_planned.setWordWrap(True)
        self.grids_planned.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow("Planned", self.grids_planned)
        self.grids_source = QtWidgets.QLabel(); self.grids_source.setWordWrap(True)
        f.addRow("Source", self.grids_source)
        self.grids_built = QtWidgets.QLabel(); self.grids_built.setWordWrap(True)
        self.grids_built.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow("Built", self.grids_built)
        self.grids_table = QtWidgets.QTableWidget(0, len(gridinfo.LADDER_COLUMNS))
        self.grids_table.setHorizontalHeaderLabels(gridinfo.LADDER_COLUMNS)
        self.grids_table.verticalHeader().setVisible(False)
        self.grids_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.grids_table.horizontalHeader().setStretchLastSection(True)
        f.addRow(self.grids_table)

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
    _EDITORS = ("mix_box", "alpha", "alpha_max", "diis_depth", "diis_sv", "diis_engage", "mom_check", "mom_start", "mol_box", "method_box", "basis_box", "pp_check", "pol_check", "mult_spin", "sym_check", "eps_box",
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
        self._render_basis(s)
        self._render_grids(s)
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
        self.alpha.setValue(float(scf.get("mixing_alpha", 1.0)))
        self.alpha_max.setValue(float(scf.get("mixing_alpha_max", 1.0)))
        linear = self.mix_box.currentText() == "Linear" or (self.mix_box.currentText() == "Auto" and mol)
        for box in (self.alpha, self.alpha_max):     # linear mixing's α means nothing to Kerker / Pulay
            self._scf_form.setRowVisible(box, linear)
        self.maxit.setValue(int(scf.get("max_iter", 100)))
        acc = self.acc_box.currentText()
        self._scf_form.setRowVisible(self.ladder_info, acc == "Ladder" and mol)
        for w in (self.diis_depth, self.diis_sv, self.diis_engage):   # the molecular facade's AcceleratorOptions only
            self._scf_form.setRowVisible(w, acc in ("DIIS", "Ladder") and mol)
        self.diis_depth.setValue(int(scf.get("diis_depth") or 0))
        for box, key in ((self.diis_sv, "diis_sv_tol"), (self.diis_engage, "diis_engage_fd")):
            v = scf.get(key)
            i = box.findText(f"{v:g}") if v else 0
            if v and i < 0:
                box.addItem(f"{v:g}"); i = box.count() - 1
            box.setCurrentIndex(i)
        self.mom_check.setChecked(bool(scf.get("mom")))
        self.mom_start.setValue(int(scf.get("mom_start") or 0))
        self._scf_form.setRowVisible(self.mom_start, bool(scf.get("mom")))

    # -- back ends ----------------------------------------------------------------
    def _render_basis(self, s: RunSpec) -> None:
        rec = gridinfo.conditioning(self._record(s)[0])    # a run's (or a crystal build's) own: read, never recomputed
        if rec:
            per = "per k" if s.structure.kind == "crystal" else "per irrep"
            self.basis_summary.setText("\n".join([f"overlap {per} (the library's record):"] + rec))
        else:
            self.basis_summary.setText("\n".join(basisinfo.summary(s, self._qc)))
        rows = basisinfo.element_rows(s)
        self.basis_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.basis_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.basis_table.resizeColumnsToContents()
        self.basis_table.setFixedHeight(self.basis_table.horizontalHeader().height() + 4
                                        + sum(self.basis_table.rowHeight(r) for r in range(len(rows))))

    def draft_record(self) -> tuple[Optional[dict], str]:
        """The draft's record and where it came from (its Build, else its newest run), whatever the panel is showing --
        what the agent's get_grids / get_basis read for the draft."""
        s = self.draft.spec
        return self._record(s, viewed=False) if s is not None else (None, "the draft is empty")

    def _record(self, s: RunSpec, viewed: bool = True) -> tuple[Optional[dict], str]:
        """The run record to read from, and where it came from: the viewed run's, else this spec's Build."""
        if viewed and self._run is not None:
            if self._run.result is not None and self._run.result.record:
                return self._run.result.record, "from this run's record"
            job = self._run.job
            if job is not None and job.live:
                return job.live, "from this run's record so far (it is running)"
            return None, "this run has no record yet (not started, or not SOLID²)"
        key = s.to_json()
        got = self._builds.get(key)
        if isinstance(got, Exception):
            return None, f"the Build refused: {got}"
        if got == "pending":
            return None, "building…"
        if got is not None:
            return got, "from this spec's Build (no SCF)"
        for run in reversed(self._runs()):            # the newest run of this very spec, finished or still running
            if run.spec.to_json() != key:
                continue
            if run.result is not None and run.result.record:
                return run.result.record, f"from the run of this spec ({run.backend_name}, {run.status.value})"
            if run.job is not None and run.job.live:
                return run.job.live, f"from the run of this spec, so far ({run.backend_name}, running)"
        why = self._why_no_build(s)
        return None, f"not built ({why})" if why else "not built yet: Build (or Run) fills this from the library's record"

    def _render_grids(self, s: RunSpec) -> None:
        try:
            self.grids_planned.setText("\n".join(gridinfo.planned(s)))
        except Exception as e:                        # noqa: BLE001 -- a readout, never a crash
            self.grids_planned.setText(f"(no plan: {e})")
        rec, source = self._record(s)
        self.grids_source.setText(source)
        crystal = s.structure.kind == "crystal"
        lines, rows = gridinfo.built(rec)
        trim = gridinfo.conditioning(rec) if crystal else []
        self.grids_built.setText("\n".join(lines + (["basis overlap per k:"] + trim if trim else [])) or "—")
        self.grids_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.grids_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.grids_table.setVisible(bool(rows))
        self.grids_table.resizeColumnsToContents()
        self.grids_table.setFixedHeight(self.grids_table.horizontalHeader().height() + 4
                                        + sum(self.grids_table.rowHeight(r) for r in range(len(rows))))

    def refresh_records(self) -> None:
        """Re-read the Basis and Grids tabs' records (a run of the shown spec started, streamed its setup, or finished)."""
        s = self._spec()
        if s is not None:
            self._render_basis(s)
            self._render_grids(s)

    def _solid2(self):
        return self.registry.get(_SOLID2) if _SOLID2 in self.registry.names() else None

    def _why_no_build(self, s: RunSpec) -> Optional[str]:
        be = self._solid2()
        return "SOLID² is not registered" if be is None else be.why_no_build(s)

    def _build(self) -> None:
        """Every pre-SCF step on SOLID², on the serial library thread; the tabs then read the build run's record."""
        s, be = self.draft.spec, self._solid2()
        if s is None or be is None or self._why_no_build(s):
            return
        key = s.to_json()
        self._builds[key] = "pending"
        self._render(); self._update_add_label()

        def job():
            try:
                got = be.build(be.translate(s))
            except Exception as e:                    # noqa: BLE001 -- the library's refusal, shown in the tabs
                got = e
            self._built.emit(key, got)
        self._serial(job)

    def _on_built(self, key: str, got) -> None:
        self._builds[key] = got
        self._render()                                # Basis + Grids read the build run's record
        self._update_add_label()

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
        why = self._why_no_build(s) if s is not None else "the draft is empty"
        pending = s is not None and self._builds.get(s.to_json()) == "pending"
        self.build_btn.setEnabled(why is None and not pending)
        self.build_btn.setToolTip(why or "Every pre-SCF step on SOLID² (basis, grids, pseudopotentials), no SCF: the Basis "
                                         "and Grids tabs then show what the library built.  Run reuses it while the spec is unchanged")

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
