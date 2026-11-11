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
import os
import pathlib
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from . import basisinfo, bse, geometry, gridinfo, materials, molecules, policyinfo, states, symmetry
from .backends import solid2_deck
from .backends.base import BackendRegistry, code_block, data_structure
from .backends.solid2 import mesh_for
from .draft import Edit, SpecDraft
from .spec import RunSpec, SpecError

#: the method box's names -> (theory, functional)
METHODS = {"HF": ("HF", None), "LDA": ("DFT", "LDA"), "Xalpha": ("DFT", "Xalpha")}
BASES = ("dzvp", "orb1", "dzvp2", "tzvp", "valence_lowq_vb", "sipp_sr")
ACCELERATORS = ("Auto", "DIIS", "GDM", "Ladder", "Null")   # Auto: the facade's policy (a molecule's is DIIS)
MIXERS = ("Auto", "Linear", "Kerker", "Pulay")         # SOLID²'s SCFParams::Mixing; Kerker/Pulay mix in G space (crystals)
TRIM_TOL = 1e-6                                 # the trim's orthoTol: the conditioning report's floor (OverlapConditioning)
BASIS_TIP = ("orb1 is the smallest/fastest; valence_lowq_vb / sipp_sr are valence (GTH) bases -- with the pseudopotential "
             "ticked, the same spec runs on SOLID² and CP2K")
EPSILONS = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-7")
_NOT_PRESET = "— draft —"
_CHOOSE_BASIS = "— choose a basis —"           # a crystal with no basis stated: the orbital basis is user selected
CRYSTAL_PHYSICS = ("A crystal runs LDA with GTH pseudopotentials: the periodic facade's one pseudopotential set is "
                   "gth-lda-valence (HF and Xα are molecular here)")
_SOLID2 = "SOLID²"
SV_TOLS = ("default", "1e-10", "1e-9", "1e-8", "1e-7", "1e-6", "1e-5")
ENGAGE_FDS = ("default", "10", "1", "0.1", "0.01", "0.001")
LADDER_INFO = ("Ladder: DIIS from the first iteration, handing off to GDM once |ΔE/E| falls below the switch-over "
               "(default 1e-6), or earlier if DIIS stalls (no progress for 5 iterations).  A molecule and a crystal read the "
               "same signal.  The DIIS rows set its DIIS rung.")
#: the convergence gates a run can state (accuracy.scf): key, label, what it measures.  Absent = the code's default.
GATE_KNOBS = (("gate_fd", "[F,D] below", "the worst irrep's orbital gradient [F,D] (the facade's default 1e-5)"),
              ("gate_drho", "Density change below", "the density change per electron: Δρ on the run's own quadrature, or ‖ΔD‖ on a "
                                                    "run with none, such as HF -- the two are not comparable (default 1e-4)"),
              ("gate_de", "|ΔE/E| below", "the relative total-energy change (default off): the physical gate when [F,D] and "
                                         "Δρ limit-cycle above a fit floor while the energy has settled"),
              ("gate_dfd", "|Δ[F,D]| below", "the change in [F,D] between iterations (default off: a stagnation measure -- "
                                            "it passes by accident when [F,D] stalls high)"))
GATE_CHOICES = ("off", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9", "1e-10")
#: what an unstated gate IS (SCFParams' defaults; a crystal's deck reads the same struct -- tests pin it to the resolved deck)
GATE_DEFAULTS = {"gate_fd": 1e-5, "gate_drho": 1e-4, "gate_de": "off", "gate_dfd": "off"}
LEBEDEV_DEGREES = (5, 7, 9, 11, 15, 17, 19, 21, 23, 29, 31, 35, 41, 47)   # the library's Lebedev menu (13/25/27: excluded there)
SWITCH_ATS = ("default (1e-6)", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "off (stall only)")                                     # the back end Build runs on (the library's own pre-SCF steps)


def _sci(v: float) -> str:
    """A gate as its menu spells it: 1e-4, not 0.0001 (a non-power of ten keeps its digits: 5e-9, 2.5e-6)."""
    t = f"{v:.0e}"
    return (t if float(t) == v else f"{v:.6g}" if "e" in f"{v:.6g}" else f"{v:.1e}").replace("e-0", "e-")


SPIN_NAMES = {1: "singlet", 2: "doublet", 3: "triplet", 4: "quartet", 5: "quintet", 6: "sextet", 7: "septet"}


def auto_multiplicity(s: RunSpec) -> str:
    """What 'auto' multiplicity resolves to for this molecule: the minimal 2S+1 its electron count allows."""
    m = 1 + s.num_electrons % 2
    return f"auto: {m}, {SPIN_NAMES[m]}, {'polarized' if s.physics.spin.polarized else 'unpolarized'}"


def accelerator_defaults(s: RunSpec, mol: bool) -> dict:
    """The DIIS knobs the library runs when the panel leaves them unstated -- spelled out, so 'default' says a value.
    A molecule: MoleculeSetup's AcceleratorOptions and its engage gate (HF all-electron: Z^2*0.08/32 on the total nuclear
    charge; DFT or a pseudopotential: 100, DIIS from the first iteration).  A crystal: the SCFAccelerator factory's DIISParams.
    Pinned against the library by tests (the molecular run's accelerators record; the deck's resolved acceleratorOptions)."""
    if not mol:
        return {"depth": 8, "sv": 1e-9, "engage": "0.1"}
    p = s.physics
    if p.theory != "HF" or p.core == "pseudopotential":
        engage = "from the start"
    else:
        engage = f"{s.num_electrons ** 2 * 0.08 / 32:.3g}, Z-scaled"
    return {"depth": 4, "sv": 5e-9, "engage": engage}


def _mono() -> QtGui.QFont:
    return QtGui.QFont("monospace")


class SetupPanel(QtWidgets.QWidget):
    workspace_library_changed = QtCore.Signal(object)   # the workspace's materials library (a path, or None): the app stores it
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
        # NEW DRAFT (the user, 2026-11-11): start fresh from a structure with its default settings, without copying a run
        self.new_btn = QtWidgets.QToolButton(); self.new_btn.setText("New draft ▾")
        self.new_btn.setToolTip("Start a new draft from a molecule preset or a material, with its default settings (the draft's "
                                "edits stay on its undo stack; File ▸ Import crystal starts one from a CIF / POSCAR / xyz file)")
        self.new_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.new_menu = QtWidgets.QMenu(self.new_btn); self.new_btn.setMenu(self.new_menu)
        self.new_menu.aboutToShow.connect(self._fill_new_menu)
        head.addWidget(self.mode_label, 1); head.addWidget(self.new_btn); head.addWidget(self.copy_btn); head.addWidget(self.draft_btn)
        outer.addLayout(head)
        # the library's policies, as the record states them: refusals and warnings shown, automatic choices one click away
        self.policy_loud = QtWidgets.QLabel(); self.policy_loud.setWordWrap(True)
        self.policy_loud.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.policy_loud.setStyleSheet("border: 1px solid #cf222e; border-radius: 3px; padding: 4px")
        self.policy_more = QtWidgets.QToolButton(); self.policy_more.setCheckable(True)
        self.policy_more.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon); self.policy_more.setAutoRaise(True)
        self.policy_more.setToolTip("settings a policy picked because the spec states none (policy, verdict choose)")
        self.policy_chosen = QtWidgets.QLabel(); self.policy_chosen.setWordWrap(True)
        self.policy_chosen.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.policy_more.toggled.connect(lambda on: (self.policy_chosen.setVisible(on), self._policy_arrow()))
        for w in (self.policy_loud, self.policy_more, self.policy_chosen):
            outer.addWidget(w); w.setVisible(False)

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
        self.build_why = QtWidgets.QLabel(); self.build_why.setWordWrap(True)     # why Build (and SOLID²) cannot run: in plain sight
        self.build_why.setStyleSheet("color: palette(mid)")
        foot.addRow(self.build_why)
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
        f = self._page("Structure"); self._sform = f
        self.mol_box = QtWidgets.QComboBox(); self.mol_box.addItems(molecules.MOLECULES.keys())
        self.mol_box.activated.connect(lambda _i: self._pick_preset(self.mol_box.currentText()))
        f.addRow("Preset", self.mol_box)
        # THE MATERIALS PICKER (OpenWork §2, 2026-11-09): the library's layers -- the shipped materials, then "mine" (the user's
        # ~/.solid2/materials.json, where Save to my materials writes).  Filled from the library on first show and after a save.
        self.material_box = QtWidgets.QComboBox()
        self.material_box.setToolTip("A crystal from the library's materials: the shipped ones, then yours (Save to my materials)")
        self.material_box.activated.connect(self._pick_material)
        self._materials: list[dict] = []
        f.addRow("Material", self.material_box)
        self._ws_library: Optional[str] = None
        wsl = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(wsl); h.setContentsMargins(0, 0, 0, 0)
        self.ws_library_label = QtWidgets.QLabel("none"); self.ws_library_label.setWordWrap(True)
        self.ws_library_btn = QtWidgets.QPushButton("Choose…")
        self.ws_library_btn.setToolTip("This workspace's own materials library (materials.json schema), listed after yours: an "
                                       "existing file, or a new name (created by the first save into it).  Saved with the workspace.")
        self.ws_library_btn.clicked.connect(self._choose_ws_library)
        self.ws_library_clear = QtWidgets.QToolButton(); self.ws_library_clear.setText("✕")
        self.ws_library_clear.setToolTip("This workspace uses no materials library of its own (the file is left as it is)")
        self.ws_library_clear.clicked.connect(lambda: self._set_ws_library(None))
        h.addWidget(self.ws_library_label, 1); h.addWidget(self.ws_library_btn); h.addWidget(self.ws_library_clear)
        f.addRow("Workspace library", wsl)
        self.spins_edit = QtWidgets.QLineEdit()
        self.spins_edit.setPlaceholderText("one per site: + - 0  (empty = no magnetic order)")
        self.spins_edit.setToolTip("The crystal's collinear magnetic order, stated after import: one of + - 0 per site in site "
                                   "order (an AFM-II MnO cell: + - 0 0).  The deck gets the signs (solid.siteSpins).")
        self.spins_edit.editingFinished.connect(self._on_spins)
        f.addRow("Site spins", self.spins_edit)
        self.save_material_btn = QtWidgets.QPushButton("Save to my materials…")
        self.save_material_btn.setToolTip("Save this crystal (cell, sites, site spins, where it came from) to your own materials "
                                          "library, ~/.solid2/materials.json -- never the shipped file")
        self.save_material_btn.clicked.connect(self._save_material)
        f.addRow(self.save_material_btn)
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
        f = self._page("Basis"); self._bform = f
        self.basis_box = QtWidgets.QComboBox(); self.basis_box.addItems(BASES)
        self.basis_box.setToolTip(BASIS_TIP)
        self.basis_box.activated.connect(self._on_basis_pick)
        f.addRow("Basis", self.basis_box)
        self.bse_btn = QtWidgets.QPushButton("Basis Set Exchange…")
        self.bse_btn.setToolTip("Download an orbital basis for this molecule's elements from basissetexchange.org; it is saved with "
                                "its version and citations under ~/.solid2/bases and listed in the Basis box")
        self.bse_btn.clicked.connect(self._open_bse)
        f.addRow("", self.bse_btn)
        self.bse_note = QtWidgets.QLabel(); self.bse_note.setWordWrap(True)
        self.bse_note.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow("Downloaded basis", self.bse_note)
        self.eps_box = QtWidgets.QComboBox(); self.eps_box.addItems(EPSILONS)
        self.eps_box.setToolTip("ε, Ha/atom: the ONE accuracy target; each code's resolved settings say what it became")
        self.eps_box.activated.connect(lambda _i: self._edit({"accuracy": {"epsilon_Ha_per_atom": float(self.eps_box.currentText())}}, "ε"))
        f.addRow("Accuracy ε (Ha/atom)", self.eps_box)
        self.kmesh = QtWidgets.QLabel(); f.addRow("k-mesh", self.kmesh)
        # THE VET-STAGE TRIM (doc/Pins.md pin 22; the user, GUI item 27): a crystal's near-dependent diffuse shells removed once, the
        # most diffuse implicated shell per pass, until no k-block drops an AO at orthoTol.  The record says what went and
        # lambda_min(S) before and after (basisTrim), shown under Size / overlap after a Build.
        self.trim_check = QtWidgets.QCheckBox(f"remove near-dependent diffuse shells (orthoTol {TRIM_TOL:g})")
        self.trim_check.setToolTip("basis.autoTrim with solid.orthoTol: the library removes the most diffuse shell implicated in a "
                                   "near-dependence, from every site of that element and every k, until S is healthy at orthoTol; "
                                   "Build shows which shells went and S's min eigenvalue before and after")
        self.trim_check.clicked.connect(lambda on: self._deck_edit(
            {"basis": {"autoTrim": True}, "solid": {"orthoTol": TRIM_TOL}} if on else {"basis": {"autoTrim": None}, "solid": {"orthoTol": None}}))
        f.addRow("Conditioning", self.trim_check)
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
        start = QtWidgets.QWidget(); sl = QtWidgets.QHBoxLayout(start); sl.setContentsMargins(0, 0, 0, 0)
        self.start_label = QtWidgets.QLabel(); self.start_label.setWordWrap(True)
        self.fresh_btn = QtWidgets.QPushButton("Start from scratch")
        self.fresh_btn.setToolTip("Drop the saved state: the run seeds its density as a new run does")
        self.fresh_btn.clicked.connect(lambda: self._continue_with(None, "Setup: start from scratch"))
        sl.addWidget(self.start_label, 1); sl.addWidget(self.fresh_btn)
        f.addRow("Starts from", start)
        self.acc_box = QtWidgets.QComboBox(); self.acc_box.addItems(ACCELERATORS)
        self.acc_box.activated.connect(lambda _i: self._edit({"accuracy": {"scf": {"accelerator": self.acc_box.currentText()}}}, "accelerator"))
        f.addRow("Accelerator", self.acc_box)
        self.ladder_info = QtWidgets.QLabel(LADDER_INFO); self.ladder_info.setWordWrap(True)
        self.ladder_info.setStyleSheet("color: palette(mid)")
        f.addRow("", self.ladder_info)
        self.ladder_switch = QtWidgets.QComboBox(); self.ladder_switch.addItems(SWITCH_ATS)
        self.ladder_switch.setToolTip("The Ladder hands DIIS over to GDM once |ΔE/E| falls below this; off = only when DIIS stalls")
        self.ladder_switch.activated.connect(lambda i: self._edit({"accuracy": {"scf": {"ladder_switch_at":
            None if i == 0 else 0.0 if i == len(SWITCH_ATS) - 1 else float(SWITCH_ATS[i])}}}, "Ladder switch-over"))
        f.addRow("Ladder → GDM below |ΔE/E|", self.ladder_switch)
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
        self.alpha, self.alpha_min, self.alpha_max = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for box, key, tip in ((self.alpha, "mixing_alpha", "where linear mixing's step α starts"),
                              (self.alpha_min, "mixing_alpha_min", "the floor the adaptive α may shrink to; raise it if α "
                                                                   "collapses and the SCF crawls"),
                              (self.alpha_max, "mixing_alpha_max", "the ceiling the adaptive α may grow to; lower it to "
                                                                   "tamp down an oscillating SCF")):
            box.setRange(0.01, 1.0); box.setSingleStep(0.05); box.setDecimals(2); box.setValue(1.0); box.setToolTip(tip)
            box.editingFinished.connect(lambda b=box, k=key: self._edit({"accuracy": {"scf": {k: round(b.value(), 4) or None}}}, k))
        f.addRow("α start", self.alpha)
        self.alpha_min.setRange(0.0, 1.0); self.alpha_min.setSpecialValueText("default (no floor)")
        f.addRow("α min", self.alpha_min)
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
        self.gates: dict[str, QtWidgets.QComboBox] = {}
        for key, label, tip in GATE_KNOBS:
            box = QtWidgets.QComboBox(); box.addItems(GATE_CHOICES)
            box.setToolTip(f"Converged once every gate that is on is met.  This one: {tip}")
            box.activated.connect(lambda i, k=key, b=box: self._edit({"accuracy": {"scf": {k:
                None if (v := "off" if i == 0 else float(b.currentText())) == GATE_DEFAULTS[k] else v}}}, f"gate {k}"))
            f.addRow(label, box)
            self.gates[key] = box
        self.maxit = QtWidgets.QSpinBox(); self.maxit.setRange(1, 10000); self.maxit.setValue(100)
        self.maxit.setToolTip("A safety net, never the recipe: a run that hits it FAILS")
        self.maxit.editingFinished.connect(lambda: self._edit({"accuracy": {"scf": {"max_iter": self.maxit.value()}}}, "max_iter"))
        f.addRow("Max iterations", self.maxit)

    def _build_grids(self):
        f = self._page("Grids")
        self.grids_planned = QtWidgets.QLabel(); self.grids_planned.setWordWrap(True)
        self.grids_planned.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow("Planned", self.grids_planned)
        self.mesh_radial = QtWidgets.QSpinBox(); self.mesh_radial.setRange(10, 400); self.mesh_radial.setSingleStep(10)
        self.mesh_radial.setToolTip("Radial points per atom of the molecular XC mesh (shown: ε's choice until you set it)")
        self.mesh_angular = QtWidgets.QComboBox(); self.mesh_angular.addItems([str(d) for d in LEBEDEV_DEGREES])
        self.mesh_angular.setToolTip("Lebedev angular degree L per radial shell (the directions grow ~ L²/3: 17 -> 110, 29 -> 302)")
        self.mesh_radial.editingFinished.connect(lambda: self._mesh_edit("mesh_radial", self.mesh_radial.value()))
        self.mesh_angular.activated.connect(lambda _i: self._mesh_edit("mesh_angular", int(self.mesh_angular.currentText())))
        self.mesh_reset = QtWidgets.QPushButton("Back to ε's mesh")
        self.mesh_reset.clicked.connect(lambda: self._mesh_edit(None, None))
        self.mesh_auto = QtWidgets.QLabel()            # in the button's place while ε chooses the mesh
        mesh = QtWidgets.QWidget(); ml = QtWidgets.QHBoxLayout(mesh); ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(QtWidgets.QLabel("radial")); ml.addWidget(self.mesh_radial)
        ml.addWidget(QtWidgets.QLabel("Lebedev L")); ml.addWidget(self.mesh_angular); ml.addWidget(self.mesh_reset)
        ml.addWidget(self.mesh_auto); ml.addStretch(1)
        f.addRow("XC mesh per atom", mesh)
        self.density_ecut = QtWidgets.QDoubleSpinBox(); self.density_ecut.setRange(0.0, 2000.0); self.density_ecut.setDecimals(1)
        self.density_ecut.setSingleStep(10.0); self.density_ecut.setSuffix(" Ha")
        self.density_ecut.setSpecialValueText("the library's floor")
        self.density_ecut.setToolTip("A crystal's density-grid cutoff (deck solid.densityEcut); 0 = the library's floor from the basis")
        self.density_ecut.editingFinished.connect(lambda: self._deck_edit({"solid": {"densityEcut": self.density_ecut.value() or None}}))
        f.addRow("Density Ecut", self.density_ecut)
        self._grids_form = f
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
        self.grids_save = QtWidgets.QPushButton("Save these grids into my deck")
        self.grids_save.setToolTip("the deck keys the raster pilot says restate this run's grids (rasterPilot.restate), merged into "
                                   "the draft's deck: accuracy 0, so a re-run builds these grids without piloting again (undoable)")
        self.grids_save.clicked.connect(self._save_grids)
        f.addRow(self.grids_save)
        self.grids_same = QtWidgets.QLabel(); self.grids_same.setWordWrap(True)
        f.addRow("Reproduced", self.grids_same)

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

    def continue_from(self, run) -> None:
        """Into the draft: `run`'s spec, starting from its saved state -- pick another accelerator (or gates) and ▶ Run."""
        self.show_run(None)
        self._continue_with(states.of(run), f"continue from run {run.summary}", base=run.spec)
        self.tabs.setCurrentIndex(self.tabs.indexOf(self._scf_form.parentWidget()))

    def _continue_with(self, state_file: Optional[str], why: str, base: Optional[RunSpec] = None) -> None:
        spec = base or self.draft.spec
        if spec is None or self._run is not None:
            return
        try:
            self.draft.set(solid2_deck.continued(spec, state_file), "user", why)
        except SpecError as e:
            self.reason.setText(f"not applied: {e}")
            self.refused.emit(str(e))

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

    def _fill_new_menu(self) -> None:
        """New draft's menu, rebuilt on each show (the materials library may have grown)."""
        self.new_menu.clear()
        mol, cry = QtWidgets.QMenu("Molecule", self.new_menu), QtWidgets.QMenu("Crystal", self.new_menu)
        self._new_submenus = (mol, cry)              # held: PySide frees a submenu whose wrapper goes out of scope
        self.new_menu.addMenu(mol); self.new_menu.addMenu(cry)
        for name in molecules.MOLECULES:
            mol.addAction(name, lambda n=name: self.new_draft(molecules.preset_spec(n), f"new draft: preset {n}"))
        if not self._materials:
            self.refresh_materials()
        for k, m in enumerate(self._materials):
            cry.addAction(self.material_box.itemText(k + 1), lambda m=m: self._new_material(m))
        cry.setEnabled(bool(self._materials))

    def _new_material(self, m: dict) -> None:
        try:
            self.new_draft(materials.pick(m), f"new draft: material {m['name']}")
        except Exception as e:                       # noqa: BLE001 -- a malformed user entry: say so, keep the draft
            self.reason.setText(f"{m['name']}: {e}")

    def new_draft(self, spec: RunSpec, why: str) -> None:
        """The draft becomes `spec` (a structure with its default settings), shown editable on the Structure tab."""
        self.show_run(None)
        self.draft.set(spec, "user", why)
        self.tabs.setCurrentIndex(0)

    def _pick_preset(self, name: str) -> None:
        if self._run is None and name in molecules.MOLECULES:
            self.draft.set(molecules.preset_spec(name), "user", f"Setup: preset {name}")

    # -- the materials picker, site spins, Save to my materials ------------------------------------------------------------
    def set_workspace_library(self, path: Optional[str]) -> None:
        """The workspace's own materials library (the app hands it over on open / new workspace); no signal back."""
        self._ws_library = path or None
        self.ws_library_label.setText(path or "none")
        self.ws_library_label.setToolTip(path or "")
        self.ws_library_clear.setEnabled(bool(path))
        self.refresh_materials()

    def _set_ws_library(self, path: Optional[str]) -> None:
        self.set_workspace_library(path)
        self.workspace_library_changed.emit(self._ws_library)

    def _choose_ws_library(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "This workspace's materials library", self._ws_library or "materials.json",
                                                        "materials library (*.json)", options=QtWidgets.QFileDialog.DontConfirmOverwrite)
        if path:
            self._set_ws_library(path)

    def refresh_materials(self) -> None:
        """The picker's list, from the library's layers: shipped, then the user's, then this workspace's."""
        try:
            self._materials = materials.listing(self._ws_library)
            self.material_box.setToolTip("A crystal from the library's materials: the shipped ones, then yours, then this workspace's")
        except materials.MaterialsError as e:
            self._materials = []
            self.material_box.setToolTip(f"no materials: {e}")
            self.reason.setText(f"materials: {e}")
        self.material_box.blockSignals(True)
        self.material_box.clear()
        self.material_box.addItem(_NOT_PRESET)
        for m in self._materials:
            where = {"shipped": [], "user": ["mine"], "workspace": ["workspace"]}[m["layer"]] + (["shadowed"] if m.get("shadowed") else [])
            self.material_box.addItem(m["name"] + (f"  ({', '.join(where)})" if where else ""))
        self.material_box.blockSignals(False)
        self._render_material(self._spec())

    def _render_material(self, s) -> None:
        src = (s.structure.source or {}) if s is not None and s.structure.kind == "crystal" else {}
        name, layer = src.get("materials"), src.get("layer", "shipped")
        i = next((k + 1 for k, m in enumerate(self._materials) if m["name"] == name and m["layer"] == layer), 0)
        self.material_box.blockSignals(True)
        self.material_box.setCurrentIndex(i if i < self.material_box.count() else 0)
        self.material_box.blockSignals(False)

    def _pick_material(self, index: int) -> None:
        if self._run is not None or index <= 0 or index > len(self._materials):
            return
        m = self._materials[index - 1]
        try:
            self.draft.set(materials.pick(m), "user", f"Setup: material {m['name']}")
        except Exception as e:                       # noqa: BLE001 -- a malformed user entry: say so, keep the draft
            self.reason.setText(f"{m['name']}: {e}")

    def set_crystal(self, spec: RunSpec, what: str) -> None:
        """A crystal from outside (File > Import crystal) into the draft: the structure and its default physics."""
        self.show_run(None)
        self.draft.set(spec, "user", what)

    def _on_spins(self) -> None:
        s = self._spec()
        if s is None or s.structure.kind != "crystal" or self._run is not None:
            return
        try:
            m = materials.parse_spins(self.spins_edit.text(), len(s.structure.sites))
        except ValueError as e:
            self.reason.setText(f"not applied: {e}")
            return
        if m == (s.physics.spin.site_moments if s.physics.spin.polarized else None):
            return
        self._edit({"physics": {"spin": {"site_moments": list(m), "polarized": True} if m else {"site_moments": None}}}, "site spins")

    def _save_material(self) -> None:
        s = self._spec()
        if s is None or s.structure.kind != "crystal":
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Save to my materials", "Name (a shipped one is fine: yours shadows it):",
                                                  text=s.label.replace(" ", "_"))
        if not ok or not name.strip():
            return
        name = name.strip()
        into_ws = False
        if self._ws_library:
            where, ok = QtWidgets.QInputDialog.getItem(self, "Save to my materials", "Into:",
                                                       ["My materials (~/.solid2)", "This workspace's library"], 0, False)
            if not ok:
                return
            into_ws = where.startswith("This workspace")
        save = lambda replace: materials.save(name, s, replace=replace, workspace=self._ws_library, into_workspace=into_ws)
        try:
            path = save(False)
        except materials.MaterialsError as e:
            if "replace it" not in str(e):
                self.reason.setText(f"not saved: {e}"); return
            if QtWidgets.QMessageBox.question(self, "Save to my materials", f"'{name}' is already there.  Replace it?") \
                    != QtWidgets.QMessageBox.Yes:
                return
            path = save(True)
        self.refresh_materials()
        if self._run is None:                         # the draft IS that entry now: the picker shows it
            self._edit({"label": name, "structure": {"source": {"materials": name, "layer": "workspace" if into_ws else "user"}}},
                       f"saved as {name}")
        self.reason.setText(f"saved {name} to {path}")

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
    _EDITORS = ("mix_box", "alpha", "alpha_min", "alpha_max", "ladder_switch", "diis_depth", "diis_sv", "diis_engage", "mom_check", "mom_start", "mol_box", "method_box", "basis_box", "pp_check", "pol_check", "mult_spin", "sym_check", "eps_box",
                "acc_box", "maxit", "mesh_radial", "mesh_angular", "density_ecut", "material_box", "spins_edit", "trim_check")

    def _render(self) -> None:
        s = self._spec()
        ro = self._run is not None
        self._rendering = True
        editors = [getattr(self, n) for n in self._EDITORS] + list(self.gates.values())
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
            if not self._materials and self.material_box.count() == 0:
                self.refresh_materials()
            self._render_material(s)
            crystal = s is not None and not mol
            self.spins_edit.setText(materials.spins_text(s) if crystal else "")
            self._sform.setRowVisible(self.spins_edit, crystal)
            self._sform.setRowVisible(self.save_material_btn, crystal and not ro)
            for w in editors:
                w.setEnabled(not ro and s is not None)
            self.material_box.setEnabled(not ro)        # a material starts a draft, even from an empty one
            self.mol_box.setEnabled(not ro)
            for w in (self.method_box, self.pp_check, self.mult_spin, self.sym_check):
                w.setEnabled(not ro and mol)          # a crystal's are fixed by the periodic facade (CRYSTAL_PHYSICS)
            self.basis_box.setEnabled(not ro and s is not None)
            for w in (self.method_box, self.pp_check):
                w.setToolTip(CRYSTAL_PHYSICS if crystal else "")
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
        self._render_policy(s)

    def _render_policy(self, s: Optional[RunSpec]) -> None:
        """The policies that acted on the shown run (or the draft's Build / newest run of its spec), from its record."""
        loud, chosen = policyinfo.lines(self._record(s)[0] if s is not None else None)
        self.policy_loud.setText("\n".join(loud)); self.policy_loud.setVisible(bool(loud))
        self.policy_chosen.setText("\n".join(chosen))
        self._n_chosen = len(chosen)
        self.policy_more.setVisible(bool(chosen)); self.policy_chosen.setVisible(bool(chosen) and self.policy_more.isChecked())
        self._policy_arrow()

    def _policy_arrow(self) -> None:
        n = getattr(self, "_n_chosen", 0)
        self.policy_more.setText(f"{'▾' if self.policy_more.isChecked() else '▸'} {n} automatic choice{'s' if n != 1 else ''} by policy")

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
        self._render_grid_knobs(s)
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
        if mol:
            self.mult_spin.setSpecialValueText(auto_multiplicity(s))
        self.mult_spin.setValue(p.spin.multiplicity or 0)
        o = p.occupations
        self.smearing.setText("integer (no smearing)" if o.smearing == "none" else f"{o.smearing}, kT = {o.kT_Ha} Ha")
        self.hubbard.setText("; ".join(f"U = {h.U_eV} eV on {', '.join(h.sites)} (l = {h.l}, {h.projector})"
                                       for h in p.hubbard) or "none")
        self.relativity.setText(p.relativity)
        self._fill_basis_box(s)
        eps = f"{ac.epsilon_Ha_per_atom:.0e}".replace("e-0", "e-")
        if self.eps_box.findText(eps) < 0:
            self.eps_box.addItem(eps)
        self.eps_box.setCurrentText(eps)
        deck = code_block(s, "solid2").get("deck") or {}
        self.trim_check.setChecked(bool((deck.get("basis") or {}).get("autoTrim")))
        self._bform.setRowVisible(self.trim_check, not mol)   # a molecule's basis is never trimmed here
        self.kmesh.setText("×".join(map(str, ac.kmesh)) if ac.kmesh else ("—" if mol else "not stated"))
        start = solid2_deck.continues_from(s)
        self.start_label.setText(f"the saved state {pathlib.Path(start).name}"
                                 + ("" if os.path.exists(start) else "  <b>(the file is gone: the run will refuse)</b>")
                                 if start else "scratch (the code's own seed)")
        self.start_label.setToolTip(start or "")
        self.fresh_btn.setVisible(bool(start) and self._run is None)
        scf = ac.scf or {}
        self.acc_box.setCurrentText(str(scf.get("accelerator", "Auto")))
        self.mix_box.setCurrentText(str(scf.get("mixer", "Auto")))
        self.alpha.setValue(float(scf.get("mixing_alpha", 1.0)))
        self.alpha_max.setValue(float(scf.get("mixing_alpha_max", 1.0)))
        self.alpha_min.setValue(float(scf.get("mixing_alpha_min") or 0.0))
        linear = self.mix_box.currentText() == "Linear" or (self.mix_box.currentText() == "Auto" and mol)
        for box in (self.alpha, self.alpha_min, self.alpha_max):     # linear mixing's α means nothing to Kerker / Pulay
            self._scf_form.setRowVisible(box, linear)
        self.maxit.setValue(int(scf.get("max_iter", 100)))
        acc = self.acc_box.currentText()
        for w in (self.ladder_info, self.ladder_switch):
            self._scf_form.setRowVisible(w, acc == "Ladder")
        sw = scf.get("ladder_switch_at")
        self.ladder_switch.setCurrentIndex(0 if sw is None else len(SWITCH_ATS) - 1 if sw == 0
                                           else max(0, self.ladder_switch.findText(f"{sw:g}")))
        for w in (self.diis_depth, self.diis_sv, self.diis_engage):   # a DIIS rung runs (a molecule's Auto is DIIS)
            self._scf_form.setRowVisible(w, acc in ("DIIS", "Ladder") or (acc == "Auto" and mol))
        dflt = accelerator_defaults(s, mol)
        self.diis_depth.setSpecialValueText(f"default ({dflt['depth']})")
        self.diis_sv.setItemText(0, f"default ({_sci(dflt['sv'])})")
        self.diis_engage.setItemText(0, f"default ({dflt['engage']})")
        self.diis_depth.setValue(int(scf.get("diis_depth") or 0))
        for box, key in ((self.diis_sv, "diis_sv_tol"), (self.diis_engage, "diis_engage_fd")):
            v = scf.get(key)
            i = box.findText(f"{v:g}") if v else 0
            if v and i < 0:
                box.addItem(f"{v:g}"); i = box.count() - 1
            box.setCurrentIndex(i)
        for key, box in self.gates.items():
            v = scf.get(key, GATE_DEFAULTS[key])
            i = 0 if v == "off" else box.findText(_sci(v))
            if i < 0:
                box.addItem(_sci(v)); i = box.count() - 1
            box.setCurrentIndex(i)
            d = GATE_DEFAULTS[key]
            box.setToolTip(box.toolTip().split("  [")[0] + f"  [default {d if d == 'off' else _sci(d)}"
                           + ("; stated" if key in scf else "") + "]")
        self.mom_check.setChecked(bool(scf.get("mom")))
        self.mom_start.setValue(int(scf.get("mom_start") or 0))
        self._scf_form.setRowVisible(self.mom_start, bool(scf.get("mom")))

    # -- back ends ----------------------------------------------------------------
    # -- the basis: shipped files and the downloaded ones (qviz.bse) ------------------------------------------
    def _fill_basis_box(self, s: RunSpec) -> None:
        """The Basis box: a molecule's shipped bases, then every downloaded set (its path as item data); a crystal's periodic
        bases that cover its elements.  The spec's choice selected (a crystal with none: "choose a basis" first)."""
        basis, file = (s.accuracy.representation or {}).get("basis"), basisinfo.basis_file(s)
        self.basis_box.blockSignals(True)
        self.basis_box.clear()
        if s.structure.kind == "crystal":
            names = solid2_deck.periodic_bases({site.element for site in s.structure.sites})
            self.basis_box.addItems(names)
            if not basis:
                self.basis_box.insertItem(0, _CHOOSE_BASIS); self.basis_box.setCurrentIndex(0)
            else:
                if self.basis_box.findText(basis) < 0:
                    self.basis_box.addItem(basis)
                self.basis_box.setCurrentText(basis)
            self.basis_box.setToolTip("The periodic (pseudopotential valence) bases that cover " + ", ".join(
                sorted({site.element for site in s.structure.sites})) + (": " + ", ".join(names) if names else ": none"))
        else:
            self.basis_box.addItems(BASES)
            for side in bse.installed():
                self.basis_box.addItem(f"{side['name']} v{side['version']} (downloaded)", side["path"])
            if file:
                i = self.basis_box.findData(file)
                if i < 0:                              # the spec names a file this machine does not have
                    self.basis_box.addItem(f"{basis} (missing: {file})", file); i = self.basis_box.count() - 1
                self.basis_box.setCurrentIndex(i)
            elif basis:
                if self.basis_box.findText(basis) < 0:
                    self.basis_box.addItem(basis)
                self.basis_box.setCurrentText(basis)
            self.basis_box.setToolTip(BASIS_TIP)
        self.basis_box.blockSignals(False)
        side = next((x for x in bse.installed() if x["path"] == file), None) if file else None
        self.bse_btn.setEnabled(s.structure.kind == "molecule")
        lines = []
        if side:
            lines += [f"{side['name']} v{side['version']}, Basis Set Exchange, fetched {side['fetched']}"] + bse.notes(side)
            gaps = bse.fit_gaps(s)
            if gaps and (s.physics.theory == "DFT" or s.physics.core == "pseudopotential"):
                lines.append(f"⚠ no Coulomb fit set (A1_coul) covers {', '.join(gaps)}: a DFT run's Hartree fit cannot be built")
        self.bse_note.setText("\n".join(lines) or "none (a shipped basis)")

    def _on_basis_pick(self, _i: int) -> None:
        i = self.basis_box.currentIndex()             # the box's own selection (what the person sees picked)
        file = self.basis_box.itemData(i)
        if self.basis_box.itemText(i) == _CHOOSE_BASIS:
            return
        if self.draft.spec is not None and self.draft.spec.structure.kind == "crystal":
            self._edit({"accuracy": {"representation": {"basis": self.basis_box.itemText(i)}}}, "basis")
            return
        if file:
            side = next((x for x in bse.installed() if x["path"] == file), None)
            key = side["key"] if side else (self.draft.spec.accuracy.representation or {}).get("basis")
            self._edit({"accuracy": {"representation": {"basis": key, "solid2": {"basis_file": file}}}}, f"basis {key} (downloaded)")
        else:
            self._edit({"accuracy": {"representation": {"basis": self.basis_box.itemText(i), "solid2": {"basis_file": None}}}}, "basis")

    def _open_bse(self) -> None:
        s = self.draft.spec
        if s is None:
            return
        from .bse_dialog import BSEDialog
        dlg = BSEDialog({site.Z for site in s.structure.sites}, serial=self._serial, parent=self)
        dlg.downloaded.connect(lambda side: self._edit({"accuracy": {"representation": {
            "basis": side["key"], "solid2": {"basis_file": side["path"]}}}}, f"basis {side['name']} (downloaded)"))
        dlg.exec()

    def _render_basis(self, s: RunSpec) -> None:
        record = self._record(s)[0]
        rec = gridinfo.conditioning(record)              # a run's (or a crystal build's) own: read, never recomputed
        if rec:
            per = "per k" if s.structure.kind == "crystal" else "per irrep"
            lines = [f"overlap {per} (the library's record):"] + rec
        else:
            lines = basisinfo.summary(s)
        self.basis_summary.setText("\n".join(lines + basisinfo.fits(s, record)))
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
        if isinstance(got, Exception):                # a library refusal carries what it filed before refusing
            return getattr(got, "record", None) or None, f"the Build refused: {got}"
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

    def _render_grid_knobs(self, s: RunSpec) -> None:
        mol = s.structure.kind == "molecule"
        dft = s.physics.theory == "DFT" or s.physics.core == "pseudopotential"
        own = (s.accuracy.representation or {}).get("solid2") or {}
        mr, ma = mesh_for(s.accuracy.epsilon_Ha_per_atom)
        mr, ma = int(own.get("mesh_radial", mr)), int(own.get("mesh_angular", ma))
        self.mesh_radial.setValue(mr)
        i = self.mesh_angular.findText(str(ma))
        if i < 0:
            self.mesh_angular.addItem(str(ma)); i = self.mesh_angular.count() - 1
        self.mesh_angular.setCurrentIndex(i)
        stated = "mesh_radial" in own or "mesh_angular" in own
        self.mesh_reset.setVisible(stated)
        self.mesh_auto.setVisible(not stated)
        self.mesh_auto.setText(f"auto, from ε = {s.accuracy.epsilon_Ha_per_atom:g}")
        self._grids_form.setRowVisible(self.mesh_reset.parentWidget(), mol and dft)
        self._grids_form.setRowVisible(self.density_ecut, not mol)
        self.density_ecut.setValue(float(((own.get("deck") or {}).get("solid") or {}).get("densityEcut") or 0.0))

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
        keys, settled = gridinfo.restate(rec)
        self._restate = keys
        self.grids_save.setVisible(keys is not None)
        self.grids_save.setText("Save these grids into my deck" + ("" if settled else " (NOT SETTLED: they miss the target)"))
        same = self._reproduced(s, rec)
        self._grids_form.setRowVisible(self.grids_same, same is not None)
        self.grids_same.setText(same or "")

    def _save_grids(self) -> None:
        """The shown record's restated grids into the draft's deck (a viewed run's spec becomes the draft first)."""
        keys = self._restate
        if keys is None:
            return
        whose = f"run {self._run.summary}" if self._run is not None else "the draft's record"
        if self._run is not None:
            self._copy_to_draft()
        self._edit({"accuracy": {"representation": {"solid2": {"deck": keys}}}}, f"the grids of {whose} (rasterPilot.restate)")

    def _reproduced(self, s: RunSpec, rec: Optional[dict]) -> Optional[str]:
        """When `s`'s deck restates an earlier run's grids: whether this record built the same ones (None: no such run)."""
        if rec is None or s.structure.kind != "crystal":
            return None
        deck = (code_block(s, "solid2").get("deck") or {})
        for other in reversed(self._runs()):
            if other.result is None or other.result.record is rec or not other.result.record:
                continue
            keys, _ = gridinfo.restate(other.result.record)
            if keys is not None and gridinfo.restated_by(deck, keys):
                diff = gridinfo.grid_differences(other.result.record, rec)
                return (f"the same grids as run {other.summary}" if not diff else
                        f"NOT the grids of run {other.summary}: they differ in " + ", ".join(diff))
        return None

    def _mesh_edit(self, key: Optional[str], value) -> None:
        """A molecular XC-mesh setting (representation.solid2), or None: both dropped, ε chooses again."""
        s = self.draft.spec
        if s is None or self._run is not None or self._rendering:
            return
        own = (s.accuracy.representation or {}).get("solid2") or {}
        if key is None:
            patch = {"mesh_radial": None, "mesh_angular": None}           # merge-patch: None deletes
        else:
            mr, ma = mesh_for(s.accuracy.epsilon_Ha_per_atom)       # setting one states both: the pair is the mesh
            patch = {"mesh_radial": int(own.get("mesh_radial", mr)), "mesh_angular": int(own.get("mesh_angular", ma)), key: value}
        self._edit({"accuracy": {"representation": {"solid2": patch}}}, "XC mesh")

    def _deck_edit(self, patch: dict) -> None:
        """A crystal's stated deck setting (representation.solid2.deck), merge-patched (None drops a key)."""
        self._edit({"accuracy": {"representation": {"solid2": {"deck": patch}}}}, "deck " + ", ".join(patch))

    def refresh_records(self) -> None:
        """Re-read the Basis and Grids tabs' records (a run of the shown spec started, streamed its setup, or finished)."""
        s = self._spec()
        if s is not None:
            self._render_basis(s)
            self._render_grids(s)
        self._render_policy(s)

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
        self.build_why.setText(f"Build and SOLID²: {why}" if why and s is not None else "")
        self.build_why.setVisible(bool(why) and s is not None)
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
