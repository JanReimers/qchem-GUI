"""
qchem viz -- desktop shell (PySide6 + pyvistaqt + pyqtgraph), Workspace{Run} driven.

Dockable single-window layout (ParaView/Blender idiom, not GaussView MDI):
    Run Browser (left) | 3D viewport (center) | Inspector + SCF convergence (right)

The window holds a Workspace; panels bind to the *selected* Run via the model's
Qt-agnostic observer. Adding a molecule = adding a Run. The compute backend is
injected by a factory (DIP): real qchem if the extension is built, else analytic.
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# VTK's interactor needs X11/GLX; a native Wayland surface can't host it -> force xcb.
if sys.platform == "linux" and os.environ.get("WAYLAND_DISPLAY") \
        and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyvistaqt import QtInteractor

from qviz import scene, compare, geometry
from qviz.scene import field_to_imagedata
from qviz.workspace import Workspace, RunSpec, RunStatus
from qviz import molecules

# -- backend factory (DIP): real qchem if available, else analytic water --------
try:
    from qviz.backend_qchem import QChemBackend
    _HAVE_QCHEM = True
except Exception as _e:                       # extension not built / QCHEM_BUILD unset
    print(f"qchem backend unavailable ({_e}); analytic only")
    _HAVE_QCHEM = False
from qviz.backend_analytic import AnalyticBackend


def make_backend(spec: RunSpec):
    if _HAVE_QCHEM:
        return QChemBackend(list(spec.numbers), list(spec.positions),
                            spec.basis, spec.method, spec.symmetry, spec.multiplicity)
    return AnalyticBackend()                   # geometry-agnostic (water) stand-in


_STATUS_ICON = {RunStatus.PENDING: "…", RunStatus.RUNNING: "⏳",
                RunStatus.READY: "●", RunStatus.FAILED: "✗"}


class _WsBridge(QtCore.QObject):
    """Marshals the (Qt-agnostic) Workspace observer onto the GUI thread: the model
    may notify from a compute worker, so we bounce it through a signal (auto-queued
    cross-thread) instead of touching widgets off the GUI thread."""
    changed = QtCore.Signal()


class _ComputeTask(QtCore.QRunnable):
    """Runs Run.compute() off the GUI thread. The binding releases the GIL during the
    C++ SCF, so the window stays live. Run.compute() catches errors -> FAILED status."""
    def __init__(self, run):
        super().__init__(); self._run = run
    def run(self):
        self._run.compute()


class _ScfBridge(QtCore.QObject):
    """Marshals a live SCF replay from a worker thread onto the GUI thread. The C++
    observer fires each iteration on the worker; `step` (queued cross-thread) hands
    the SCFStep to the plot, `done` (error string, '' = ok) re-enables the button."""
    step = QtCore.Signal(object)
    done = QtCore.Signal(str)


class _ScfTask(QtCore.QRunnable):
    """Streams Run.stream_scf() off the GUI thread, emitting each step through a
    _ScfBridge so the convergence trace fills in real time without freezing the UI.
    `pace` throttles the stream to a watchable cadence -- a tiny molecular SCF
    otherwise converges in <0.1 s. time.sleep releases the GIL, so pacing on the
    worker keeps the window responsive rather than blocking it."""
    def __init__(self, run, bridge, pace=0.08):
        super().__init__(); self._run = run; self._bridge = bridge; self._pace = pace
    def run(self):
        import time
        def on_step(s):
            self._bridge.step.emit(s)
            if self._pace:
                time.sleep(self._pace)
        try:
            self._run.stream_scf(on_step)
            self._bridge.done.emit("")
        except Exception as e:                       # noqa: BLE001 -- surface to the status bar
            self._bridge.done.emit(str(e))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, open_path=None):
        super().__init__()
        self.scf_steps: list = []
        self._syncing = False
        self._cam_run = None          # which run the camera is fitted to (reset only on change)
        self.setWindowTitle("qchem viz")
        self.resize(1360, 860)
        self._settings = QtCore.QSettings("qchem-GUI", "desktop")
        self._bridge = _WsBridge()
        # changed carries no payload; supply the current workspace (queued when emitted from a worker)
        self._bridge.changed.connect(lambda: self._on_ws_change(self.ws))
        self._pool = QtCore.QThreadPool()
        self._pool.setMaxThreadCount(1)          # serialize computes (integral cache is a global singleton)
        self._build_ui()
        self._set_workspace(Workspace("Untitled", make_backend=make_backend))
        if open_path:
            self._open_workspace_path(open_path)  # CLI: load a .qproj.h5 instead of seeding
        else:
            self._on_mol_changed(self.mol_box.currentText())   # sync multiplicity to the default molecule
            self._add_molecule("Oxygen (O2, triplet)")  # open-shell default (polarized primary, on-tenet)
        self._restore_layout()                    # after docks exist + a run is shown

    def _set_workspace(self, ws):
        self.ws = ws
        self.ws.on_change(lambda _w: self._bridge.changed.emit())   # model -> GUI thread, thread-safe

    # -- layout -----------------------------------------------------------
    def _build_ui(self):
        self.plotter = QtInteractor(self)
        scene.style(self.plotter)
        self.setCentralWidget(self.plotter.interactor)

        # left dock: Run Browser + Structure readout (tabbed together)
        self.run_list = QtWidgets.QListWidget()
        self.run_list.currentRowChanged.connect(self._on_row_changed)
        runs_dock = self._dock("Runs", self.run_list, QtCore.Qt.LeftDockWidgetArea)

        self.struct_view = QtWidgets.QPlainTextEdit(); self.struct_view.setReadOnly(True)
        self.struct_view.setFont(QtGui.QFont("monospace"))
        struct_dock = self._dock("Structure", self.struct_view, QtCore.Qt.LeftDockWidgetArea)

        self.orb_view = QtWidgets.QPlainTextEdit(); self.orb_view.setReadOnly(True)
        self.orb_view.setFont(QtGui.QFont("monospace"))
        orb_dock = self._dock("Orbitals", self.orb_view, QtCore.Qt.LeftDockWidgetArea)

        self.tabifyDockWidget(runs_dock, struct_dock)
        self.tabifyDockWidget(struct_dock, orb_dock)
        runs_dock.raise_()

        # right dock (top): Inspector
        insp = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(insp)
        self.mol_box = QtWidgets.QComboBox(); self.mol_box.addItems(molecules.MOLECULES.keys())
        self.method_box = QtWidgets.QComboBox(); self.method_box.addItems(["HF", "LDA", "Xalpha"])
        # the Add button names the method it will use, so it's clear BEFORE you click
        self.add_btn = QtWidgets.QPushButton()
        self.add_btn.clicked.connect(lambda: self._add_molecule(self.mol_box.currentText()))
        self.add_btn.setEnabled(_HAVE_QCHEM)
        self.method_box.currentTextChanged.connect(self._update_add_label)
        row = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
        h.addWidget(self.mol_box); h.addWidget(self.method_box); h.addWidget(self.add_btn)
        form.addRow("Molecule", row)
        self._update_add_label()
        self.basis_box = QtWidgets.QComboBox(); self.basis_box.addItems(["dzvp", "orb1", "dzvp2", "tzvp"])
        self.basis_box.setToolTip("orb1 is the smallest/fastest — use it for bigger molecules (e.g. benzene)")
        form.addRow("Basis", self.basis_box)
        self.mult_spin = QtWidgets.QSpinBox(); self.mult_spin.setRange(0, 7)
        self.mult_spin.setSpecialValueText("auto")   # 0 shows as 'auto' (minimal spin)
        self.mult_spin.setToolTip("Spin multiplicity 2S+1 (auto = minimal). >1 → open-shell, spin-polarized.")
        form.addRow("Multiplicity", self.mult_spin)
        self.mol_box.currentTextChanged.connect(self._on_mol_changed)
        self.sym_check = QtWidgets.QCheckBox("point-group symmetry (SALC → irrep labels)")
        self.sym_check.setChecked(True)          # default on so the orbital table carries irrep labels
        form.addRow(self.sym_check)

        self.field_box = QtWidgets.QComboBox()
        self.field_box.addItems(["Electron density", "Spin density (ρ↑−ρ↓)",
                                 "HOMO", "HOMO-1", "HOMO-2", "HOMO-3",
                                 "LUMO", "LUMO+1", "LUMO+2"])
        self.field_box.currentTextChanged.connect(self._refresh_3d); form.addRow("Field", self.field_box)

        self.rep_box = QtWidgets.QComboBox()
        self.rep_box.addItems(["Isosurface", "Wireframe", "Surface (height)"])
        self.rep_box.setToolTip("Wireframe = see-through mesh; "
                                "Surface (height) = lift the slice plane into a z=f height surface")
        self.rep_box.currentTextChanged.connect(self._refresh_3d); form.addRow("Representation", self.rep_box)

        self.logrho = QtWidgets.QCheckBox("log₁₀ height (density surface)")
        self.logrho.setToolTip("Height surface only: plot log₁₀ρ to reveal shell structure "
                               "(density fields only -- a signed field can't be logged)")
        self.logrho.toggled.connect(self._refresh_3d); form.addRow(self.logrho)

        # Δρ vs another SAME-GEOMETRY run (populated on workspace change)
        self.cmp_box = QtWidgets.QComboBox(); self.cmp_box.currentIndexChanged.connect(self._refresh_3d)
        self._cmp_runs = [None]; form.addRow("Δρ vs", self.cmp_box)

        self.iso = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.iso.setRange(1, 200); self.iso.setValue(40)
        self.iso.valueChanged.connect(self._refresh_3d); form.addRow("Iso level", self.iso)

        self.slice_on = QtWidgets.QCheckBox("show slice"); self.slice_on.toggled.connect(self._refresh_3d)
        self.axis_box = QtWidgets.QComboBox(); self.axis_box.addItems(["Z", "X", "Y"])
        self.axis_box.setToolTip("Plane-normal axis for the slice and the height surface")
        self.axis_box.currentTextChanged.connect(self._refresh_3d)
        _sl = QtWidgets.QHBoxLayout(); _sl.addWidget(self.slice_on); _sl.addStretch(1)
        _sl.addWidget(QtWidgets.QLabel("axis")); _sl.addWidget(self.axis_box)
        form.addRow(_sl)
        self.slice = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slice.setRange(0, 100); self.slice.setValue(50)
        self.slice.valueChanged.connect(self._refresh_3d); form.addRow("Slice pos", self.slice)

        self.spin_on = QtWidgets.QCheckBox("spin view"); self.spin_on.toggled.connect(self._toggle_spin)
        self.sweep_on = QtWidgets.QCheckBox("sweep view"); self.sweep_on.toggled.connect(self._toggle_sweep)
        self.sweep_on.setToolTip("Animate the slice plane back and forth along the selected axis")
        _sv = QtWidgets.QHBoxLayout(); _sv.addWidget(self.spin_on); _sv.addWidget(self.sweep_on); _sv.addStretch(1)
        form.addRow(_sv)
        self._spin_timer = QtCore.QTimer(self); self._spin_timer.timeout.connect(self._spin_tick)
        self._sweep_timer = QtCore.QTimer(self); self._sweep_timer.timeout.connect(self._sweep_tick)
        self._sweep_dir = 1

        self.run_btn = QtWidgets.QPushButton("Replay SCF ▶")
        self.run_btn.setToolTip("Re-animate the already-converged SCF of the selected run "
                                "(the calculation itself runs on '+ Add run').")
        self.run_btn.clicked.connect(self._run_scf)
        form.addRow(self.run_btn)
        insp_dock = self._dock("Inspector", insp, QtCore.Qt.RightDockWidgetArea)

        # right dock (bottom): SCF convergence
        pg.setConfigOptions(antialias=True, background="#101216", foreground="w")
        self.conv = pg.PlotWidget(title="SCF convergence"); self.conv.setLogMode(y=True); self.conv.addLegend()
        self.conv.setLabel("bottom", "iteration"); self.conv.setLabel("left", "residual")
        self.c_dE   = self.conv.plot(pen="#4dd0e1", symbol="o", name="|dE|")
        self.c_comm = self.conv.plot(pen="#ff8a65", symbol="s", name="||[F,D]||")
        self.c_drho = self.conv.plot(pen="#aed581", symbol="t", name="||drho||")
        conv_dock = self._dock("Convergence", self.conv, QtCore.Qt.RightDockWidgetArea)
        self.splitDockWidget(insp_dock, conv_dock, QtCore.Qt.Vertical)

        # right dock: MO energy-level diagram (tabbed with convergence)
        self.mo_plot = pg.PlotWidget(title="MO diagram")
        self.mo_plot.setLabel("left", "Energy (Ha)"); self.mo_plot.showGrid(x=False, y=True, alpha=0.2)
        self.mo_plot.setMouseEnabled(x=False, y=True)    # irreps are categorical; zoom the energy axis
        mo_dock = self._dock("MO diagram", self.mo_plot, QtCore.Qt.RightDockWidgetArea)
        self.tabifyDockWidget(conv_dock, mo_dock); conv_dock.raise_()

        m = self.menuBar().addMenu("&File")
        m.addAction("Import molecule (xyz)…", self._import_xyz)
        m.addSeparator()
        m.addAction("Open Workspace…", self._open_workspace)
        m.addAction("Save Workspace…", self._save_workspace)
        m.addSeparator()
        m.addAction("Export 3D image…", self._export_image)
        m.addAction("Export SCF plot…", self._export_scf)
        m.addSeparator()
        m.addAction("Save Layout", self._save_layout)

    def _dock(self, title, widget, area):
        d = QtWidgets.QDockWidget(title, self); d.setWidget(widget)
        d.setObjectName(title)                       # REQUIRED for saveState/restoreState to work
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d); return d

    # -- workspace <-> UI --------------------------------------------------
    def _update_add_label(self, *_):
        m = self.method_box.currentText()
        self.add_btn.setText(f"+ Add {m} run" if _HAVE_QCHEM else "+ Add run (needs qchem)")

    def _add_spec(self, spec: RunSpec):
        """Add a Run and compute it OFF the GUI thread -- the window stays live; the
        Run Browser shows ⏳ then ● (or ✗). Serialized via the single-thread pool."""
        run = self.ws.add_run(spec, compute=False)   # shows immediately as pending
        self.ws.select(run)
        self.statusBar().showMessage(f"computing {spec.summary} … (window stays responsive)")
        self._pool.start(_ComputeTask(run))

    def _on_mol_changed(self, name):
        if name in molecules.MOLECULES:          # default the multiplicity to the preset's
            self.mult_spin.setValue(molecules.MOLECULES[name][4])

    def _add_molecule(self, name: str):
        Z, pos, basis, n, mult = molecules.MOLECULES[name]
        self._add_spec(RunSpec(label=name.split(" (")[0], numbers=tuple(Z), positions=tuple(pos),
                               basis=self.basis_box.currentText(), method=self.method_box.currentText(),
                               symmetry=self.sym_check.isChecked(), multiplicity=self.mult_spin.value()))

    def _import_xyz(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import molecule (xyz)", "", "XYZ (*.xyz)")
        if not path:
            return
        try:
            numbers, positions, label = molecules.from_xyz(path)
        except Exception as e:                   # noqa: BLE001
            self.statusBar().showMessage(f"xyz import failed: {e}"); return
        self._add_spec(RunSpec(label=label, numbers=tuple(numbers), positions=tuple(positions),
                               basis=self.basis_box.currentText(), method=self.method_box.currentText(),
                               symmetry=self.sym_check.isChecked(), multiplicity=self.mult_spin.value()))

    def _on_ws_change(self, ws: Workspace):
        self._syncing = True
        self.run_list.clear()
        sel_row = -1
        for i, r in enumerate(ws.runs):
            e = f"{r.energy:.5f}" if r.energy is not None else "—"
            self.run_list.addItem(f"{_STATUS_ICON[r.status]}  {r.spec.summary}   E={e}")
            if r is ws.selected: sel_row = i
        if sel_row >= 0: self.run_list.setCurrentRow(sel_row)
        self._syncing = False
        sel = ws.selected
        if sel is not None and sel.is_ready:
            self.statusBar().showMessage(f"{sel.spec.summary}:  E = {sel.energy:.6f} Ha")
        elif sel is not None and sel.status is RunStatus.FAILED:
            self.statusBar().showMessage(f"{sel.spec.summary} failed: {sel.error}")
        self.struct_view.setPlainText(
            f"{sel.spec.summary}\n\n{geometry.report(sel.structure())}"
            if (sel is not None and sel.is_ready) else "")
        self._update_orbitals(sel)
        self._refresh_cmp_box()
        self._refresh_3d()

    def _update_orbitals(self, run):
        if run is None or not run.is_ready:
            self.orb_view.setPlainText(""); self.mo_plot.clear(); return
        rep = run.report("orbitals")
        if not rep or "levels" not in rep:
            self.orb_view.setPlainText("(orbital report unavailable)"); self.mo_plot.clear(); return
        levels = rep["levels"]
        n_occ = sum(1 for L in levels if L["occ"] > 0.5)
        n_show = min(len(levels), n_occ + 3)         # occupied + LUMO..LUMO+2 (high virtuals are noise)
        lines = [f"{run.spec.summary}   E = {rep.get('energy', 0):.6f} Ha", "",
                 f"{'e (Ha)':>10} {'occ':>4}  MO"]
        for i, L in enumerate(levels[:n_show]):
            tag = f"{L['n']}{L['label'].lower()}" if L['label'] != "No Symmetry" else str(L['n'])
            mark = "  ← HOMO" if i == n_occ - 1 else ("  ← LUMO" if i == n_occ else "")
            lines.append(f"{L['e']:10.4f} {L['occ']:4.1f}  {tag}{mark}")
        if len(levels) > n_show:
            lines.append(f"     … {len(levels) - n_show} higher virtual(s) hidden")
        self.orb_view.setPlainText("\n".join(lines))
        self._update_mo_diagram(run, levels, n_occ, n_show)

    def _update_mo_diagram(self, run, levels, n_occ, n_show):
        """2D MO energy diagram: irreps across x, energy up y, ↑↓ for occupation.
        Pan/zoom the energy axis to spread a tight valence cluster (the deep core is
        far below)."""
        p = self.mo_plot; p.clear()
        shown = levels[:n_show]
        order = []                                    # irrep columns, in first-appearance order
        for L in shown:
            if L["label"] not in order: order.append(L["label"])
        xmap = {lab: i for i, lab in enumerate(order)}
        for i, L in enumerate(shown):
            x, e, occ = xmap[L["label"]], float(L["e"]), float(L["occ"])
            if   i == n_occ - 1: col = "#7CFC00"      # HOMO green
            elif i == n_occ:     col = "#ffa500"      # LUMO orange
            elif occ > 0.5:      col = "#4dd0e1"      # occupied cyan
            else:                col = "#7a7a7a"      # virtual grey
            # offset α left / β right within the irrep column so the pair doesn't overlap;
            # narrower level lines (~0.6x) leave room and survive a scrunched panel better
            spin = L.get("spin", "none")             # up/down (polarized) | none (restricted)
            half = 0.15
            if   spin == "up":   xc, lx, la = x - 0.18, x - 0.18 - half - 0.02, (1, 0.5)
            elif spin == "down": xc, lx, la = x + 0.18, x + 0.18 + half + 0.02, (0, 0.5)
            else:                xc, lx, la = x,        x + half + 0.02,        (0, 0.5)
            p.plot([xc - half, xc + half], [e, e], pen=pg.mkPen(col, width=3))
            if   spin == "up":   arrows = "↑" if occ > 0.5 else ""
            elif spin == "down": arrows = "↓" if occ > 0.5 else ""
            else:                arrows = "↑↓" if occ >= 1.5 else ("↑" if occ >= 0.5 else "")
            if arrows:
                t = pg.TextItem(arrows, color=col, anchor=(0.5, 0.5))
                t.setFont(QtGui.QFont("sans", 22)); t.setPos(xc, e); p.addItem(t)
            tag = f"{L['n']}{L['label'].lower()}" if L['label'] != "No Symmetry" else str(L['n'])
            lab = pg.TextItem(tag, color="#bbb", anchor=la); lab.setPos(lx, e); p.addItem(lab)
        p.getAxis("bottom").setTicks([[(i, lab) for lab, i in xmap.items()]])
        p.setTitle(f"MO diagram — {run.spec.summary}")
        if shown:                                     # frame the valence cluster (exclude deep core)
            es = [float(L["e"]) for L in shown]
            valence = [e for e in es if e > -5.0] or es
            p.setXRange(-0.9, len(order) - 0.1, padding=0.05)   # room for α (left) / β (right) labels
            p.setYRange(min(valence) - 0.3, max(es) + 0.3, padding=0)

    def _refresh_cmp_box(self):
        """List READY runs that share the selected run's geometry (valid Δρ partners)."""
        self._syncing = True
        self.cmp_box.clear(); self.cmp_box.addItem("— none —"); self._cmp_runs = [None]
        sel = self.ws.selected
        if sel is not None and sel.is_ready:
            for r in self.ws.runs:
                if r is not sel and r.is_ready and compare.same_geometry(r, sel):
                    self.cmp_box.addItem(f"{r.spec.method}/{r.spec.basis}"); self._cmp_runs.append(r)
        self._syncing = False

    def _on_row_changed(self, row: int):
        if self._syncing or row < 0 or row >= len(self.ws.runs):
            return
        self.ws.select(self.ws.runs[row])

    # -- 3D viewport ------------------------------------------------------
    def _current_field(self):
        run = self.ws.selected
        if run is None or not run.is_ready:
            return None, None
        ci = self.cmp_box.currentIndex()
        if ci > 0 and ci < len(self._cmp_runs):          # Δρ mode (same-geometry partner chosen)
            return compare.difference_density(run, self._cmp_runs[ci], n=72), run.structure()
        t = self.field_box.currentText()
        if t == "Electron density":
            return run.density(n=80), run.structure()
        if t.startswith("Spin density"):
            sf = run.spin_density(n=80)                   # None -> unpolarized run
            if sf is None:
                self.statusBar().showMessage(
                    "Spin density: this run is unpolarized (ρ↑=ρ↓) -- set multiplicity > 1")
            return sf, run.structure()
        # orbital field: map HOMO-k / LUMO+k to an ABSOLUTE MO index via num_occupied
        rep = run.report("orbitals")
        nocc = int(rep.get("num_occupied", 1)); ntot = int(rep.get("num_orbitals", nocc))
        if t == "HOMO":         idx = nocc - 1
        elif t.startswith("HOMO-"): idx = nocc - 1 - int(t.split("-")[1])
        elif t == "LUMO":       idx = nocc
        elif t.startswith("LUMO+"): idx = nocc + int(t.split("+")[1])
        else:                   idx = nocc - 1
        idx = max(0, min(idx, ntot - 1))
        return run.orbital(idx, n=72), run.structure()

    def _refresh_3d(self):
        if self._syncing:                    # skip renders during programmatic list rebuilds
            return
        f, struct = self._current_field()
        p = self.plotter
        rep = self.rep_box.currentText()
        rep_changed = rep != getattr(self, "_last_rep", None)   # switching rep re-frames the view
        keep = (self.ws.selected is self._cam_run) and not rep_changed
        cam = p.camera_position if keep else None
        p.suppress_rendering = True
        try:
            p.clear()
            axis = {"X": 0, "Y": 1, "Z": 2}[self.axis_box.currentText()]
            if f is not None and rep == "Surface (height)":   # standalone height surface (no atoms/slice)
                scene.add_height_surface(p, f, frac=self.slice.value() / 100.0, axis=axis,
                                         log=self.logrho.isChecked())
            elif f is not None:
                if f.signed:                         # orbital / Δρ: slider = ±frac·max|f|
                    scene.add_isosurface(p, f, signed_frac=self.iso.value() / 1000.0,
                                         style="wireframe" if rep == "Wireframe" else "surface")
                else:
                    vmax = float(f.values.max())     # density spans a huge range -> LOG slider:
                    level = vmax * 10.0 ** (-3.0 + (self.iso.value() / 200.0) * 2.5)  # ~0.001..0.3·max
                    scene.add_isosurface(p, f, levels=[level],
                                         style="wireframe" if rep == "Wireframe" else "surface")
                scene.add_structure(p, struct)
                if self.slice_on.isChecked():
                    grid = field_to_imagedata(f)
                    a0 = f.origin[axis]; a1 = a0 + (f.dims[axis]-1)*f.spacing[axis]
                    o = [0.0, 0.0, 0.0]; o[axis] = a0 + self.slice.value()/100.0*(a1 - a0)
                    sl = grid.slice(normal=("x", "y", "z")[axis], origin=tuple(o))
                    if f.signed:                     # diverging + robust symmetric range (skip the core spike)
                        c = float(np.percentile(np.abs(f.values), 99.0)) or 1e-9
                        p.add_mesh(sl, name="slice", cmap="coolwarm", clim=[-c, c], show_scalar_bar=False)
                    else:
                        p.add_mesh(sl, name="slice", cmap="inferno", show_scalar_bar=False)
        finally:
            p.suppress_rendering = False
        if cam is not None:
            p.camera_position = cam                  # keep view across control changes
        else:
            if rep == "Surface (height)":
                p.camera_position = "iso"             # a 3/4 view suits the height surface
            p.reset_camera(); p.camera.zoom(0.85)    # fit the new run + a little breathing room
            self._cam_run = self.ws.selected
        self._last_rep = rep
        p.render()

    def _toggle_spin(self, on):
        self._spin_timer.start(33) if on else self._spin_timer.stop()

    def _spin_tick(self):
        self.plotter.camera.Azimuth(1.5); self.plotter.render()

    def _toggle_sweep(self, on):
        self._sweep_timer.start(60) if on else self._sweep_timer.stop()

    def _sweep_tick(self):
        # drive the slice slider back and forth; setValue re-fires _refresh_3d
        v = self.slice.value() + self._sweep_dir * 2
        if   v >= 100: v, self._sweep_dir = 100, -1
        elif v <= 0:   v, self._sweep_dir = 0, 1
        self.slice.setValue(v)

    # -- streamed SCF (of the selected run), off the GUI thread -----------
    def _run_scf(self):
        run = self.ws.selected
        if run is None or not run.is_ready:
            return
        self.scf_steps.clear()
        self._scf_bridge = _ScfBridge()                  # kept alive on self for the task's lifetime
        self._scf_bridge.step.connect(self._on_scf_step)
        self._scf_bridge.done.connect(self._on_scf_done)
        self.run_btn.setEnabled(False)
        self.statusBar().showMessage("SCF replay running…")
        self._pool.start(_ScfTask(run, self._scf_bridge))   # single-thread pool: serialized vs computes

    def _on_scf_step(self, s):
        self.scf_steps.append(s)                          # runs on the GUI thread (queued from the worker)
        steps = self.scf_steps; it = [x.iteration for x in steps]
        self.c_dE.setData(it, [x.dE for x in steps]); self.c_comm.setData(it, [x.commutator for x in steps])
        self.c_drho.setData(it, [x.drho for x in steps])
        self.statusBar().showMessage(f"iter {s.iteration}  E = {s.energy:.6f} Ha")

    def _on_scf_done(self, err):
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage(f"SCF replay failed: {err}" if err
                                     else f"SCF replay done — {len(self.scf_steps)} iterations")

    def _save_workspace(self):
        from qviz import project
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Workspace", "workspace.qproj.h5", "qviz workspace (*.h5)")
        if path:
            project.save_workspace(path, self.ws)
            self.statusBar().showMessage(f"saved {len(self.ws.runs)} run(s) → {path}")

    def _open_workspace(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Workspace", "", "qviz workspace (*.h5)")
        if path:
            self._open_workspace_path(path)

    def _open_workspace_path(self, path):
        from qviz import project
        try:
            ws = project.load_workspace(path, make_backend, compute=False)   # pending runs
        except Exception as e:                       # noqa: BLE001
            self.statusBar().showMessage(f"open failed: {e}"); return
        self._set_workspace(ws)
        self._cam_run = None
        self._on_ws_change(ws)                        # show the pending runs immediately
        for run in ws.runs:                           # compute off-thread, serialized
            self._pool.start(_ComputeTask(run))
        self.statusBar().showMessage(f"opening {len(ws.runs)} run(s) from {path} …")

    # -- PaperMill v1: publication export ---------------------------------
    def _export_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export 3D image", "figure.png",
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;BMP (*.bmp)")
        if path:
            self.plotter.screenshot(path, scale=3)      # format from extension, 3x resolution
            self.statusBar().showMessage(f"exported 3D image (3×) → {path}")

    def _export_scf(self):
        r = self.ws.selected
        if r is None or not r.is_ready:
            self.statusBar().showMessage("no ready run selected"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export SCF plot", "convergence.pdf", "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)")
        if not path:
            return
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        steps = list(r.run_scf())                        # fresh trace for the selected run
        it = [s.iteration for s in steps]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.semilogy(it, [max(s.dE, 1e-13) for s in steps], "o-", label=r"$|\Delta E|$")
        ax.semilogy(it, [s.commutator for s in steps], "s-", label=r"$\|[F,D]\|$")
        ax.semilogy(it, [s.drho for s in steps], "^-", label=r"$\|\Delta\rho\|$")
        ax.set_xlabel("SCF iteration"); ax.set_ylabel("residual")
        ax.set_title(f"{r.spec.summary}   E = {r.energy:.6f} Ha")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(path); plt.close(fig)
        self.statusBar().showMessage(f"exported SCF plot → {path}")

    def _save_layout(self):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        self.statusBar().showMessage("layout saved (~/.config/qchem-GUI/desktop.conf)")

    def _restore_layout(self):
        geo = self._settings.value("geometry"); st = self._settings.value("windowState")
        if geo is not None: self.restoreGeometry(geo)
        if st is not None:  self.restoreState(st)

    def closeEvent(self, e):
        self._save_layout()                      # persist dock arrangement on exit
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # optional workspace file on the command line: python app_desktop.py my.qproj.h5
    open_path = next((a for a in sys.argv[1:] if a.endswith((".h5", ".qproj.h5"))), None)
    w = MainWindow(open_path=open_path); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
