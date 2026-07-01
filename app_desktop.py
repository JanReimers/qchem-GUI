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
        return QChemBackend(list(spec.numbers), list(spec.positions), spec.basis, spec.method)
    return AnalyticBackend()                   # geometry-agnostic (water) stand-in


_STATUS_ICON = {RunStatus.PENDING: "…", RunStatus.RUNNING: "⏳",
                RunStatus.READY: "●", RunStatus.FAILED: "✗"}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ws = Workspace("Untitled", make_backend=make_backend)
        self.scf_steps: list = []
        self._syncing = False
        self._cam_run = None          # which run the camera is fitted to (reset only on change)
        self.setWindowTitle("qchem viz")
        self.resize(1360, 860)
        self._build_ui()
        self.ws.on_change(self._on_ws_change)
        self._add_molecule("Water (H2O)")       # seed one run so the window isn't empty

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
        self.tabifyDockWidget(runs_dock, struct_dock)
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

        self.field_box = QtWidgets.QComboBox()
        self.field_box.addItems(["Electron density", "HOMO", "HOMO-1", "HOMO-2", "HOMO-3"])
        self.field_box.currentTextChanged.connect(self._refresh_3d); form.addRow("Field", self.field_box)

        # Δρ vs another SAME-GEOMETRY run (populated on workspace change)
        self.cmp_box = QtWidgets.QComboBox(); self.cmp_box.currentIndexChanged.connect(self._refresh_3d)
        self._cmp_runs = [None]; form.addRow("Δρ vs", self.cmp_box)

        self.iso = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.iso.setRange(1, 200); self.iso.setValue(40)
        self.iso.valueChanged.connect(self._refresh_3d); form.addRow("Iso level", self.iso)

        self.slice_on = QtWidgets.QCheckBox("show slice"); self.slice_on.toggled.connect(self._refresh_3d)
        form.addRow(self.slice_on)
        self.slice = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slice.setRange(0, 100); self.slice.setValue(50)
        self.slice.valueChanged.connect(self._refresh_3d); form.addRow("Slice z", self.slice)

        self.spin_on = QtWidgets.QCheckBox("spin view"); self.spin_on.toggled.connect(self._toggle_spin)
        form.addRow(self.spin_on)
        self._spin_timer = QtCore.QTimer(self); self._spin_timer.timeout.connect(self._spin_tick)

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

        m = self.menuBar().addMenu("&File")
        m.addAction("Open Workspace…", self._open_workspace)
        m.addAction("Save Workspace…", self._save_workspace)
        m.addSeparator()
        m.addAction("Export 3D image…", self._export_image)
        m.addAction("Export SCF plot…", self._export_scf)
        m.addSeparator()
        m.addAction("Save Layout", self._save_layout)

    def _dock(self, title, widget, area):
        d = QtWidgets.QDockWidget(title, self); d.setWidget(widget)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d); return d

    # -- workspace <-> UI --------------------------------------------------
    def _update_add_label(self, *_):
        m = self.method_box.currentText()
        self.add_btn.setText(f"+ Add {m} run" if _HAVE_QCHEM else "+ Add run (needs qchem)")

    def _add_molecule(self, name: str):
        Z, pos, basis, n = molecules.MOLECULES[name]
        spec = RunSpec(label=name.split(" (")[0], numbers=tuple(Z), positions=tuple(pos),
                       basis=basis, method=self.method_box.currentText())
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage(f"computing {spec.summary} …"); QtWidgets.QApplication.processEvents()
        try:
            run = self.ws.add_run(spec)          # runs the SCF (synchronous today)
            self.ws.select(run)
            self.statusBar().showMessage(
                f"{spec.summary}: E = {run.energy:.6f} Ha" if run.is_ready
                else f"{spec.summary}: {run.status.value}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if run.is_ready:
            self._run_scf()                      # auto-animate convergence as feedback

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
        self.struct_view.setPlainText(
            f"{sel.spec.summary}\n\n{geometry.report(sel.structure())}"
            if (sel is not None and sel.is_ready) else "")
        self._refresh_cmp_box()
        self._refresh_3d()

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
        if t.startswith("HOMO"):
            k = 0 if t == "HOMO" else int(t.split("-")[1])   # HOMO-k -> occupied index from the top
            return run.orbital(k, n=72), run.structure()
        return run.density(n=80), run.structure()

    def _refresh_3d(self):
        if self._syncing:                    # skip renders during programmatic list rebuilds
            return
        f, struct = self._current_field()
        p = self.plotter
        keep = self.ws.selected is self._cam_run     # same run -> preserve the user's view
        cam = p.camera_position if keep else None
        p.suppress_rendering = True
        try:
            p.clear()
            if f is not None:
                if f.signed:                         # orbital / Δρ: slider = ±frac·max|f|
                    scene.add_isosurface(p, f, signed_frac=self.iso.value() / 1000.0)
                else:
                    vmax = float(f.values.max())
                    scene.add_isosurface(p, f, levels=[self.iso.value() / 1000.0 * vmax])
                scene.add_structure(p, struct)
                if self.slice_on.isChecked():
                    grid = field_to_imagedata(f)
                    z0, z1 = f.origin[2], f.origin[2] + (f.dims[2]-1)*f.spacing[2]
                    zpos = z0 + self.slice.value()/100.0*(z1 - z0)
                    sl = grid.slice(normal="z", origin=(0, 0, zpos))
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
            p.reset_camera(); self._cam_run = self.ws.selected
        p.render()

    def _toggle_spin(self, on):
        self._spin_timer.start(33) if on else self._spin_timer.stop()

    def _spin_tick(self):
        self.plotter.camera.Azimuth(1.5); self.plotter.render()

    # -- streamed SCF (of the selected run) -------------------------------
    def _run_scf(self):
        run = self.ws.selected
        if run is None or not run.is_ready:
            return
        self.scf_steps.clear()
        self._gen = run.run_scf()
        self.run_btn.setEnabled(False)
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self._scf_tick); self._timer.start(120)

    def _scf_tick(self):
        try:
            self.scf_steps.append(next(self._gen))
        except StopIteration:
            self._timer.stop(); self.run_btn.setEnabled(True); return
        s = self.scf_steps; it = [x.iteration for x in s]
        self.c_dE.setData(it, [x.dE for x in s]); self.c_comm.setData(it, [x.commutator for x in s])
        self.c_drho.setData(it, [x.drho for x in s])
        self.statusBar().showMessage(f"iter {s[-1].iteration}  E = {s[-1].energy:.6f} Ha")

    def _save_workspace(self):
        from qviz import project
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Workspace", "workspace.qproj.h5", "qviz workspace (*.h5)")
        if path:
            project.save_workspace(path, self.ws)
            self.statusBar().showMessage(f"saved {len(self.ws.runs)} run(s) → {path}")

    def _open_workspace(self):
        from qviz import project
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Workspace", "", "qviz workspace (*.h5)")
        if not path:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage("opening workspace (recomputing runs) …")
        QtWidgets.QApplication.processEvents()
        try:
            self.ws = project.load_workspace(path, make_backend)
            self.ws.on_change(self._on_ws_change)
            self._cam_run = None
            self._on_ws_change(self.ws)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"opened {len(self.ws.runs)} run(s) from {path}")

    # -- PaperMill v1: publication export ---------------------------------
    def _export_image(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export 3D image", "figure.png", "PNG image (*.png)")
        if path:
            self.plotter.screenshot(path, scale=3)      # 3x window resolution
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
        self._layout = self.saveState()          # stub: Qt serializes dock geometry
        self.statusBar().showMessage("layout saved (in-memory)")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
