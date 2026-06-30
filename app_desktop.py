"""
qchem viz -- desktop shell demo  (PySide6 + pyvistaqt + pyqtgraph)

The recommended app architecture, wired end to end:

    +-------------------------------------------------------------+
    |  File: Open / Save project (HDF5)                           |
    +---------------------------+---------------------------------+
    |                           |   live SCF convergence          |
    |   PyVista 3D viewport     |   (pyqtgraph, streamed)         |
    |   (isosurface + atoms)    +---------------------------------+
    |                           |   controls: field, iso level,   |
    |                           |   slice slider, [Run SCF]       |
    +---------------------------+---------------------------------+

Everything to the *left of the seam* (qviz.ComputeBackend) is analytic today and
becomes your nanobind(qchem) module later -- this file does not change.

Run on a machine with a display:
    python app_desktop.py
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# VTK's interactor embeds a native X11/GLX window, which a native Wayland surface
# can't host (-> BadWindow on X_ConfigureWindow). Force Qt onto xcb/XWayland so
# Qt and VTK agree. Must happen before QApplication is created.
if sys.platform == "linux" and os.environ.get("WAYLAND_DISPLAY") \
        and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from pyvistaqt import QtInteractor

from qviz import AnalyticBackend
from qviz import scene, project
from qviz.scene import field_to_imagedata


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, backend=None):
        super().__init__()
        self.be = backend or AnalyticBackend()
        self.struct = self.be.structure()
        self.fields = {"Electron density": self.be.density(n=80),
                       "HOMO": self.be.orbital(0, n=80)}
        self.scf_steps: list = []
        self.setWindowTitle("qchem viz  -  feasibility demo")
        self.resize(1280, 800)
        self._build_ui()
        self._refresh_3d()

    # -- layout ------------------------------------------------------------
    def _build_ui(self):
        split = QtWidgets.QSplitter(self)
        self.setCentralWidget(split)

        # left: 3D viewport
        self.plotter = QtInteractor(split)
        scene.style(self.plotter)
        split.addWidget(self.plotter.interactor)

        # right: convergence plot + controls
        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right)
        pg.setConfigOptions(antialias=True, background="#101216", foreground="w")
        self.conv = pg.PlotWidget(title="Live SCF convergence")
        self.conv.setLogMode(y=True); self.conv.addLegend()
        self.conv.setLabel("bottom", "iteration"); self.conv.setLabel("left", "residual")
        self.c_dE = self.conv.plot(pen="#4dd0e1", symbol="o", name="|dE|")
        self.c_comm = self.conv.plot(pen="#ff8a65", symbol="s", name="||[F,D]||")
        self.c_drho = self.conv.plot(pen="#aed581", symbol="t", name="||drho||")
        rl.addWidget(self.conv, 3)

        ctl = QtWidgets.QFormLayout()

        # molecule picker (only meaningful with the real qchem backend)
        from qviz import molecules
        self.mol_box = QtWidgets.QComboBox()
        if type(self.be).__name__ == "QChemBackend":
            self.mol_box.addItems(molecules.MOLECULES.keys())
            self.mol_box.currentTextChanged.connect(self._switch_molecule)
        else:
            self.mol_box.addItem("Water (analytic)"); self.mol_box.setEnabled(False)
        ctl.addRow("Molecule", self.mol_box)

        self.field_box = QtWidgets.QComboBox(); self.field_box.addItems(self.fields)
        self.field_box.currentTextChanged.connect(self._refresh_3d)
        ctl.addRow("Field", self.field_box)

        self.iso = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso.setRange(1, 200); self.iso.setValue(40)
        self.iso.valueChanged.connect(self._refresh_3d)
        ctl.addRow("Iso level", self.iso)

        self.slice_on = QtWidgets.QCheckBox("show slice")
        self.slice_on.toggled.connect(self._refresh_3d)
        ctl.addRow(self.slice_on)
        self.slice = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice.setRange(0, 100); self.slice.setValue(50)
        self.slice.valueChanged.connect(self._refresh_3d)
        ctl.addRow("Slice z", self.slice)

        self.spin_on = QtWidgets.QCheckBox("spin view")
        self.spin_on.toggled.connect(self._toggle_spin)
        ctl.addRow(self.spin_on)
        self._spin_timer = QtCore.QTimer(self)
        self._spin_timer.timeout.connect(self._spin_tick)

        self.run_btn = QtWidgets.QPushButton("Run SCF")
        self.run_btn.clicked.connect(self._run_scf)
        ctl.addRow(self.run_btn)
        rl.addLayout(ctl, 1)
        split.addWidget(right)
        split.setSizes([800, 480])

        # menu: project save/restore
        m = self.menuBar().addMenu("&File")
        m.addAction("Open project...", self._open)
        m.addAction("Save project...", self._save)

    # -- spin (auto-rotate the camera) -------------------------------------
    def _toggle_spin(self, on):
        if on:
            self._spin_timer.start(33)        # ~30 fps
        else:
            self._spin_timer.stop()

    def _spin_tick(self):
        self.plotter.camera.Azimuth(1.5)      # pure GL, no recompute
        self.plotter.render()

    # -- molecule switch (rebuilds the qchem backend) ----------------------
    def _switch_molecule(self, name):
        from qviz import molecules
        from qviz.backend_qchem import QChemBackend
        numbers, pos, basis, n = molecules.MOLECULES[name]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage(f"computing {name} ...")
        QtWidgets.QApplication.processEvents()
        try:
            self.be = QChemBackend(numbers, pos, basis)
            self.struct = self.be.structure()
            self.fields = {"Electron density": self.be.density(n=n),
                           "HOMO": self.be.orbital(0, n=n)}
            self.field_box.blockSignals(True)
            self.field_box.clear(); self.field_box.addItems(self.fields)
            self.field_box.blockSignals(False)
            self.scf_steps.clear()
            for c in (self.c_dE, self.c_comm, self.c_drho): c.setData([], [])
            self._refresh_3d()
            self.statusBar().showMessage(f"{name}:  E = {self.be.total_energy():.6f} Ha")
        except Exception as e:
            self.statusBar().showMessage(f"{name} failed: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # -- 3D refresh --------------------------------------------------------
    def _refresh_3d(self):
        f = self.fields[self.field_box.currentText()]
        # Batch the whole rebuild into ONE repaint. Without this, clear() + each
        # add_mesh repaints individually, so the slice briefly vanishes (it's
        # re-added last) -> flicker on every slider move.
        self.plotter.suppress_rendering = True
        try:
            self.plotter.clear()
            if f.signed:
                scene.add_isosurface(self.plotter, f)
            else:
                vmax = float(f.values.max())
                scene.add_isosurface(self.plotter, f,
                                     levels=[self.iso.value() / 1000.0 * vmax])
            scene.add_structure(self.plotter, self.struct)
            if self.slice_on.isChecked():
                grid = field_to_imagedata(f)
                z0, z1 = f.origin[2], f.origin[2] + (f.dims[2]-1)*f.spacing[2]
                zpos = z0 + self.slice.value()/100.0*(z1 - z0)
                sl = grid.slice(normal="z", origin=(0, 0, zpos))
                self.plotter.add_mesh(sl, name="slice", cmap="inferno",
                                      show_scalar_bar=False)
        finally:
            self.plotter.suppress_rendering = False
        self.plotter.render()

    # -- streamed SCF ------------------------------------------------------
    def _run_scf(self):
        self.scf_steps.clear()
        self._gen = self.be.run_scf()
        self.run_btn.setEnabled(False)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._scf_tick)
        self._timer.start(120)            # 120 ms/iter -> visibly "live"

    def _scf_tick(self):
        try:
            self.scf_steps.append(next(self._gen))
        except StopIteration:
            self._timer.stop(); self.run_btn.setEnabled(True); return
        s = self.scf_steps
        it = [x.iteration for x in s]
        self.c_dE.setData(it, [x.dE for x in s])
        self.c_comm.setData(it, [x.commutator for x in s])
        self.c_drho.setData(it, [x.drho for x in s])
        self.statusBar().showMessage(f"iter {s[-1].iteration}  E = {s[-1].energy:.6f} Ha")

    # -- project I/O -------------------------------------------------------
    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project", "water.qproj.h5", "qviz project (*.h5)")
        if path:
            project.save(path, title=self.windowTitle(), backend="AnalyticBackend",
                         structure=self.struct, fields=self.fields,
                         scf=self.scf_steps,
                         viewstate={"iso": self.iso.value()})

    def _open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open project", "", "qviz project (*.h5)")
        if not path:
            return
        p = project.load(path)
        self.struct = p["structure"]; self.fields = p["fields"]
        self.field_box.clear(); self.field_box.addItems(self.fields)
        self._refresh_3d()


def default_backend():
    """Prefer the real qchem backend (computed HF water); fall back to the
    analytic stand-in if the nanobind extension isn't built."""
    try:
        from qviz.backend_qchem import QChemBackend
        be = QChemBackend()
        print(f"qchem backend: real HF/dzvp water, E = {be.total_energy():.6f} Ha")
        return be
    except Exception as e:
        from qviz import AnalyticBackend
        print(f"qchem extension unavailable ({e}); using analytic backend")
        return AnalyticBackend()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(default_backend()); w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
