"""
qchem viz -- desktop shell (PySide6 + pyvistaqt + pyqtgraph), Workspace{Run} driven.

Default layout (every panel is a dock: move, tab or float it; File > Save Layout keeps yours):

    | Runs · Setup · Output · Agent |     3D view      |
    |  (tabbed, full height)        |------------------|
    |                               |   3D controls    |

  * Runs   -- the workspace's runs; selecting one shows its spec read-only in Setup and its results in Output.
  * Setup  -- what DEFINES a run (qviz.setup_panel): Structure, Hamiltonian, Basis, SCF, Spec (the neutral JSON),
              Codes (each code's resolved settings and native input); the back-end strip and ▶ Run.  Nothing runs
              on its own.  Setup and the Agent are two VIEWS of one SpecDraft (one undo stack, one refusal path).
  * Output -- the selected run's results: Iteration log, SCF graphs (a plot per column of the code's own table),
              Levels (the orbital list beside the MO diagram), SCF log.
  * Agent  -- edits the shared draft with the user; its run requests are cards the user clicks.  Until RunSpec Q4
              is ruled it runs the offline planner: no model is called and nothing leaves the machine.
The back-end strip is option 2.2: every back end that CAN run the spec as stated is enabled, the others are greyed
with their reasons on hover.
"""
from __future__ import annotations
import os, signal, sys, pathlib
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
from qviz.workspace import Workspace, RunStatus
from qviz.backends import default_registry, solid2_deck
from qviz import materials, molecules, states
from qviz.setup_panel import SetupPanel
from qviz.backends.solid2 import _import_qchem_py
from qviz.scfview import SCFView, export_column
from qviz.agent.bridge import AgentBridge
from qviz.agent.client import AnthropicClient, OfflinePlanner
from qviz.agent.consent import ConsentGate, Q4_RULED
from qviz.draft import SpecDraft
from qviz.agent.panel import AgentPanel
from qviz.agent.session import AgentSession
from qviz.agent.tools import Toolset

# -- the back ends (DIP): the Workspace asks the registry by name, never a concrete class ----
REGISTRY = default_registry()
_SOLID2 = REGISTRY.get("SOLID²").capabilities()
if not _SOLID2.installed:
    print(f"SOLID² unavailable ({_SOLID2.why_absent}); analytic stand-in only")

#: bumped when the DEFAULT dock layout changes: a layout saved under another version is not restored
_LAYOUT_VERSION = 2

AUTOSAVE = pathlib.Path.home() / ".local" / "share" / "qchem-GUI" / "autosave.qproj.h5"   # an unnamed workspace, on quit

_STATUS_ICON = {RunStatus.PENDING: "…", RunStatus.RUNNING: "⏳",
                RunStatus.READY: "●", RunStatus.FAILED: "✗"}


class _WsBridge(QtCore.QObject):
    """Marshals the (Qt-agnostic) Workspace observer onto the GUI thread: the model
    may notify from a compute worker, so we bounce it through a signal (auto-queued
    cross-thread) instead of touching widgets off the GUI thread."""
    changed = QtCore.Signal()


class _ComputeTask(QtCore.QRunnable):
    """Runs Run.compute() off the GUI thread, streaming the SCF live: the back end hands
    each SCFRow over DURING the converge (SOLID² releases the GIL between steps), so the
    trace fills in real time and the window stays responsive. Run.compute() catches errors
    -> FAILED status; we relay that as the bridge's done payload.  `bridge` None = no live
    trace (a workspace reopen computes its runs quietly)."""
    def __init__(self, run, bridge=None):
        super().__init__(); self._run = run; self._bridge = bridge
    def run(self):
        if self._bridge:
            self._bridge.started.emit(self._run)
        self._run.compute(on_step=self._bridge.step.emit if self._bridge else None)
        if self._bridge:
            self._bridge.done.emit(self._run.error or "" if self._run.status is RunStatus.FAILED else "")


class _ScfBridge(QtCore.QObject):
    """Marshals the live SCF stream from a compute worker onto the GUI thread. The C++
    observer fires each iteration on the worker; `started` (the Run) resets the trace when the
    queued run actually begins, `step` (queued cross-thread) hands the SCFRow to the
    convergence view (plot tabs + table), `done` (error string, '' = ok) writes the footer."""
    started = QtCore.Signal(object)
    step = QtCore.Signal(object)
    done = QtCore.Signal(str)


class _SampleBridge(QtCore.QObject):
    """Signals a finished off-thread field sample back to the GUI thread (err '' = ok)."""
    done = QtCore.Signal(str)


class _SampleTask(QtCore.QRunnable):
    """Samples a field grid off the GUI thread (the ~2 s n^3 ScalarFunction sweep) and
    stows the result in the shared cache under `key`; then signals done. The dict write
    is atomic under the GIL and only read after `done`, so no lock is needed."""
    def __init__(self, key, sampler, cache, bridge):
        super().__init__(); self._key = key; self._sampler = sampler
        self._cache = cache; self._bridge = bridge
    def run(self):
        try:
            self._cache[self._key] = self._sampler()   # may be None (e.g. spin on an unpolarized run)
            self._bridge.done.emit("")
        except Exception as e:                          # noqa: BLE001 -- surface to the status bar
            self._bridge.done.emit(str(e))



class _FnTask(QtCore.QRunnable):
    """Any library call on the serial compute pool (the library's caches and run record are process globals)."""
    def __init__(self, fn):
        super().__init__(); self._fn = fn
    def run(self):
        self._fn()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, open_path=None):
        super().__init__()
        self._live_run = None                        # the run whose trace is streaming into the SCF view
        self._scf_shown = None                       # the run whose trace the SCF view holds
        self._syncing = False
        self._cam_run = None          # which run the camera is fitted to (reset only on change)
        self._cam_pair = None         # which (left,right) pair the split camera is fitted to
        self._split = False           # is the viewport currently two linked panes?
        self._fld_cache = {}          # GUI-side field cache: key -> ScalarField|None (off-thread sampled)
        self._sample_bridges = []     # in-flight sample bridges, kept alive until they fire
        self.setWindowTitle("qchem viz")
        self.resize(1360, 860)
        self._settings = QtCore.QSettings("qchem-GUI", "desktop")
        self._bridge = _WsBridge()
        # changed carries no payload; supply the current workspace (queued when emitted from a worker)
        self._bridge.changed.connect(lambda: self._on_ws_change(self.ws))
        self._pool = QtCore.QThreadPool()
        self._pool.setMaxThreadCount(1)          # serialize computes (integral cache is a global singleton)
        self._build_ui()
        self._set_workspace(Workspace("Untitled", registry=REGISTRY))
        # open-shell default (on-tenet), DRAFTED only: a run starts on the user's ▶ Run, never on its own
        self.draft.set(molecules.preset_spec("Oxygen (O2, triplet)"), "user", "start-up preset")
        last = str(self._settings.value("workspace/last", "") or "")
        if open_path:
            self._open_workspace_path(open_path)  # CLI: load a .qproj.h5
        elif last and pathlib.Path(last).exists():
            self._open_workspace_path(last)       # the workspace quitting saved: restored, nothing recomputed
        self._default_state = self.saveState(_LAYOUT_VERSION)   # File > Reset Layout returns here
        self._restore_layout()                    # after docks exist

    def _set_workspace(self, ws):
        self.ws = ws
        self.ws.on_change(lambda _w: self._bridge.changed.emit())   # model -> GUI thread, thread-safe
        self.agent.session.tools.workspace = ws                      # the agent reads the runs of THIS workspace

    # -- layout -----------------------------------------------------------
    def _build_ui(self):
        # A stable container is the permanent central widget; the QtInteractor lives INSIDE its
        # stretching layout. Swapping the interactor (single <-> split) then never resizes the
        # central area -- the layout forces the new interactor to fill immediately (a bare
        # setCentralWidget swap leaves it at a zero size hint until a manual resize).
        self._viewport = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self._viewport); lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self._viewport)
        scene.style(self.plotter)
        lay.addWidget(self.plotter.interactor)
        self.setCentralWidget(self._viewport)
        # the left dock area owns the bottom-left corner: the run panels run full height, the 3D controls sit under the view
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setDockOptions(self.dockOptions() | QtWidgets.QMainWindow.AllowTabbedDocks)
        pg.setConfigOptions(antialias=True, background="#101216", foreground="w")

        # Runs
        self.run_list = QtWidgets.QListWidget()
        self.run_list.currentRowChanged.connect(self._on_row_changed)
        self.run_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.run_list.customContextMenuRequested.connect(self._run_menu)
        runs_box = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(runs_box); rl.setContentsMargins(0, 0, 0, 0)
        self.disk_label = QtWidgets.QLabel(); self.disk_label.setWordWrap(True)
        self.disk_label.setStyleSheet("color: palette(mid); padding: 2px 4px")
        self.disk_label.setToolTip(f"Saved SCF states live in {states.state_dir()}: right-click a run to delete its state "
                                   "or remove the run")
        rl.addWidget(self.run_list, 1); rl.addWidget(self.disk_label)
        self._runs_dock = self._dock("Runs", runs_box, QtCore.Qt.LeftDockWidgetArea)

        # Setup: one view of the shared draft (or a run's spec, read-only)
        self.draft = SpecDraft()
        qc, _why = _import_qchem_py()
        self.setup = SetupPanel(self.draft, REGISTRY, qc=qc,
                                legs=lambda run: compare.code_table(compare.legs(self.ws, run)),
                                serial=lambda fn: self._pool.start(_FnTask(fn)),
                                runs=lambda: self.ws.runs)
        self.setup.run_requested.connect(lambda spec, names: [self._add_spec(spec, n) for n in names])
        self.setup.refused.connect(lambda why: self.statusBar().showMessage(f"not applied: {why}"))
        self._setup_dock = self._dock("Setup", self.setup, QtCore.Qt.LeftDockWidgetArea)

        # Output: the selected run's results
        self.output = QtWidgets.QTabWidget(); self.output.setDocumentMode(True)
        # one plot tab per column of the code's own table + the table; the graph the user last clicked is kept across sessions
        self.scf_view = SCFView(self, wanted=str(self._settings.value("output/graph", "") or "") or None)
        self.scf_view.tabs.tabBarClicked.connect(
            lambda i: i >= 0 and self._settings.setValue("output/graph", self.scf_view.tabs.tabText(i)))
        self.output.addTab(self.scf_view.table, "Iteration log")
        self.output.addTab(self.scf_view.tabs, "SCF graphs")
        self.orb_view = QtWidgets.QPlainTextEdit(); self.orb_view.setReadOnly(True)
        self.orb_view.setFont(QtGui.QFont("monospace")); self.orb_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.mo_plot = pg.PlotWidget(title="MO diagram")
        self.mo_plot.setLabel("left", "Energy (Ha)"); self.mo_plot.showGrid(x=False, y=True, alpha=0.2)
        self.mo_plot.setMouseEnabled(x=False, y=True)    # irreps are categorical; zoom the energy axis
        levels = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        levels.addWidget(self.orb_view); levels.addWidget(self.mo_plot)
        levels.setSizes([400, 600])                   # proportional until shown: the list keeps 40% (its hint is ~0)
        self.output.addTab(levels, "Levels")
        self.scf_log = QtWidgets.QPlainTextEdit(); self.scf_log.setReadOnly(True)
        self.scf_log.setFont(QtGui.QFont("monospace")); self.scf_log.setMaximumBlockCount(500)
        self.output.addTab(self.scf_log, "SCF log")
        self.stop_btn = QtWidgets.QPushButton("■ Stop"); self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("Stop the running SCF at its next iteration (an external code's process is terminated)")
        self.stop_btn.clicked.connect(self._stop_live_run)
        self.output.setCornerWidget(self.stop_btn, QtCore.Qt.TopRightCorner)
        self._output_dock = self._dock("Output", self.output, QtCore.Qt.LeftDockWidgetArea)
        # the Output tab the user last CLICKED (kept across sessions); until there is one, a run brings SCF graphs forward
        picked = str(self._settings.value("output/tab", "") or "")
        for i in range(self.output.count()):
            if self.output.tabText(i) == picked:
                self.output.setCurrentIndex(i)
        self.output.tabBarClicked.connect(lambda i: i >= 0 and self._settings.setValue("output/tab", self.output.tabText(i)))

        # Agent -- edits the shared draft; its run requests are cards only the user's click starts
        tools = Toolset(self.draft, REGISTRY, None, start_run=lambda card: [self._add_spec(card.spec, b) for b in card.backends],
                        draft_record=self.setup.draft_record)
        # Claude when ANTHROPIC_API_KEY is set (RunSpec Q4), else the offline planner, which says how to connect
        client = AnthropicClient() if AnthropicClient.why_absent() is None else OfflinePlanner()
        self.agent = AgentPanel(AgentSession(client, tools, ConsentGate(Q4_RULED)), self.draft)
        self._agent_dock = self._dock("Agent", self.agent, QtCore.Qt.LeftDockWidgetArea)
        # Claude Code (on the user's subscription) drives the same tools through scripts/qchem_mcp.py -> this localhost bridge
        try:
            self._bridge_agent = AgentBridge(tools, on_call=self.agent.note_external, on_say=self.agent.note_said,
                                             on_hello=self.agent.note_bridge, parent=self)
            self.agent.note_bridge(f"Claude Code can drive this draft: the qchem-gui MCP server reaches this window "
                                   f"(127.0.0.1:{self._bridge_agent.port})")
        except RuntimeError as e:
            self._bridge_agent = None
            self.agent.note_bridge(str(e))

        for d in (self._setup_dock, self._output_dock, self._agent_dock):
            self.tabifyDockWidget(self._runs_dock, d)
        self._setup_dock.raise_()

        # 3D controls, under the view
        self._controls_dock = self._dock("3D controls", self._build_controls(), QtCore.Qt.BottomDockWidgetArea)
        self.resizeDocks([self._runs_dock], [560], QtCore.Qt.Horizontal)
        self.resizeDocks([self._controls_dock], [300], QtCore.Qt.Vertical)

        m = self.menuBar().addMenu("&File")
        m.addAction("Import molecule (xyz)…", self._import_xyz)
        m.addAction("Import crystal (CIF / POSCAR / extended xyz)…", self._import_crystal)
        m.addAction("Export structure (xyz)…", self._export_xyz)
        m.addAction("Export crystal (CIF / POSCAR / extended xyz)…", self._export_crystal)
        m.addSeparator()
        m.addAction("Open Workspace…", self._open_workspace)
        m.addAction("Save Workspace…", self._save_workspace)
        m.addSeparator()
        m.addAction("Export 3D image…", self._export_image)
        m.addAction("Export SCF plot…", self._export_scf)
        m.addSeparator()
        m.addAction("Save Layout", self._save_layout)
        m.addAction("Reset Layout", self._reset_layout)
        m.addSeparator()
        quit_act = m.addAction("Quit", self.close)
        quit_act.setShortcut(QtGui.QKeySequence.Quit)        # Ctrl+Q (Linux / Windows convention); same as closing the window

    def _build_controls(self) -> QtWidgets.QWidget:
        """The 3D view's controls.  A control shows only when it does something (_sync_controls)."""
        w = QtWidgets.QWidget(); form = QtWidgets.QFormLayout(w); self._controls = form

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
        self.logrho.setToolTip("Plot log₁₀ρ to reveal shell structure (a signed field can't be logged)")
        self.logrho.toggled.connect(self._refresh_3d); form.addRow(self.logrho)

        # compare against another SAME-GEOMETRY run (populated on workspace change).
        # split off -> Δρ overlay in one view; split on -> two linked panes, each its own field.
        self.cmp_box = QtWidgets.QComboBox(); self.cmp_box.currentIndexChanged.connect(self._refresh_3d)
        self._cmp_runs = [None]; form.addRow("Compare vs", self.cmp_box)
        self.split_on = QtWidgets.QCheckBox("split view (two linked panes)")
        self.split_on.setToolTip("Show the selected run and the 'Compare vs' run side by side, "
                                 "cameras linked; unchecked = Δρ difference in one view")
        self.split_on.toggled.connect(self._refresh_3d); form.addRow(self.split_on)

        self.iso = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.iso.setRange(1, 200); self.iso.setValue(40)
        self.iso.valueChanged.connect(self._refresh_3d); form.addRow("Iso level", self.iso)

        self.slice_on = QtWidgets.QCheckBox("show slice"); self.slice_on.toggled.connect(self._refresh_3d)
        form.addRow(self.slice_on)
        self.axis_box = QtWidgets.QComboBox(); self.axis_box.addItems(["Z", "X", "Y"])
        self.axis_box.setToolTip("Plane-normal axis for the slice and the height surface")
        self.axis_box.currentTextChanged.connect(self._refresh_3d); form.addRow("Axis", self.axis_box)
        self.slice = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slice.setRange(0, 100); self.slice.setValue(50)
        self.slice.valueChanged.connect(self._refresh_3d); form.addRow("Slice pos", self.slice)
        self.sweep_on = QtWidgets.QCheckBox("sweep (animate the slice along the axis)")
        self.sweep_on.toggled.connect(self._toggle_sweep); form.addRow(self.sweep_on)

        self.spin_on = QtWidgets.QCheckBox("spin view (rotate the camera)"); self.spin_on.toggled.connect(self._toggle_spin)
        form.addRow(self.spin_on)
        self._spin_timer = QtCore.QTimer(self); self._spin_timer.timeout.connect(self._spin_tick)
        self._sweep_timer = QtCore.QTimer(self); self._sweep_timer.timeout.connect(self._sweep_tick)
        self._sweep_dir = 1
        for c in (self.rep_box, self.field_box, self.cmp_box):
            (c.currentIndexChanged).connect(self._sync_controls)
        self.slice_on.toggled.connect(self._sync_controls)
        self._sync_controls()
        return w

    def _sync_controls(self, *_):
        """Show a control only when it changes the picture: the iso level for a surface, the slice position (and its
        sweep) with a slice or the height surface, log height for an unsigned height surface, split with a partner."""
        rep = self.rep_box.currentText()
        height = rep == "Surface (height)"
        signed = not self.field_box.currentText().startswith("Electron")
        sliced = height or self.slice_on.isChecked()
        f = self._controls
        f.setRowVisible(self.iso, not height)
        f.setRowVisible(self.slice_on, not height)
        f.setRowVisible(self.logrho, height and not signed)
        f.setRowVisible(self.axis_box, sliced)
        f.setRowVisible(self.slice, sliced)
        f.setRowVisible(self.sweep_on, sliced)
        f.setRowVisible(self.split_on, self.cmp_box.currentIndex() > 0)
        if not sliced and self.sweep_on.isChecked():
            self.sweep_on.setChecked(False)

    def _dock(self, title, widget, area):
        d = QtWidgets.QDockWidget(title, self); d.setWidget(widget)
        d.setObjectName(title)                       # REQUIRED for saveState/restoreState to work
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(area, d); return d

    # -- workspace <-> UI --------------------------------------------------
    def _add_spec(self, spec, backend=None):
        """Add a Run on `backend` (None = the registry's default) and compute it OFF the GUI thread -- the window
        stays live; the Run Browser shows … then ⏳ then ● (or ✗).  Serialized via the single-thread pool, so a
        spec ticked on two back ends queues two runs; each one's trace starts when it does (_on_compute_started)."""
        run = self.ws.add_run(spec, backend=backend, compute=False)   # shows immediately as pending
        self.ws.select(run)
        self._compute(run)

    def _compute(self, run):
        """Compute `run` on the serial pool with its live trace (a new run, or a restored one being recomputed)."""
        bridge = _ScfBridge()
        bridge.started.connect(self._on_compute_started)
        bridge.step.connect(self._on_scf_step)
        bridge.done.connect(self._on_compute_done)
        # each queued run's bridge lives until its run settles (a bridge collected early drops its queued signals)
        self._compute_bridges = [(b, r) for b, r in getattr(self, "_compute_bridges", [])
                                 if r.status in (RunStatus.PENDING, RunStatus.RUNNING)] + [(bridge, run)]
        self._pool.start(_ComputeTask(run, bridge))

    def _on_compute_started(self, run):
        self.scf_view.clear()                         # fresh convergence trace for this run
        self._live_run = self._scf_shown = run
        self.stop_btn.setEnabled(True)
        self._output_dock.raise_()                    # the live trace: the Output panel forward
        if not self._settings.value("output/tab", ""):   # a first run: show the graphs; after that, the user's own tab
            self.output.setCurrentWidget(self.scf_view.tabs)
        self._log(f"▶ computing {run.summary}", clear=True)
        try:
            for k, v in run.backend.translate(run.spec).resolved.items():   # the resolved settings, before the run
                self._log(f"  {k}: {v}")
        except Exception as e:                        # noqa: BLE001 -- compute() reports it as FAILED
            self._log(f"  (cannot translate: {e})")
        self.statusBar().showMessage(f"computing {run.summary} … (window stays responsive)")

    def _log(self, msg, clear=False):
        if clear:
            self.scf_log.clear()
        self.scf_log.appendPlainText(msg)

    def _import_xyz(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import molecule (xyz)", "", "XYZ (*.xyz)")
        if not path:
            return
        try:
            numbers, positions, label = molecules.from_xyz(path)
        except Exception as e:                   # noqa: BLE001
            self.statusBar().showMessage(f"xyz import failed: {e}"); return
        self.setup.set_structure(label, numbers, positions)   # into the draft; the user clicks ▶ Run
        self._setup_dock.raise_()

    def _import_crystal(self):
        """The LIBRARY translates the file into a materials entry (qchem.CrystalFiles); the draft becomes that crystal.  What
        it refuses (a partial occupancy, ...) or warns about (a space group it detects differently) is said in its words."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import crystal", "", materials.IMPORT_FILTER)
        if not path:
            return
        try:
            spec, warnings = materials.import_file(path)
        except materials.MaterialsError as e:
            self.statusBar().showMessage(f"crystal import refused: {e}")
            QtWidgets.QMessageBox.warning(self, "Import crystal", str(e)); return
        self.setup.set_crystal(spec, f"import {pathlib.Path(path).name}")
        self._setup_dock.raise_()
        note = f"{spec.label}: {len(spec.structure.sites)} sites imported; state the site spins in Setup > Structure"
        self.statusBar().showMessage(note + (f"  ⚠ {len(warnings)} warning(s)" if warnings else ""))
        for w in warnings:
            self._log(f"⚠ import {pathlib.Path(path).name}: {w}")
        if warnings:
            QtWidgets.QMessageBox.information(self, "Import crystal", "\n\n".join(warnings))

    def _export_crystal(self):
        """The crystal the Setup panel shows, written by the library as CIF / POSCAR / extended xyz (for the oracles)."""
        spec = self.setup._spec()
        if spec is None or spec.structure.kind != "crystal":
            self.statusBar().showMessage("nothing to export: the Setup panel shows no crystal"); return
        filters = ";;".join(f for f, _ in materials.FORMATS.values())
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(self, "Export crystal", f"{spec.label.replace(' ', '_')}.cif", filters)
        if not path:
            return
        fmt = next(k for k, (f, _) in materials.FORMATS.items() if f == chosen)
        try:
            pathlib.Path(path).write_text(materials.export_text(spec, fmt))
        except materials.MaterialsError as e:
            self.statusBar().showMessage(f"export refused: {e}"); return
        self.statusBar().showMessage(f"{spec.label} ({len(spec.structure.sites)} sites) → {path}")

    def _run_menu(self, pos):
        row = self.run_list.indexAt(pos).row()
        if not 0 <= row < len(self.ws.runs):
            return
        run = self.ws.runs[row]
        menu = QtWidgets.QMenu(self)
        idle = run.status not in (RunStatus.PENDING, RunStatus.RUNNING)
        act = menu.addAction("Recompute" + (" (brings back its 3D fields)" if run.restored else ""))
        act.setEnabled(idle)
        saved = states.of(run)
        cont = menu.addAction("Continue from this run…" if saved else
                              "Continue from this run (no saved state: " + ("a molecule cannot save one yet)"
                                                                            if run.spec.structure.kind == "molecule" else "none on disk)"))
        cont.setToolTip("Its spec into the Setup draft, starting from its saved state: change the accelerator or gates, then ▶ Run")
        cont.setEnabled(idle and saved is not None)
        menu.addSeparator()
        drop = menu.addAction(f"Delete saved state ({states.human(states.size(saved))})" if saved else "Delete saved state")
        drop.setEnabled(idle and saved is not None)
        gone = menu.addAction("Remove run" + (" and its saved state" if saved else ""))
        gone.setEnabled(idle)
        got = menu.exec(self.run_list.viewport().mapToGlobal(pos))
        if got is act:
            self.ws.select(run)
            self._compute(run)
        elif got is cont:
            self.setup.continue_from(run)
            self._setup_dock.raise_()
            self.statusBar().showMessage(f"the draft continues from {run.summary}: pick the accelerator / gates, then ▶ Run")
        elif got is drop:
            self._release(run, "deleted")
        elif got is gone:
            if QtWidgets.QMessageBox.question(self, "Remove run", f"Remove {run.summary} from the workspace"
                                              + (" and delete its saved state?" if saved else "?")) \
                    is not QtWidgets.QMessageBox.Yes:
                return
            self._release(run, "removed with the run")
            if self.setup.viewing_run is run:
                self.setup.show_run(None)
            self.ws.remove(run)

    def _release(self, run, how: str) -> None:
        """Delete `run`'s saved state unless another run points at the same file (the same deck)."""
        try:
            p = states.release(run, self.ws.runs)
        except OSError as e:
            self.statusBar().showMessage(f"could not delete the saved state: {e}"); return
        self.statusBar().showMessage(f"saved state {how}: {p}" if p else
                                     "the saved state is kept: another run in this workspace points at the same file")
        self._on_ws_change(self.ws)

    def _autosave(self) -> None:
        """Quitting saves the workspace: to its own file when it has one, else to AUTOSAVE; the next start reopens it."""
        from qviz import project
        path = self.ws.path or str(AUTOSAVE)
        try:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            project.save_workspace(path, self.ws)
            self._settings.setValue("workspace/last", path)
        except Exception as e:                   # noqa: BLE001 -- never block quitting
            print(f"qchem viz: auto-save to {path} failed: {e}", file=sys.stderr)

    def _show_disk(self, ws) -> None:
        """The footer under the Runs list: what the saved states take, and what the disk has left -- a standing reminder."""
        files, used = states.usage(ws.runs)
        try:
            free = states.human(states.free_bytes()) + " free"
        except OSError:
            free = "free space unknown"
        self.disk_label.setText(f"saved states: {files} file{'s' * (files != 1)}, {states.human(used)}  ·  disk {free}")

    def _export_xyz(self):
        """The structure the Setup panel shows (the draft, or the viewed run's) to an .xyz file."""
        spec = self.setup._spec()
        if spec is None:
            self.statusBar().showMessage("nothing to export: the draft is empty"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export structure (xyz)", f"{spec.label}.xyz", "XYZ (*.xyz)")
        if path:
            pathlib.Path(path).write_text(molecules.to_xyz(spec))
            self.statusBar().showMessage(f"{spec.label} ({len(spec.structure.sites)} atoms) → {path}"
                                         + ("  (extended xyz: the cell is in the comment line)" if spec.structure.kind == "crystal" else ""))

    def _on_ws_change(self, ws: Workspace):
        self._syncing = True
        self.run_list.clear()
        sel_row = -1
        for i, r in enumerate(ws.runs):
            e = f"{r.energy:.5f}" if r.energy is not None else "—"
            n = states.size(states.of(r))
            acc = (r.spec.accuracy.scf or {}).get("accelerator")
            cont = f"   ↻ continued{' on ' + acc if acc else ''}" if solid2_deck.continues_from(r.spec) else ""
            self.run_list.addItem(f"{_STATUS_ICON[r.status]}  {r.summary}   E={e}{cont}" + (f"   ⛁ {states.human(n)}" if n else ""))
            if r is ws.selected: sel_row = i
        if sel_row >= 0: self.run_list.setCurrentRow(sel_row)
        self._syncing = False
        self._show_disk(ws)
        sel = ws.selected
        if sel is not None and sel.is_ready:
            self.statusBar().showMessage(f"{sel.summary}:  E = {sel.energy:.6f} Ha")
        elif sel is not None and sel.status is RunStatus.FAILED:
            self.statusBar().showMessage(f"{sel.summary} failed: {sel.error}")
        if sel is not None and sel.is_ready and self._live_run is None and sel is not self._scf_shown:
            self.scf_view.set_rows(sel.scf); self._scf_shown = sel    # a finished run's trace, on selection
        self._update_orbitals(sel)
        if self.setup.viewing_run is not None:        # a run's status / energy moved: its Codes legs re-read
            self.setup.show_run(self.setup.viewing_run if self.setup.viewing_run in ws.runs else None)
        else:
            self.setup.refresh_records()             # the draft reads the newest run of its own spec
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
        lines = [f"{run.summary}   E = {rep.get('energy', 0):.6f} Ha", "",
                 f"{'e (Ha)':>10} {'occ':>4}  MO"]
        for i, L in enumerate(levels[:n_show]):
            tag = f"{L['n']}{L['label']}" if L['label'] != "No Symmetry" else str(L['n'])
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
        def irrep(L):                                 # the lib labels a polarized level "Ag↑": α and β share the irrep's column
            return L["label"].rstrip("↑↓ ")
        order = []                                    # irrep columns, in first-appearance order
        for L in shown:
            if irrep(L) not in order: order.append(irrep(L))
        xmap = {lab: i for i, lab in enumerate(order)}
        for i, L in enumerate(shown):
            x, e, occ = xmap[irrep(L)], float(L["e"]), float(L["occ"])
            if   i == n_occ - 1: col = "#7CFC00"      # HOMO green
            elif i == n_occ:     col = "#ffa500"      # LUMO orange
            elif occ > 0.5:      col = "#4dd0e1"      # occupied cyan
            else:                col = "#7a7a7a"      # virtual grey
            # offset α left / β right within the irrep column so the pair doesn't overlap;
            # narrower level lines (~0.6x) leave room and survive a scrunched panel better
            # +1 / -1 / 0 from the back ends (a project file may hold "up" / "down" / "none")
            spin = {1: "up", -1: "down", 0: "none"}.get(L.get("spin", 0), L.get("spin", "none"))
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
            tag = f"{L['n']}{L['label']}" if L['label'] != "No Symmetry" else str(L['n'])
            lab = pg.TextItem(tag, color="#bbb", anchor=la); lab.setPos(lx, e); p.addItem(lab)
        p.getAxis("bottom").setTicks([[(i, lab) for lab, i in xmap.items()]])
        p.setTitle(f"MO diagram — {run.summary}")
        if shown:                                     # frame the valence cluster (exclude deep core)
            es = [float(L["e"]) for L in shown]
            valence = [e for e in es if e > -5.0] or es
            p.setXRange(-0.9, len(order) - 0.1, padding=0.05)   # room for α (left) / β (right) labels
            p.setYRange(min(valence) - 0.3, max(es) + 0.3, padding=0)

    def _refresh_cmp_box(self):
        """List READY runs that share the selected run's geometry and, with it, give a density (valid Δρ partners)."""
        self._syncing = True
        self.cmp_box.clear(); self.cmp_box.addItem("— none —"); self._cmp_runs = [None]
        sel = self.ws.selected
        if sel is not None and sel.offers("density"):
            for r in self.ws.runs:
                if r is not sel and r.offers("density") and compare.same_geometry(r, sel):
                    self.cmp_box.addItem(compare.method_tag(r)); self._cmp_runs.append(r)
        self._syncing = False

    def _on_row_changed(self, row: int):
        if self._syncing or row < 0 or row >= len(self.ws.runs):
            return
        self.ws.select(self.ws.runs[row])
        self.setup.show_run(self.ws.runs[row])       # the user picked a run: Setup shows what it ran, read-only

    # -- 3D viewport ------------------------------------------------------
    def _partner(self):
        """The 'Compare vs' run, or None if none chosen / not applicable."""
        ci = self.cmp_box.currentIndex()
        return self._cmp_runs[ci] if 0 < ci < len(self._cmp_runs) else None

    def _field_spec(self):
        """The selected run's field for SINGLE view: (run, key, sampler, struct). With a
        Compare-vs partner and split OFF this is the Δρ overlay; otherwise the plain field."""
        run = self.ws.selected
        if run is None or not run.is_ready:
            return None, None, None, None
        if run.restored:                         # reopened from a file: its fields need the live calculation
            self.statusBar().showMessage(f"{run.summary}: restored from the workspace file -- right-click it → "
                                         "Recompute to see its 3D fields")
            return None, None, None, None
        other = self._partner()
        if other is not None and not self.split_on.isChecked():   # Δρ mode (single view)
            return (run, (id(run), "drho", id(other)),
                    (lambda r=run, o=other: compare.difference_density(r, o, n=72)), run.structure())
        return self._field_spec_for(run)

    def _field_spec_for(self, run):
        """Resolve the field_box selection for a GIVEN run into (run, key, sampler, struct)
        WITHOUT the heavy grid sample. `key` is hashable (field-cache key); `sampler` is a
        nullary closure safe to run off the GUI thread -- it reads NO widgets, only the
        captured run + params. The cheap index resolution (report) stays on the GUI thread."""
        struct = run.structure(); rid = id(run)
        t = self.field_box.currentText()
        if t == "Electron density":
            return run, (rid, "density"), (lambda r=run: r.density(n=80)), struct
        if t.startswith("Spin density"):
            return run, (rid, "spin"), (lambda r=run: r.spin_density(n=80)), struct
        # orbital field: map HOMO-k / LUMO+k to an ABSOLUTE MO index via num_occupied
        rep = run.report("orbitals")
        nocc = int(rep.get("num_occupied", 1)); ntot = int(rep.get("num_orbitals", nocc))
        if t == "HOMO":         idx = nocc - 1
        elif t.startswith("HOMO-"): idx = nocc - 1 - int(t.split("-")[1])
        elif t == "LUMO":       idx = nocc
        elif t.startswith("LUMO+"): idx = nocc + int(t.split("+")[1])
        else:                   idx = nocc - 1
        idx = max(0, min(idx, ntot - 1))
        return run, (rid, "orbital", idx), (lambda r=run, i=idx: r.orbital(i, n=72)), struct

    def _refresh_3d(self):
        if self._syncing:                    # skip renders during programmatic list rebuilds
            return
        partner = self._partner()
        sel = self.ws.selected
        want_split = bool(self.split_on.isChecked() and partner is not None
                          and sel is not None and sel.is_ready)
        if want_split != self._split:        # only rebuild the viewport on the transition
            self._set_viewport(want_split)
        if self._split:
            self._refresh_compare(sel, partner); return

        run, key, sampler, struct = self._field_spec()
        if run is None:
            self._render_field(None, None); return
        if key in self._fld_cache:           # already sampled -> render immediately (slider drags land here)
            self._render_field(self._fld_cache[key], struct); return
        # cache miss: the n^3 sample is ~2 s -> do it OFF the GUI thread; show the molecule meanwhile
        self._render_field(None, struct)
        self.statusBar().showMessage("sampling field … (window stays responsive)")
        bridge = _SampleBridge(); self._sample_bridges.append(bridge)
        bridge.done.connect(lambda err, k=key, st=struct, b=bridge: self._on_sample_done(k, st, err, b))
        self._pool.start(_SampleTask(key, sampler, self._fld_cache, bridge))

    def _on_sample_done(self, key, struct, err, bridge):
        if bridge in self._sample_bridges:
            self._sample_bridges.remove(bridge)
        if err:
            self.statusBar().showMessage(f"field sampling failed: {err}"); return
        if self._split:                      # split view redraws both panes from the (now warmer) cache
            self._refresh_3d(); return
        # render only if the finished sample is STILL what's selected (guards rapid field switches)
        run, cur_key, _s, cur_struct = self._field_spec()
        if cur_key != key:
            return
        f = self._fld_cache.get(key)
        if f is None and key[1] == "spin":
            self.statusBar().showMessage("Spin density: this run is unpolarized (ρ↑=ρ↓) -- set multiplicity > 1")
        elif run is not None and run.is_ready:
            self.statusBar().showMessage(f"{run.summary}:  E = {run.energy:.6f} Ha")
        self._render_field(f, cur_struct)

    def _draw_field(self, p, f, struct):
        """Add the field actors for ONE renderer (no clear / camera / render -- the caller
        owns those). Shared by the single view and each pane of the split view."""
        rep = self.rep_box.currentText()
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
            scene.add_structure(p, struct, nuclei=True)
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
        elif struct is not None:                 # no field yet (sampling in flight) -> show the molecule
            scene.add_structure(p, struct)

    def _render_field(self, f, struct):
        p = self.plotter
        rep = self.rep_box.currentText()
        rep_changed = rep != getattr(self, "_last_rep", None)   # switching rep re-frames the view
        keep = (self.ws.selected is self._cam_run) and not rep_changed
        cam = p.camera_position if keep else None
        p.suppress_rendering = True
        try:
            p.clear(); self._draw_field(p, f, struct)
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

    # -- split view: two linked panes -------------------------------------
    def _set_viewport(self, split):
        """Swap the plotter for a (1,2) grid (split) or (1,1) (single). PyVista fixes the
        renderer grid at construction, so a mode change rebuilds the interactor -- inside the
        persistent container layout, so it fills the central area at once (no zero-size flash)."""
        new = QtInteractor(self._viewport, shape=(1, 2) if split else (1, 1))
        scene.style(new)                             # set_background defaults to all renderers
        old = self.plotter
        lay = self._viewport.layout()
        lay.removeWidget(old.interactor)             # detach the old, attach the new in its place
        old.interactor.setParent(None)
        self.plotter = new
        lay.addWidget(new.interactor)
        try: old.close()                             # best-effort VTK teardown, then let Qt reap it
        except Exception: pass                       # noqa: BLE001
        old.interactor.deleteLater()
        self._split = split
        self._cam_run = None; self._cam_pair = None   # force a reframe in the new viewport

    def _refresh_compare(self, left_run, right_run):
        panes = []
        for run in (left_run, right_run):
            _r, key, sampler, struct = self._field_spec_for(run)
            if key in self._fld_cache:
                panes.append((run, self._fld_cache[key], struct))
            else:                                    # sample this pane off-thread; redraw when it lands
                panes.append((run, None, struct))
                bridge = _SampleBridge(); self._sample_bridges.append(bridge)
                bridge.done.connect(lambda err, b=bridge: self._on_compare_sample_done(b))
                self._pool.start(_SampleTask(key, sampler, self._fld_cache, bridge))
        self._draw_compare(panes)

    def _on_compare_sample_done(self, bridge):
        if bridge in self._sample_bridges:
            self._sample_bridges.remove(bridge)
        if self._split:
            self._refresh_3d()                       # both panes redraw from the warmer cache

    def _draw_compare(self, panes):
        p = self.plotter
        pair = tuple(id(r) for r, _f, _s in panes)
        reframe = pair != self._cam_pair
        rep = self.rep_box.currentText()
        p.suppress_rendering = True
        try:
            p.clear()
            for idx, (run, f, struct) in enumerate(panes):
                p.subplot(0, idx)
                self._draw_field(p, f, struct)
                lbl = run.summary + (f"\nE = {run.energy:.4f} Ha" if run.energy is not None else "")
                p.add_text(lbl, font_size=9, color="w")
            p.link_views()                           # one camera drives both panes
        finally:
            p.suppress_rendering = False
        if reframe:
            p.subplot(0, 0)
            if rep == "Surface (height)":
                p.camera_position = "iso"
            p.reset_camera(); p.camera.zoom(0.85)    # propagates to the linked pane
            self._cam_pair = pair
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

    # -- live SCF trace (streamed from the compute worker) ----------------
    def _on_scf_step(self, s):
        self.scf_view.add_row(s)                          # runs on the GUI thread (queued from the worker)
        self.statusBar().showMessage(f"iter {s.iteration}  E = {s.energy:.6f} Ha")

    def _stop_live_run(self):
        run = getattr(self, "_live_run", None)
        if run is not None:
            run.stop()
            self.stop_btn.setEnabled(False)
            self.statusBar().showMessage(f"stopping {run.summary} at its next iteration …")

    def _on_compute_done(self, err):
        self._live_run = None
        self.stop_btn.setEnabled(False)
        if err:
            self.statusBar().showMessage(f"compute failed: {err}"); self._log(f"✗ {err}")
        else:
            rows = self.scf_view.rows; n = len(rows); e = rows[-1].energy if n else float("nan")
            self._log(f"✔ converged in {n} iterations,  E = {e:.8f} Ha")

    def _save_workspace(self):
        from qviz import project
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Workspace", "workspace.qproj.h5", "qviz workspace (*.h5)")
        if path:
            project.save_workspace(path, self.ws)
            self._settings.setValue("workspace/last", path)
            self._title()
            self.statusBar().showMessage(f"saved {len(self.ws.runs)} run(s) → {path}")

    def _open_workspace(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Workspace", "", "qviz workspace (*.h5)")
        if path:
            self._open_workspace_path(path)

    def _open_workspace_path(self, path):
        from qviz import project
        try:
            ws = project.load_workspace(path, REGISTRY, compute=False)   # pending runs
        except Exception as e:                       # noqa: BLE001
            self.statusBar().showMessage(f"open failed: {e}"); return
        self._set_workspace(ws)
        self._cam_run = None
        self._on_ws_change(ws)                        # restored runs show at once
        if ws.selected is not None:                   # Setup agrees with the Runs list: the selected run, its spec the draft
            self.draft.set(ws.selected.spec, "user", f"opened workspace: the selected run {ws.selected.summary}")
            self.setup.show_run(ws.selected)
        todo = [r for r in ws.runs if r.status is RunStatus.PENDING]   # saved without a result: recompute
        for run in todo:                              # off-thread, serialized
            self._pool.start(_ComputeTask(run))
        self._settings.setValue("workspace/last", path)
        self._title()
        self.statusBar().showMessage(f"opened {len(ws.runs)} run(s) from {path}"
                                     + (f"; recomputing {len(todo)} saved without results" if todo else
                                        " (restored: right-click a run → Recompute for its 3D fields)" if ws.runs else ""))

    def _title(self) -> None:
        self.setWindowTitle("qchem viz" + (f" — {pathlib.Path(self.ws.path).name}" if self.ws.path else ""))

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
        title = self.scf_view.current_title if self._scf_shown is r else None
        title = title or next(c.title for row in r.scf for c in row.cells if c.scale == "log")
        export_column(path, r.scf, title, f"{r.summary}   E = {r.energy:.6f} Ha")
        self.statusBar().showMessage(f"exported SCF plot → {path}")

    def _save_layout(self):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState(_LAYOUT_VERSION))
        self.statusBar().showMessage("layout saved (~/.config/qchem-GUI/desktop.conf)")

    def _restore_layout(self):
        geo = self._settings.value("geometry"); st = self._settings.value("windowState")
        if geo is not None: self.restoreGeometry(geo)
        if st is not None:  self.restoreState(st, _LAYOUT_VERSION)   # another version's layout: the default stands

    def _reset_layout(self):
        """Back to the default dock layout (the one _build_ui made, kept before any saved layout was restored)."""
        self.restoreState(self._default_state, _LAYOUT_VERSION)
        self.statusBar().showMessage("layout reset to the default")

    def closeEvent(self, e):
        # A running SCF is asked to stop (it does at its next iteration) and the 3D view releases its VTK window: either
        # left alone kept the process alive after the window closed.  main() then exits without waiting on workers.
        for r in self.ws.runs:
            if r.status in (RunStatus.PENDING, RunStatus.RUNNING):
                r.stop()
        try:
            self.plotter.close()
        except Exception:                        # noqa: BLE001 -- best effort on the way out
            pass
        self._save_layout()                      # persist dock arrangement on exit
        self._autosave()
        if self._bridge_agent is not None:
            self._bridge_agent.close()           # removes the bridge file: Claude Code's calls then say the GUI is closed
        super().closeEvent(e)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # optional workspace file on the command line: python app_desktop.py my.qproj.h5
    open_path = next((a for a in sys.argv[1:] if a.endswith((".h5", ".qproj.h5"))), None)
    w = MainWindow(open_path=open_path); w.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl-C in the launching terminal ends the app (Qt never hands it to Python)
    code = app.exec()
    w._pool.waitForDone(3000)                     # a stopped SCF notices at its next iteration; give it a moment
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)                                # never hang on a still-running library call or a VTK finaliser


if __name__ == "__main__":
    main()
