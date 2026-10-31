# qchem-GUI

Python GUI / visualization frontend for [qchem](https://github.com/JanReimers/qchem).
Flashy scientific plots (shaded isosurfaces, slider-driven slices, vector fields,
live SCF convergence) over a thin, frontend-agnostic data seam.

```
qchem (C++)  ──nanobind──▶  qchem_py.so  ──▶  qviz.ComputeBackend  ──▶  PyVista / pyqtgraph / PySide6
   (separate repo)          (built there)        (the seam, here)          (the apps, here)
```

This repo is **pure Python**. The C++ binding lives in the qchem repo (`pybind/`,
built with `-DQCHEM_PYBIND=ON`) because it must compile against qchem's C++20
modules. We just consume the resulting `qchem_py*.so`.

## Setup

1. **Build the binding in the qchem repo** (once, and whenever the lib API changes):
   ```bash
   cmake -S ~/Code/qchem -B ~/Code/qchem/build/PIC -G Ninja \
     -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DQCHEM_PYBIND=ON \
     -DPython_EXECUTABLE=$PWD/.venv/bin/python
   ninja -C ~/Code/qchem/build/PIC -j4 qchem_py     # -j4: this box is RAM-tight
   ```
2. **Point this repo at that build** (pull forward Release/Debug/PIC at will):
   ```bash
   export QCHEM_BUILD=~/Code/qchem/build/PIC        # dir containing pybind/qchem_py*.so
   ```
   (If unset, `backend_qchem` probes common sibling checkouts as a convenience.)
3. **Create the venv** (Python 3.12 — VTK/PySide6 wheels exist there):
   ```bash
   uv venv -p 3.12 .venv && . .venv/bin/activate
   uv pip install -r requirements.txt
   ```

## Run

The one-line launch on this box (the binding from the lane-g build; checked on the real display 2026-10-31):
```bash
cd ~/Code/qchem-GUI && QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python app_desktop.py
```
On Wayland it forces Qt's xcb platform itself (VTK's render window is X11-only).  CP2K, QE and ABINIT are found at their
usual paths (`CP2K_EXE`, `QE_PW`, `ABINIT_EXE` override) and always launched under `mpirun`; their run data goes to
`~/Code/<code>-runs/<Material>/`.  Without `qchem_py` the SOLID² box in the back-end strip greys out with the reason.

```bash
QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest discover -s tests   # the quick suite
python scripts/make_artifacts.py     # headless: render all plot types -> out/
```

## Layout

- `qviz/spec.py` — the back-end-neutral `RunSpec` (structure, physics, task, ONE accuracy target ε); imports no back end.
- `qviz/draft.py` — `SpecDraft`: the one editable spec; the Inspector and the Agent dock are two views of it.
- `qviz/backends/` — `solid2.py` (in process, `qchem_py`: molecules through MoleculeSetup, crystals through `deck::Build`), `solid2_deck.py` (RunSpec → qchem deck), `cp2k.py`, `qe.py`,
  `abinit.py`, `fake.py`; `base.py` holds the registry and the back-end strip (option 2.2).
- `qviz/runner.py` — the ONE launcher for external codes (`mpirun`, always).
- `qviz/compare.py` — cross-code legs and the Codes table.
- `qviz/agent/` — the agent's tools, loop, consent gate and dock (no model client until RunSpec Q4 is ruled).
- `qviz/inspector.py` — the Inspector widget.  `qviz/scene.py` — the only PyVista-aware module.
- `qviz/data.py` — `Structure` / `ScalarField` / `VectorField` / `SCFRow` (of `SCFCell`s: each code's own SCF columns) records.  `qviz/project.py` — workspace save/restore.
- `qviz/scfview.py` — the Convergence dock: one plot tab per SCF column (log axis for convergence measures), the iteration table beside it.
- `app_desktop.py` — the PySide6 + pyvistaqt + pyqtgraph shell.
- `doc/RunSpecPlan.md` — the plan under execution (stages, findings, the open questions for the user).

## Status

Molecule app is the current focus (see `doc/MoleculeAppPlan.md`). Working today:
ball-and-stick + density/HOMO isosurfaces, ∇ρ field, 2D slice slider, live SCF
trace, HDF5 projects, molecule picker, spin-view. Next big foundation: the
multi-run `Workspace{Run}` model (compare HF vs DFT, AE vs PP).
