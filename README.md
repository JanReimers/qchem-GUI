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

```bash
python app_desktop.py                # the desktop shell (auto-forces xcb on Wayland)
python scripts/make_artifacts.py     # headless: render all plot types -> out/
```

`app_desktop.py` uses the real qchem backend if `qchem_py` is found, and falls back
to the analytic stand-in (`backend_analytic`) otherwise — so the UI runs even
without a built binding.

## Layout

- `qviz/data.py` — the seam: `Structure` / `ScalarField` / `VectorField` / `SCFStep`
  records + the `ComputeBackend` protocol. **Nothing else leaks across this line.**
- `qviz/backend_qchem.py` — real backend over `qchem_py` (set `QCHEM_BUILD`).
- `qviz/backend_analytic.py` — closed-form water stand-in (no binding needed).
- `qviz/scene.py` — the only PyVista-aware module (records → actors).
- `qviz/project.py` — workspace save/restore on HDF5.
- `qviz/molecules.py` — geometry presets (small species; benzene-class omitted, RAM).
- `app_desktop.py` — PySide6 + pyvistaqt + pyqtgraph shell: 3D viewport, live SCF
  convergence, iso/slice sliders, molecule picker, spin-view, project open/save.
- `doc/MoleculeAppPlan.md` — the architecture + staged GaussView feature backlog.
- `doc/NanobindBindingPlan.md` — how the binding (in the qchem repo) works.

## Status

Molecule app is the current focus (see `doc/MoleculeAppPlan.md`). Working today:
ball-and-stick + density/HOMO isosurfaces, ∇ρ field, 2D slice slider, live SCF
trace, HDF5 projects, molecule picker, spin-view. Next big foundation: the
multi-run `Workspace{Run}` model (compare HF vs DFT, AE vs PP).
