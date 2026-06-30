# Molecule App — architecture & feature plan

Scope: **the Molecule app only.** But the four cross-cutting pillars below are
designed once, here, so Solids and the Battery Studio inherit them instead of
re-inventing. Today's demo (`viz-demo/`) is the seed; this is the road from
"spectacular toy" to "the tool you'd pick over GaussView for this workflow."

---

## Part 1 — Four things to architect in NOW

These aren't features; they're shapes imposed on the core data model. Cheap now,
brutal to retrofit. All four converge on **one decision**:

> **The workspace is a collection of comparable `Run`s. Panels are decoupled
> widgets bound to selected Run(s). Everything else consumes that.**

```
Workspace
 ├─ Run  (HF/dzvp, water)        Run = converged Calculation snapshot + metadata
 ├─ Run  (LDA/dzvp, water)              { geometry, method, basis, AE|PP, energy,
 ├─ Run  (HF/dzvp, water, opt)            density, orbitals, convergence trace, … }
 └─ ...
        ▲                ▲                ▲                ▲
   layouts          compare          optimizer         papermill
 (arrange         (overlay/diff    (sweeps params   (export selected
  panels over      N runs)          → makes Runs)    runs/plots/tables)
  the runs)
```

### Pillar 1 — Multiple UI layouts
**Constraint:** no panel may own global state or assume its neighbours exist.
**Design:** every capability is a self-contained **Panel** widget that talks only
to the Workspace/selected-Run via signals (3D viewport, convergence plot, run
list, structure table, spectra, …). A **layout** is just a named arrangement of
docked panels — Qt `QMainWindow::saveState/restoreState` already serializes dock
geometry. Ship presets: `GaussView-like`, `Compact`, `Productivity` (ours),
user-savable. Switching layout reparents panels; it never touches data.

### Pillar 2 — Compare multiple runs  ⟵ the foundational one
**Constraint:** the app is multi-run from line one. A "Run" is first-class,
immutable once converged, and carries enough metadata to be diffed.
**Design:** a `Run` registry in the Workspace. Compare operations are panels that
take ≥2 selected runs: energy/property **table** (HF vs DFT, AE vs PP), **overlaid
convergence**, **side-by-side 3D** (synchronized cameras), **difference density**
Δρ = ρ_A − ρ_B as its own ScalarField. This is why `ComputeBackend` must hand back
*self-describing snapshots*, not mutate one global state.

### Pillar 3 — OptimizerAssistant (multi-factor)
**Constraint:** compute must be **parameterized + batchable**, producing many Runs
without blocking the UI.
**Design:** a driver that takes a parameter space (geometry, basis, functional,
AE/PP, convergence knobs) + an objective, runs jobs (queued — see compute
strategy), and drops each result into the Workspace as a Run. Then Pillar 2
visualizes the sweep. Geometry optimization is the first instance (needs forces —
see compute-gated below). Generic enough to also tune basis/functional.

### Pillar 4 — PaperMill (publication output)
**Constraint:** data must be separable from rendering; every view needs a
*second* render path to a publication target.
**Design:** panels render from structured records (already true). A parallel
exporter re-renders the same records to: vector **PDF/SVG** (2D, via matplotlib),
**hi-res/anti-aliased 3D** stills (PyVista off-screen at arbitrary DPI), and
**tables** to CSV/LaTeX. A report assembler composes selected runs + plots +
tables into one document. Caption/units/styling live with the record, not the
widget.

### Where the pillars sit in the stack
```
Apps        Molecule app  (layouts = arrangements of panels below)
Panels      3D viewport · convergence · run-list · structure-table · spectra · compare · papermill
Workspace   Run registry + selection + diff ops            ← Pillars 1-4 all hang here
Seam        qviz.ComputeBackend  (Run snapshots: Structure/ScalarField/VectorField/SCFStep/…)
Facade      qchem::Calculation   (lib-side build+converge; see APIErgonomicsReview.md)
Core        qchem  (basis/Hamiltonian/SCF/symmetry/DFT/PP …; future: forces/Hessian/TDDFT/NMR)
Compute     interactive (small) ·· batched/queued (heavy) → HDF5 Run → instant open
```

---

## Part 2 — GaussView-derived feature backlog (molecules)

Source: gaussian.com/gv6glance. Filtered to molecule-relevant items, mapped to our
status and staged. **S0** = have it · **S1** = foundation (frontend/facade, no new
core) · **S2** = high-value viz (computable from what core already gives) · **S3**
= compute-gated (needs NEW qchem core — flagged for the lib team) · **S4** =
polish/workflow.

### S0 — already in the demo
- 3D rotate/translate/zoom + auto-spin
- Ball-&-stick display; molecule picker (small set)
- Molecular-orbital isosurfaces (HOMO/MO i); electron-density isosurface
- 2D contour slice with slider; ∇ρ vector field
- Live SCF convergence (ΔE / [F,D] / Δρ)
- HDF5 project save/restore

### S1 — foundation (build the pillars; mostly frontend + the facade)
- **Workspace + Run model & compare**: run list, ΔE/property table, overlaid
  convergence, side-by-side 3D, difference density Δρ  *(Pillars 1–2)*
- **Pluggable layouts**: dock save/restore + GaussView-like / Productivity presets
- **`qchem::Calculation` facade** + run-config UI: method (HF/DFT), basis, charge,
  multiplicity, AE vs PP, symmetry on/off
- **Geometry input**: file import (xyz, Gaussian .gjf/.log/.fchk, PDB, CIF) + picker
- Display formats: wireframe · tube · space-fill (CPK)
- Atom labels (element, serial); structural-parameter readout (pick atoms → bond
  length / angle / dihedral); highlight/hide atoms
- Surface render modes: solid · translucent · wire-mesh; **color a surface by a
  second property** (sets up ESP); multiple synchronized views

### S2 — high-value visualization (computable from current core)
- **ESP-mapped surface** (density colored by electrostatic potential) — the
  signature GaussView image; we have ρ and the nuclear potential
- Spin-density surface (open-shell)
- Atomic charges (Mulliken): color atoms by charge + numeric table
- Dipole-moment vector
- Cube import/export (interop with Gaussian and our own runs)
- **PaperMill v1**: vector-PDF/SVG export of 2D plots + hi-res 3D stills + CSV/LaTeX tables

### S3 — compute-gated (need NEW qchem core; lib-team dependencies)
- **Geometry optimization** + animate the path  ← needs **forces/gradients**
  (also unlocks OptimizerAssistant's first real instance; forces are already on
  the battery north-star critical path — shared win)
- Normal modes / vibrational animation + **IR/Raman** spectra ← needs **Hessian**
- **UV-Vis** spectra ← needs **TDDFT/excited states**
- **NMR** shielding & spectra ← needs **GIAO NMR**
- PES scan / IRC animation ← needs optimizer + constraints

### S4 — workflow & polish
- OptimizerAssistant (full multi-factor) once batch-compute + forces land  *(Pillar 3)*
- PaperMill full report assembler (compose runs + plots + tables → document)  *(Pillar 4)*
- Batch processing / queue / job monitor; in-app animation export (MP4/GIF)
- Interactive molecule builder (fragments, add/delete, clean, symmetrize, isotopes)
- Solvent (PCM) setup + cavity display

---

## Suggested first moves (Molecule app)
1. **Refactor the demo to the Workspace+Run model** (Pillar 2 foundation) — even
   with one run. This is the load-bearing change; do it before more features.
2. **Promote `qchem::Calculation`** (APIErgonomicsReview item 1) so a Run is
   "build calc → snapshot". Bridge shrinks to calling it.
3. **Pluggable layouts** on the now-decoupled panels (Pillar 1).
4. Then S1 viz niceties, then **ESP** (the S2 crowd-pleaser).
S3 items are gated on core work — track them as lib-team requests (forces first,
since battery needs them too).
