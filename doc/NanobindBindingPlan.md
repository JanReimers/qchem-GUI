# nanobind binding plan — qchem C++ → qviz.ComputeBackend

Goal: implement `backend_qchem.QChemBackend(ComputeBackend)` so `app_desktop.py`
shows a **genuinely computed** water density/orbital, replacing `AnalyticBackend`.
Scope (decided): **static first** — Structure + density + orbital + gradient,
read-only, no changes to SCF code. Live SCF streaming is a later pass.

## The recipe (all production libraries — no test harness / gtest)

Lifted from `UnitTests/QchemTesterImp.C` + `M_HF_U.C`, using only production APIs:

```cpp
// 1. geometry (BOHR, experimental water, C2 along z)
Molecule* mol = new Molecule();
mol->Insert(new Atom(8, 0, {0,  0.0,   0.0  }));
mol->Insert(new Atom(1, 0, {0,  1.431, 1.107}));
mol->Insert(new Atom(1, 0, {0, -1.431, 1.107}));

// 2. assemble
Molecule_EC ec(mol->GetNumElectrons());
Real_BS* basis = BasisSet::Molecule::Factory({{"basis","dzvp"}}, mol);
Hamiltonian* ham = Hamiltonian::Factory(Model::HF, Pol::UnPolarized, structure);
auto* acc = SCFAccelerators::Factory(Type::DIIS,
              {{"NProj",4},{"EMax",Z*Z*0.1/32},{"EMin",1e-7},{"SVTol",5e-9}});
SCFIterator scf(basis, &ec, ham, acc, SeedStrategy::Default, mol);

// 3. converge
scf.Iterate({20, 1e-4, 1e-7, 1e-13, 1e-5, 1.0, 1e-4, false});

// 4. extract (everything below IS-A ScalarFunction<double>)
DM_CD*          rho  = scf.GetWaveFunction()->GetChargeDensity();      // operator()(rvec3_t)
const Orbitals* orbs = scf.GetWaveFunction()->GetOrbitals(irrep);      // per BasisSet::GetIrreps(Spin)
// HOMO = highest GetEigenEnergy() among IsOccupied(); dynamic_cast<ScalarFunction<double>*>
```

## Grid sampling lives in C++ (zero-copy out)

One helper does the tight loop (per-point virtual call stays in C++):

```cpp
// returns a flat (nx*ny*nz) buffer, C-order; nanobind hands it to NumPy zero-copy
rvec_t SampleScalar(const ScalarFunction<double>&, rvec3_t origin, rvec3_t spacing, ivec3_t dims);
vec3vec_t SampleGradient(const ScalarFunction<double>&, ... );   // for ∇ρ
```

Maps onto `qviz.data`: Structure / ScalarField / VectorField. Symbols via
`PeriodicTableSaito::GetSymbol(Z)`.

## Build integration

- **nanobind** via CMake `FetchContent` (no new git submodule).
- New target `qchem_py` (nanobind module) under `pybind/`, links the qc* libs
  (qcWaveFunction qcSCFIterator qcSCFAccelerator qcOrbitals qcHamiltonian
  qcChargeDensity qcFitting qcFactory qcMolecule_BS qcBasisSet qcElectronConfig
  qcLASolver qcStructure qcMesh qcSymmetry qcCommon + lapack blas libxc libcint).
- **PIC requirement:** the existing `build/Release` static libs are NOT
  position-independent (346 PC32 relocs), so they can't link into a `.so`.
  Build a separate `build/PIC` with `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`
  (clang-21, Ninja) — `build/Release` stays untouched.

## Status -- STATIC PASS DONE
- [x] interface mapping + production recipe nailed
- [x] PIC rebuild of core libs (build/PIC, -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
- [x] nanobind FetchContent + pybind/CMakeLists + qchem_py target
- [x] binding source: 3-file split (module-unit bridge + flat C ABI + nanobind)
- [x] backend_qchem.QChemBackend + app_desktop.py uses it by default
- [x] VERIFIED: water HF/dzvp E = -76.0229032 Ha (matches M_HF_U_Water anchor);
      real density + HOMO rendered (out/qchem_density.png, out/qchem_homo.png)
- [x] live SCF streaming: SCFIterator observer hook + qcb_run_scf + backend_qchem.run_scf
      -> real 20-iter HF/dzvp water trace (E climbs -71.88 -> -76.0229, [F,D] 6->4e-7,
      DIIS visible ~iter 11). out/qchem_scf_convergence.gif
- [ ] later: more elements in backend_qchem._SYMBOL; DFT model; geometry input UI
- [ ] later: true concurrent streaming (thread + GIL) so the GUI repaints mid-run
      (today: collect-then-replay via the app's per-step timer; fine for fast molecules)

## Live SCF streaming (DONE)
Lib change (src/SCFIterator): a `SCFProgress{iter,energy,dE,commutator,drho}` struct
+ `SetObserver(std::function<void(const SCFProgress&)>)`; the loop fires it each
iteration (right after DisplayEnergies, ungated by Verbose). The bridge's
`qcb_run_scf(cb)` rebuilds a fresh Hamiltonian+accelerator+SCFIterator (so it starts
from the seed), wires the observer to a C callback, and re-converges. nanobind wraps
a Python callable via a captureless trampoline. backend_qchem.run_scf() collects the
push-based callbacks into SCFStep list and yields them (the app's QTimer replays).

## Build the extension
```
cmake -S . -B build/PIC -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DQCHEM_PYBIND=ON -DPython_EXECUTABLE=$PWD/viz-demo/.venv/bin/python \
  -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_Fortran_COMPILER=flang
ninja -C build/PIC qchem_py          # -> build/PIC/pybind/qchem_py.*.so
```
backend_qchem.py adds build/PIC/pybind to sys.path automatically.

## The 3-file binding (why split this way)
You cannot `#include` heavy std headers AND `import` C++20 modules in one plain
TU -- the textual <ranges>/<type_traits> collide with the imported modules'
global-module-fragments (ODR redefinition). So:
- `qchem_bridge.cpp` -- a MODULE UNIT (`export module qchem.bridge;`): its #includes
  live in a GMF that de-dups against imported module GMFs. Imports qchem, runs the
  SCF, samples grids. Exposes a flat `extern "C"` API (C linkage -> plain symbols).
- `qcb_api.h` -- the C ABI (caller-allocated fixed buffers; no STL crosses).
- `qchem_py.cpp` -- pure nanobind, includes only qcb_api.h, never a module.
