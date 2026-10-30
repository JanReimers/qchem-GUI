"""
qviz.backend_qchem -- the REAL ComputeBackend, backed by the nanobind module.

This replaces backend_analytic.AnalyticBackend with genuine qchem output: it
runs a molecular HF SCF in C++ and samples the converged density / HOMO /
gradient onto grids. The frontends (PyVista, the desktop app) are unchanged --
they still only see qviz.data records.

Requires the compiled extension `qchem_py`, built in the SEPARATE qchem repo
(`-DQCHEM_PYBIND=ON`). This GUI repo just consumes the built `.so`: point
`QCHEM_BUILD` at the qchem build dir that contains `pybind/qchem_py*.so`, e.g.

    export QCHEM_BUILD=~/Code/qchem-lane-g/build/Release

The build tree must be configured with -DCMAKE_POSITION_INDEPENDENT_CODE=ON
-DQCHEM_PYBIND=ON. If QCHEM_BUILD is unset we probe a few common sibling-checkout
build dirs as a convenience.

The binding drives qchem's production front door: MoleculeSetup::Resolve +
qchem::Calculation (lane G bridge, 2026-10-30) -- its levels (every MO, virtuals
included), density, spin density and live per-iteration observer.
"""
from __future__ import annotations
import os, sys, pathlib
import numpy as np

from .data import Structure, ScalarField, VectorField, SCFStep, ComputeBackend


def _locate_qchem_py() -> None:
    """Put the dir containing qchem_py*.so on sys.path. QCHEM_BUILD wins; else
    fall back to common sibling qchem-checkout build dirs."""
    cands: list[pathlib.Path] = []
    env = os.environ.get("QCHEM_BUILD")
    if env:
        p = pathlib.Path(env).expanduser()
        cands += [p / "pybind", p]                       # build-dir or its pybind/ subdir
    home = pathlib.Path.home()
    for root in (home/"Code/qchem-lane-g", home/"Code/qchem-main", home/"Code/qchem6"):
        for b in ("build/PIC/pybind", "build/Release/pybind", "build/Debug/pybind"):
            cands.append(root / b)
    for c in cands:
        if c.is_dir() and any(c.glob("qchem_py*.so")):
            sys.path.insert(0, str(c))
            return
    raise ImportError(
        "qchem_py extension not found. Build it in the qchem repo "
        "(-DQCHEM_PYBIND=ON) and set QCHEM_BUILD to that build dir, e.g.\n"
        "    export QCHEM_BUILD=~/Code/qchem-lane-g/build/Release")


_locate_qchem_py()
import qchem_py   # the nanobind extension

# minimal Z -> symbol (extend as needed; the C bridge returns Z, symbols map here)
_SYMBOL = {1: "H", 3: "Li", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na",
           14: "Si", 15: "P", 16: "S", 17: "Cl", 25: "Mn", 27: "Co", 28: "Ni"}


# water, experimental geometry in BOHR (matches UnitTests/M_HF_U.C MakeWater)
WATER_NUMBERS = [8, 1, 1]
WATER_POSITIONS = [0.0, 0.0, 0.0,
                   0.0, 1.431, 1.107,
                   0.0, -1.431, 1.107]


class QChemBackend(ComputeBackend):
    def __init__(self, numbers=None, positions=None, basis="dzvp", method="HF",
                 symmetry=False, multiplicity=0, max_iter=100, on_iteration=None):
        numbers = list(numbers if numbers is not None else WATER_NUMBERS)
        positions = list(np.asarray(positions if positions is not None
                                    else WATER_POSITIONS, float).ravel())
        self._max_iter = max_iter
        # on_iteration(iter, E, dE, [F,D], drho) streams the (single) converge live; the ctor
        # runs the SCF, so this fires DURING construction, not on a later call.
        self._calc = qchem_py.Calculator(numbers, positions, basis=basis, model=method,
                                         symmetry=symmetry, multiplicity=multiplicity,
                                         max_iter=max_iter, on_iteration=on_iteration)
        self._struct = self._read_structure()

    def report(self, kind: str) -> dict:
        """Structured report. 'orbitals': every level (lowest first) + counts, the shape the
        orbital panel reads."""
        if kind != "orbitals":
            return {}
        # the lib's label is "<n><irrep>" (e.g. "5No Symmetry "); the panel numbers levels itself
        levels = [dict(L, n=i + 1, label=L["label"].lstrip("0123456789").strip())
                  for i, L in enumerate(self._calc.levels())]
        nocc = sum(1 for L in levels if L["occ"] > 0.5)
        return {"energy": self._calc.total_energy(), "levels": levels,
                "num_occupied": nocc, "num_orbitals": len(levels)}

    # -- geometry ----------------------------------------------------------
    def _read_structure(self) -> Structure:
        d = self._calc.structure()
        numbers = np.asarray(d["numbers"], int)
        pos = np.asarray(d["positions"], float).reshape(-1, 3)
        return Structure(symbols=[_SYMBOL.get(int(z), "X") for z in numbers],
                         positions=pos, numbers=numbers)

    def structure(self) -> Structure:
        return self._struct

    def total_energy(self) -> float:
        return self._calc.total_energy()

    # -- fields ------------------------------------------------------------
    @staticmethod
    def _scalar(d, name, signed) -> ScalarField:
        return ScalarField(name, np.asarray(d["values"]),
                           tuple(d["origin"]), tuple(d["spacing"]), signed=signed)

    def density(self, n: int = 64, pad: float = 5.0) -> ScalarField:
        return self._scalar(self._calc.density(n, pad), "Electron density", False)

    def spin_density(self, n: int = 64, pad: float = 5.0):
        """Signed ρ↑−ρ↓, or None for an unpolarized run (its m(r) is the zero field --
        nothing to draw)."""
        if not any(L["spin"] for L in self._calc.levels()):
            return None
        return self._scalar(self._calc.spin_density(n, pad), "Spin density (ρ↑−ρ↓)", True)

    def orbital(self, index: int = 0, n: int = 64, pad: float = 5.0) -> ScalarField:
        """MO `index` in levels order (0 = lowest); for a closed shell the HOMO is num_occupied-1."""
        d = self._calc.orbital(index, n, pad)
        return self._scalar(d, f"MO #{index}", bool(d["signed"]))

    def density_gradient(self, n: int = 22, pad: float = 4.0) -> VectorField:
        d = self._calc.gradient(n, pad)
        return VectorField("grad(rho)", np.asarray(d["vectors"]),
                           tuple(d["origin"]), tuple(d["spacing"]))

    # -- SCF: live streaming -----------------------------------------------
    def stream_scf(self, on_step):
        """Re-run the real SCF from the seed, invoking on_step(SCFStep) LIVE as each
        iteration is computed (blocking -- call from a worker thread). The C++ side is
        push-based (an observer fires per iteration) and the binding releases the GIL
        during the solve, re-acquiring it only for the callback -- so streaming this
        off the GUI thread keeps the window responsive while the trace fills in real
        time (vs collecting then replaying)."""
        self._calc.run_scf(lambda it, E, dE, comm, drho:
                           on_step(SCFStep(it, E, dE, comm, drho)), self._max_iter)

    def run_scf(self):
        """Collect-then-yield fallback for non-streaming callers (kept for the
        ComputeBackend generator contract)."""
        steps: list[SCFStep] = []
        self.stream_scf(steps.append)
        yield from steps
