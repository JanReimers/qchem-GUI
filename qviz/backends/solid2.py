"""
qviz.backends.solid2 -- SOLID² (qchem) in process, through the nanobind bridge (doc/RunSpecPlan.md §3.1).

translate(spec) fills the arguments of qchem's production front door (MoleculeSetup::Resolve +
qchem::Calculation, bound by `qchem_py.Calculator`); run() converges on the calling (worker) thread
with the GIL released, streaming each iteration through the facade's constructor observer; the
Result samples density / spin density / orbitals in process, lazily.

Requires the compiled extension `qchem_py`, built in the SEPARATE qchem repo: point QCHEM_BUILD at a
build tree configured with -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DQCHEM_PYBIND=ON, e.g.

    export QCHEM_BUILD=~/Code/qchem-lane-g/build/Release

When the extension is absent the back end still REGISTERS; its capabilities say why it is not
installed, so the back-end strip greys it out instead of the app failing to start.

Scope today (S2): molecules.  A crystal is refused by capability ("SolidSetup not bound yet") until
the bridge binds SolidSetup / SolidSession.
"""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional

import numpy as np

from ..capability import Capabilities, Capability
from ..data import ScalarField, SCFStep, VectorField
from ..spec import RunSpec
from .base import FieldRef, Job, Result, StepSink, code_block, data_structure, shared_representation

NAME = "SOLID²"

# The models the molecular facade has (CalcOptions::model) and the spec's names for them.
_MODEL = {("HF", None): "HF", ("DFT", "LDA"): "LDA", ("DFT", "Xalpha"): "Xalpha"}

# The registered Gaussian basis sets (BasisSet::Gaussian::Factory) -- a stated representation must be one.
BASES = ("dzvp", "dzvp2", "tzvp", "orb", "orb1", "sipp", "sipp_sr", "valence_lowq", "valence_lowq_sph",
         "valence_lowq_sr", "valence_lowq_sr2", "valence_lowq_va", "valence_lowq_vb")
_VALENCE_BASES = tuple(b for b in BASES if b.startswith(("sipp", "valence")))


_REP_KEYS = {"mesh_radial", "mesh_angular"}          # representation["solid2"]: the molecular XC mesh


def mesh_for(eps: float) -> tuple[int, int]:
    """ε -> the molecular XC mesh (radial points, Lebedev degree) per atom.  PROVISIONAL, from one measurement
    (2026-10-30): F₂ GTH/valence_lowq_vb converges to −48.19259 Ha (±3e-5) from 80/29 up; 50/17 is 0.6 mHa/atom
    off, and the facade's own default (30 radial, degree 5 = 12 directions) is 0.52 Ha/atom off (all-electron LDA
    water/dzvp: 20 mHa/atom) -- so the default is never used for a DFT run."""
    return (50, 17) if eps >= 1e-3 else (80, 29) if eps >= 1e-6 else (120, 35)


def basis_for(eps: float) -> str:
    """ε -> the all-electron basis when none is stated.  PROVISIONAL rule of thumb (RunSpecPlan §1.2): replaced by
    a measured ε-calibration curve once one is banked."""
    return "dzvp" if eps >= 1e-4 else "tzvp"


def _import_qchem_py():
    """The extension module, or the reason it is unavailable.  QCHEM_BUILD wins; else common sibling build dirs."""
    cands: list[pathlib.Path] = []
    env = os.environ.get("QCHEM_BUILD")
    if env:
        p = pathlib.Path(env).expanduser()
        cands += [p / "pybind", p]
    home = pathlib.Path.home()
    for root in (home / "Code/qchem-lane-g", home / "Code/qchem-main", home / "Code/qchem6"):
        for b in ("build/Release/pybind", "build/PIC/pybind", "build/Debug/pybind"):
            cands.append(root / b)
    for c in cands:
        if c.is_dir() and any(c.glob("qchem_py*.so")):
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            try:
                import qchem_py
                return qchem_py, None
            except ImportError as e:                   # built for another Python, a missing lib, ...
                return None, f"qchem_py in {c} failed to import: {e}"
    return None, ("qchem_py extension not found: build it in qchem (-DQCHEM_PYBIND=ON "
                  "-DCMAKE_POSITION_INDEPENDENT_CODE=ON) and set QCHEM_BUILD to that build dir")


class Solid2Backend:
    name = NAME

    def __init__(self):
        self._qc, self._why_absent = _import_qchem_py()

    # -- capabilities ------------------------------------------------------
    def capabilities(self) -> Capabilities:
        def scf(s: RunSpec) -> dict:
            return s.accuracy.scf or {}
        rep = shared_representation
        return Capabilities(NAME, installed=self._qc is not None, why_absent=self._why_absent, features=[
            Capability("structure.kind=molecule", lambda s: s.structure.kind == "molecule",
                       "crystals: SolidSetup is not bound in the GUI bridge yet"),
            Capability("physics.theory/functional", lambda s: (s.physics.theory, s.physics.functional
                       if s.physics.theory == "DFT" else None) in _MODEL,
                       "molecular models are HF, LDA and Xalpha"),
            Capability("physics.relativity", lambda s: s.physics.relativity == "none"
                       or (s.physics.relativity == "scalar" and s.physics.core == "pseudopotential"),
                       "non-relativistic all-electron (scalar relativity only through a pseudopotential)"),
            Capability("physics.hubbard=()", lambda s: not s.physics.hubbard, "no molecular DFT+U in the facade"),
            Capability("physics.occupations=none", lambda s: s.physics.occupations.smearing == "none",
                       "no smearing for molecules"),
            Capability("physics.core=pseudopotential needs LDA + a valence basis",
                       lambda s: s.physics.core != "pseudopotential"
                       or (s.physics.functional == "LDA" and rep(s).get("basis") in _VALENCE_BASES),
                       "a molecular pseudopotential run is LDA with a stated valence basis (sipp*, valence_lowq*)"),
            Capability("accuracy.representation keys", lambda s: set(rep(s)) <= {"basis"}
                       and set(code_block(s, "solid2")) <= _REP_KEYS,
                       "a stated representation is a shared basis, or SOLID² settings under representation['solid2'] "
                       "(mesh_radial, mesh_angular)"),
            Capability("accuracy.representation.basis", lambda s: rep(s).get("basis") in (None, *BASES),
                       "the stated basis is not a registered Gaussian basis"),
            Capability("task.kind=energy", lambda s: s.task.kind == "energy",
                       "single points only (MoleculeSession relax is not bound yet)"),
        ])

    # -- translate ----------------------------------------------------------
    def translate(self, spec: RunSpec) -> Job:
        st, ph, ac = spec.structure, spec.physics, spec.accuracy
        scf = ac.scf or {}
        rep = shared_representation(spec)
        own = code_block(spec, "solid2")
        basis = rep.get("basis") or basis_for(ac.epsilon_Ha_per_atom)
        dft = ph.theory == "DFT" or ph.core == "pseudopotential"
        mr, ma = mesh_for(ac.epsilon_Ha_per_atom)
        mesh = (int(own.get("mesh_radial", mr)), int(own.get("mesh_angular", ma))) if dft else (0, 0)
        numbers = [s.Z for s in st.sites]
        positions = [x for s in st.sites for x in s.xyz]
        native = {
            "numbers": numbers, "positions": positions,
            "basis": basis,
            "model": _MODEL[(ph.theory, ph.functional if ph.theory == "DFT" else None)],
            "pp": "gth-lda-valence" if ph.core == "pseudopotential" else "",
            "symmetry": bool(scf.get("impose_symmetry", False)),
            "polarized": bool(ph.spin.polarized),
            "multiplicity": int(ph.spin.multiplicity or 0),
            "max_iter": int(scf.get("max_iter", 100)),
            "accelerator": str(scf.get("accelerator", "DIIS")),
            "mesh_radial": mesh[0], "mesh_angular": mesh[1],
        }
        resolved = {
            "basis": f"{basis} ({'stated' if rep.get('basis') else 'from ε, provisional'})",
            "model": native["model"],
            "pseudopotential": native["pp"] or "all-electron",
            "spin": ("polarized" if native["polarized"] or native["multiplicity"] > 1 else "unpolarized")
                    + (f", 2S+1={native['multiplicity']}" if native["multiplicity"] else ""),
            "point-group blocking": native["symmetry"],
            "accelerator": native["accelerator"],
            "XC mesh": (f"{mesh[0]} radial x Lebedev {mesh[1]} per atom"
                        + (" stated" if own else " from ε, provisional")) if dft else "none (HF)",
            "SCF gates": "facade defaults ([F,D] 1e-5, Δρ/e 1e-4); ε not yet mapped (provisional)",
            "max_iter": native["max_iter"],
        }
        return Job(NAME, spec, native=native, resolved=resolved)

    # -- run + parse --------------------------------------------------------
    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result:
        if self._qc is None:
            raise RuntimeError(self._why_absent)
        n = job.native
        steps: list[SCFStep] = []

        def step(it, E, dE, comm, drho):
            s = SCFStep(it, E, dE, comm, drho)
            steps.append(s)
            if on_step:
                on_step(s)

        calc = self._qc.Calculator(n["numbers"], n["positions"], basis=n["basis"], model=n["model"],
                                   symmetry=n["symmetry"], polarized=n["polarized"],
                                   multiplicity=n["multiplicity"], max_iter=n["max_iter"], pp=n["pp"],
                                   accelerator=n["accelerator"], mesh_radial=n["mesh_radial"],
                                   mesh_angular=n["mesh_angular"], on_iteration=step)
        # the lib's label is "<n><irrep>" (e.g. "5No Symmetry "); the panel numbers levels itself
        levels = [dict(L, n=i + 1, label=L["label"].lstrip("0123456789").strip())
                  for i, L in enumerate(calc.levels())]
        polarized = any(L["spin"] for L in levels)

        def scalar(d, name, signed) -> ScalarField:
            return ScalarField(name, np.asarray(d["values"]), tuple(d["origin"]), tuple(d["spacing"]), signed=signed)

        def gradient(npts: int, pad: float) -> VectorField:
            d = calc.gradient(npts, pad)
            return VectorField("grad(rho)", np.asarray(d["vectors"]), tuple(d["origin"]), tuple(d["spacing"]))

        fields = {"density":  FieldRef("Electron density", False, lambda npts, pad: scalar(calc.density(npts, pad), "Electron density", False)),
                  "gradient": FieldRef("grad(rho)", False, gradient)}
        if polarized:      # an unpolarized run's m(r) is the zero field: nothing to draw, so not offered
            fields["spin_density"] = FieldRef("Spin density (ρ↑−ρ↓)", True,
                                              lambda npts, pad: scalar(calc.spin_density(npts, pad), "Spin density (ρ↑−ρ↓)", True))
        return Result(converged=calc.converged(), energy_Ha=calc.total_energy(), structure=data_structure(job.spec),
                      scf=steps, levels=levels, polarized=polarized, fields=fields,
                      orbital=lambda i, npts, pad: scalar(calc.orbital(i, npts, pad), f"MO #{i}", True),
                      resolved=job.resolved,
                      provenance={"code": NAME, "iterations": calc.iterations()})   # the closures above own `calc`
