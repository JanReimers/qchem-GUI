"""
qviz.backends.solid2 -- SOLID² (qchem) in process, through the nanobind bridge (doc/RunSpecPlan.md §3.1).

A MOLECULE: translate(spec) fills the arguments of qchem's production front door (MoleculeSetup::Resolve +
qchem::Calculation, bound by `qchem_py.Calculator`); run() converges on the calling (worker) thread
with the GIL released, streaming each iteration through the facade's constructor observer; the
Result samples density / spin density / orbitals in process, lazily.

A CRYSTAL: translate(spec) writes the qchem INPUT DECK (solid2_deck.to_deck -- the one translator, the file `rundeck`
runs); run() hands its JSON to `qchem_py.SolidRun` (deck::Build: resolve, lattice, basis, SolidSetup) and converges the
deck's schedule (deck::ScheduleToRun), streaming the same IterationRow cells.  A deck or setup the library refuses
raises with the library's text; an SCF that does not converge returns its SCFFailure reason and no energy.

Requires the compiled extension `qchem_py`, built in the SEPARATE qchem repo: point QCHEM_BUILD at a
build tree configured with -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DQCHEM_PYBIND=ON, e.g.

    export QCHEM_BUILD=~/Code/qchem-lane-g/build/Release

When the extension is absent the back end still REGISTERS; its capabilities say why it is not
installed, so the back-end strip greys it out instead of the app failing to start.

SAVED STATES: every crystal deck carries state.save (a file qviz.states names for the deck), so a finished run -- converged
or not -- leaves a state a later run can CONTINUE from (solid2_deck.continued: the deck's state.restartFrom, under whatever
accelerator and gates that spec states).  A molecule has no saved state yet (the molecular facade has none).

Scope today: single points.  A crystal's post-SCF actions (compute_U) need `rundeck` (deck::Build stops at the
SCF); a crystal's fields and bands are not sampled through the bridge yet.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import math
from typing import Optional

import numpy as np

from .. import states
from ..capability import Capabilities, Capability
from . import gaussian_basis
from ..data import ScalarField, SCFCell, SCFRow, VectorField
from ..spec import RunSpec
from . import solid2_deck
from .base import FieldRef, Job, Result, StepSink, code_block, data_structure, shared_representation

NAME = "SOLID²"

# The models the molecular facade has (CalcOptions::model) and the spec's names for them.
_MODEL = {("HF", None): "HF", ("DFT", "LDA"): "LDA", ("DFT", "Xalpha"): "Xalpha"}

# The registered Gaussian basis sets (BasisSet::Gaussian::Factory) -- a stated representation must be one.
BASES = ("dzvp", "dzvp2", "tzvp", "orb", "orb1", "sipp", "sipp_sr", "valence_lowq", "valence_lowq_sph",
         "valence_lowq_sr", "valence_lowq_sr2", "valence_lowq_va", "valence_lowq_vb")
_VALENCE_BASES = tuple(b for b in BASES if b.startswith(("sipp", "valence")))


_REP_KEYS = {"mesh_radial", "mesh_angular"}          # representation["solid2"]: the molecular XC mesh
#: console columns that are STATE, not a convergence trace -- the mixer ("Lin 0.50"), the accelerator ("DIIS:3") and the
#: configuration-changed flag: their numbers plot as nonsense, so they stay table-only (their printed text is kept)
_TABLE_ONLY = {"ρ_mix", "accel", "cfg"}


def mesh_for(eps: float) -> tuple[int, int]:
    """ε -> the molecular XC mesh (radial points, Lebedev degree) per atom.  PROVISIONAL, from one measurement
    (2026-10-30): F₂ GTH/valence_lowq_vb converges to −48.19259 Ha (±3e-5) from 80/29 up; 50/17 is 0.6 mHa/atom
    off, and the facade's own default (30 radial, degree 5 = 12 directions) is 0.52 Ha/atom off (all-electron LDA
    water/dzvp: 20 mHa/atom) -- so the default is never used for a DFT run."""
    return (50, 17) if eps >= 1e-3 else (80, 29) if eps >= 1e-6 else (120, 35)


def _uncovered(s: RunSpec) -> Optional[str]:
    """The reason a molecule's basis (stated, else basis_for) has no block for some of its elements -- None when it
    covers them all, or when BasisSetData cannot be read (then the facade itself refuses, with its own message)."""
    name = shared_representation(s).get("basis") or basis_for(s.accuracy.epsilon_Ha_per_atom)
    try:
        have = gaussian_basis.read_bsd(name)
    except (OSError, ValueError):
        return None
    missing = sorted({x.element for x in s.structure.sites} - {e.capitalize() for e in have})
    return f"basis {name} has no block for {', '.join(missing)}" if missing else None


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


#: the spec's DIIS knobs (accuracy.scf) -> the molecular facade's AcceleratorOptions (the Ladder's DIIS rung reads them too)
DIIS_KNOBS = {"diis_depth": ("diis_depth", 0), "diis_sv_tol": ("diis_sv_tol", 0.0), "diis_engage_fd": ("diis_engage_fd", 0.0)}

_GATES = (("minFD", "[F,D]"), ("minDrho", "Δρ"), ("minDE", "ΔE"), ("minDFD", "Δ[F,D]"))


def _gates_run(record: dict) -> Optional[str]:
    """The SCF gates the run USED, from its record's scf.standard (the SCF iterator writes it); None when absent."""
    for key, run in (record or {}).items():
        std = (run.get("scf") or {}).get("standard")
        if std and not key.startswith("gridAudit"):
            on = [f"{label} {std[k]:g}" for k, label in _GATES
                  if isinstance(std.get(k), (int, float))]                              # "off" / absent: not a gate
            return "run: " + ", ".join(on) + f"; at most {std.get('nMaxIter')} iterations"
    return None


def _as_run(resolved: dict, record: dict) -> dict:
    """A job's resolved settings with what the run's record states in place of what was planned."""
    out = dict(resolved)
    gates = _gates_run(record)
    if gates:
        out["SCF gates"] = gates
    return out


class Solid2Backend:
    name = NAME

    def __init__(self):
        self._qc, self._why_absent = _import_qchem_py()
        self._built = None          # (deck text, built SolidRun, its build record): the ONE Build kept for a Run of that deck
        self._prior: dict = {}      # the reused build's record, for the live snapshot of the run it feeds

    # -- build: every pre-SCF step, no SCF -----------------------------------
    def why_no_build(self, spec: RunSpec) -> Optional[str]:
        """Why `build` cannot run for `spec` (None: it can)."""
        if self._qc is None:
            return self._why_absent
        if spec.structure.kind == "molecule":
            return "a molecule has no build-only step in the library yet (OpenWork §3: lane B is adding it)"
        return solid2_deck.why_not(spec)

    def build(self, job: Job) -> dict:
        """deck::Build alone -- basis, grids, pseudopotentials, audits -- and the build run's record.  The built calculation
        is KEPT: a Run of the same deck converges it instead of building again (an edited spec's deck differs: rebuilt).
        Raises with the library's text on a refusal.  Call on the app's serial library thread."""
        if "deck" not in job.native:
            raise RuntimeError(self.why_no_build(job.spec) or "no build-only step for this spec")
        self._built = None                                       # one kept build at a time (a crystal's can be large)
        self._qc.clear_report()
        run = self._qc.SolidRun(job.native["deck"])
        record = json.loads(self._qc.report())
        self._built = (job.native["deck"], run, record)
        return record

    # -- capabilities ------------------------------------------------------
    def capabilities(self) -> Capabilities:
        def scf(s: RunSpec) -> dict:
            return s.accuracy.scf or {}
        rep = shared_representation
        def molecule(check):          # the molecular facade's constraints bind a molecule only; a crystal answers to its deck
            return lambda s: s.structure.kind != "molecule" or check(s)

        def crystal(s: RunSpec) -> bool:
            return s.structure.kind != "crystal" or solid2_deck.why_not(s) is None
        return Capabilities(NAME, installed=self._qc is not None, why_absent=self._why_absent, features=[
            Capability("structure.kind=crystal -> deck", crystal, lambda s: solid2_deck.why_not(s) or ""),
            Capability("task.kind=energy", lambda s: s.task.kind == "energy",
                       "single points only (MoleculeSession relax is not bound; a crystal's compute_U runs in rundeck: "
                       "deck::Build stops at the SCF)"),
            Capability("physics.theory/functional", molecule(lambda s: (s.physics.theory, s.physics.functional
                       if s.physics.theory == "DFT" else None) in _MODEL),
                       "molecular models are HF, LDA and Xalpha"),
            Capability("physics.relativity", lambda s: s.physics.relativity == "none"
                       or (s.physics.relativity == "scalar" and s.physics.core == "pseudopotential"),
                       "non-relativistic all-electron (scalar relativity only through a pseudopotential)"),
            Capability("physics.hubbard=()", molecule(lambda s: not s.physics.hubbard), "no molecular DFT+U in the facade"),
            Capability("physics.occupations=none", molecule(lambda s: s.physics.occupations.smearing == "none"),
                       "no smearing for molecules"),
            Capability("physics.core=pseudopotential needs LDA + a valence basis",
                       molecule(lambda s: s.physics.core != "pseudopotential"
                                or (s.physics.functional == "LDA" and rep(s).get("basis") in _VALENCE_BASES)),
                       "a molecular pseudopotential run is LDA with a stated valence basis (sipp*, valence_lowq*)"),
            Capability("accuracy.representation keys", molecule(lambda s: set(rep(s)) <= {"basis"}
                       and set(code_block(s, "solid2")) <= _REP_KEYS),
                       "a stated representation is a shared basis, or SOLID² settings under representation['solid2'] "
                       "(mesh_radial, mesh_angular)"),
            Capability("accuracy.representation.basis", molecule(lambda s: rep(s).get("basis") in (None, *BASES)),
                       "the stated basis is not a registered Gaussian basis"),
            Capability("accuracy.scf.mixer", molecule(lambda s: scf(s).get("mixer", "Auto") in ("Auto", "Linear")),
                       "Kerker and Pulay mix in G space: a molecule's density mixer is Auto (= linear) or Linear"),
            Capability("accuracy.scf.mixing_alpha", lambda s: all(0.0 < float(scf(s)[k]) <= 1.0 for k in
                                                                 ("mixing_alpha", "mixing_alpha_max") if k in scf(s)),
                       "a linear mixing step α is in (0, 1]"),
            Capability("continue from a saved state", molecule(lambda s: solid2_deck.continues_from(s) is None),
                       "the molecular facade has no saved state to continue from yet"),
            Capability("the basis covers every element", molecule(lambda s: _uncovered(s) is None),
                       lambda s: _uncovered(s) or ""),
        ])

    # -- translate ----------------------------------------------------------
    def translate(self, spec: RunSpec) -> Job:
        if spec.structure.kind == "crystal":
            deck = solid2_deck.to_deck(spec)
            if (deck.get("basis") or {}).get("family") != "planeWave":     # a plane-wave state cannot be saved (OpenWork §3)
                deck.setdefault("state", {})["save"] = states.path_for(spec.label, json.dumps(deck, sort_keys=True))
            text = json.dumps(deck, indent=2)
            return Job(NAME, spec, native={"deck": text}, resolved=dict(solid2_deck.resolved(spec)),
                       files={"deck": text.encode()})
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
            "accelerator": "" if scf.get("accelerator", "Auto") == "Auto" else str(scf["accelerator"]),   # "": the facade's DIIS
            "mesh_radial": mesh[0], "mesh_angular": mesh[1],
            "relax_start": float(scf.get("mixing_alpha", 0.0)), "relax_max": float(scf.get("mixing_alpha_max", 0.0)),
            **{arg: type(zero)(scf.get(key, zero)) for key, (arg, zero) in DIIS_KNOBS.items()},
            "mom": bool(scf.get("mom", False)), "mom_start": int(scf.get("mom_start", 0)),
            "relax_min": float(scf.get("mixing_alpha_min") or 0.0),
            "ladder_switch_at": -1.0 if scf.get("ladder_switch_at") is None else float(scf["ladder_switch_at"]),
            **{key: -1.0 if scf.get(key) is None else 0.0 if scf[key] == "off" else float(scf[key])   # < 0: default, 0: off
               for key, _ in solid2_deck.DECK_GATES},
        }
        resolved = {
            "basis": f"{basis} ({'stated' if rep.get('basis') else 'from ε, provisional'})",
            "model": native["model"],
            "pseudopotential": native["pp"] or "all-electron",
            "spin": ("polarized" if native["polarized"] or native["multiplicity"] > 1 else "unpolarized")
                    + (f", 2S+1={native['multiplicity']}" if native["multiplicity"] else ""),
            "point-group blocking": native["symmetry"],
            "accelerator": native["accelerator"] or "DIIS (Auto: the molecular facade's default)",
            "XC mesh": (f"{mesh[0]} radial x Lebedev {mesh[1]} per atom"
                        + (" stated" if own else " from ε, provisional")) if dft else "none (HF)",
            "SCF gates": "planned: " + (", ".join(f"{k[5:]} {scf[k]}" for k, _ in solid2_deck.DECK_GATES if k in scf)
                                        + "; the rest the facade's defaults" if any(k in scf for k, _ in solid2_deck.DECK_GATES)
                                        else "the facade's defaults (ε not yet mapped)") + "; the run states what it used",
            "max_iter": native["max_iter"],
            "linear mixing α": (f"start {native['relax_start'] or 'default (1.0)'}, "
                                f"min {native['relax_min'] or 'default (no floor)'}, "
                                f"max {native['relax_max'] or 'default (1.0)'}"),
            "Ladder → GDM": ("default (|ΔE/E| < 1e-6)" if native["ladder_switch_at"] < 0 else "on stall only"
                             if native["ladder_switch_at"] == 0 else f"|ΔE/E| < {native['ladder_switch_at']:g}"),
            "DIIS": (f"depth {native['diis_depth'] or 'default (4)'}, SV floor {native['diis_sv_tol'] or 'default (5e-9)'}, "
                     f"engages below [F,D] {native['diis_engage_fd'] or 'default (Z-scaled)'}"),
            "MOM": (f"on, reference captured at iteration {native['mom_start'] or 'default (10)'}" if native["mom"] else "off"),
        }
        return Job(NAME, spec, native=native, resolved=resolved)

    # -- run + parse --------------------------------------------------------
    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result:
        if self._qc is None:
            raise RuntimeError(self._why_absent)
        n = job.native
        steps: list[SCFRow] = []

        def step(it, E, cells):
            # the lib's IterationRow, cell for cell: the console's columns, titles, printed text, gates and scales
            s = SCFRow(it, E, tuple(SCFCell(title, math.nan if title in _TABLE_ONLY else value, scale, text, tol)
                                    for title, value, text, tol, scale in cells))
            steps.append(s)
            if not job.live:                          # setup has filed its sections (grids, basis) by iteration 1
                snap = self._qc.current_report()
                if snap:                              # + the runs already filed: a crystal's grids are its BUILD run's
                    job.live.update(self._prior | json.loads(self._qc.report()))
                    job.live["running@live"] = json.loads(snap)
            if on_step:
                on_step(s)

        if "deck" in n:
            return self._run_deck(job, step, steps)

        self._qc.clear_report()
        calc = self._qc.Calculator(n["numbers"], n["positions"], basis=n["basis"], model=n["model"],
                                   symmetry=n["symmetry"], polarized=n["polarized"],
                                   multiplicity=n["multiplicity"], max_iter=n["max_iter"], pp=n["pp"],
                                   accelerator=n["accelerator"], mesh_radial=n["mesh_radial"],
                                   mesh_angular=n["mesh_angular"], relax_start=n.get("relax_start", 0.0),
                                   relax_max=n.get("relax_max", 0.0), diis_depth=n["diis_depth"], diis_sv_tol=n["diis_sv_tol"],
                                   diis_engage_fd=n["diis_engage_fd"], mom=n["mom"], mom_start=n["mom_start"],
                                   relax_min=n["relax_min"], ladder_switch_at=n["ladder_switch_at"],
                                   **{k: n[k] for k, _ in solid2_deck.DECK_GATES}, on_iteration=step)
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
        converged = calc.converged()
        return Result(converged=converged, energy_Ha=calc.total_energy() if converged else None, failure=calc.failure(),
                      structure=data_structure(job.spec),
                      scf=steps, levels=levels, polarized=polarized, fields=fields,
                      orbital=lambda i, npts, pad: scalar(calc.orbital(i, npts, pad), f"MO #{i}", True),
                      resolved=_as_run(job.resolved, record := json.loads(self._qc.report())),
                      provenance={"code": NAME, "iterations": calc.iterations()}, record=record)   # the closures above own `calc`

    def _run_deck(self, job: Job, step, steps: list[SCFRow]) -> Result:
        """A crystal: deck::Build (a refusal raises with the library's text), then the deck's schedule, streamed."""
        built, self._built = self._built, None
        states.state_dir().mkdir(parents=True, exist_ok=True)      # the facade writes the state aside and renames it in
        self._qc.clear_report()
        if built is not None and built[0] == job.native["deck"]:
            run, self._prior = built[1], built[2]                  # the Build of THIS deck: converge it, do not rebuild
        else:
            run, self._prior = self._qc.SolidRun(job.native["deck"]), {}
        try:
            converged = run.converge(step)
        finally:
            prior, self._prior = self._prior, {}
        record = prior | json.loads(self._qc.report())
        saved = (json.loads(job.native["deck"]).get("state") or {}).get("save")
        return Result(converged=converged, energy_Ha=run.total_energy(), failure=run.failure(),
                      structure=data_structure(job.spec), scf=steps,
                      resolved=_as_run(job.resolved, record),
                      provenance={"code": NAME, "iterations": run.iterations(), "deck": json.loads(run.resolved())},
                      record=record, state_file=saved if saved and os.path.exists(saved) else None)
