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
accelerator and gates that spec states).  A molecule saves too (Calculator.save_state after every run, the file qviz.states
names for its native arguments) and continues through the same spec key (restart_from); the library judges the file (an exact
resume, a warm start naming each difference, or RestartRefused) and the Result's provenance carries what it did
(restartedFrom).

BUILD -> RUN: build(job) stops before the SCF (a crystal's deck::Build, a molecule's MoleculeCalculation::Prepare) and keeps
the calculation; a Run of the same job converges it, so the Basis and Grids tabs read what the run will use.

Scope today: single points.  A crystal's post-SCF actions (compute_U) need `rundeck` (deck::Build stops at the
SCF); a crystal's density and spin density are sampled (SolidRun.field, over the cell's bounding box), its bands are not yet.
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
from ..data import ScalarField, SCFCell, SCFRow, VectorField
from ..spec import RunSpec
from . import solid2_deck
from .base import FieldRef, Job, Refused, Result, RunStopped, StepSink, code_block, data_structure, shared_representation

NAME = "SOLID²"

# The models the molecular facade has (CalcOptions::model) and the spec's names for them.
_MODEL = {("HF", None): "HF", ("DFT", "LDA"): "LDA", ("DFT", "Xalpha"): "Xalpha"}

# The registered Gaussian basis sets (BasisSet::Gaussian::Factory) -- a stated representation must be one.
BASES = ("dzvp", "dzvp2", "tzvp", "orb", "orb1", "sipp", "sipp_sr", "valence_lowq", "valence_lowq_sph",
         "valence_lowq_sr", "valence_lowq_sr2", "valence_lowq_va", "valence_lowq_vb")


_REP_KEYS = {"mesh_radial", "mesh_angular", "basis_file"}   # representation["solid2"]: the molecular XC mesh, a downloaded basis
#: console columns that are STATE, not a convergence trace -- the mixer ("Lin 0.50"), the accelerator ("DIIS:3") and the
#: configuration-changed flag: their numbers plot as nonsense, so they stay table-only (their printed text is kept)
_TABLE_ONLY = {"ρ_mix", "accel", "cfg"}


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


#: the spec's DIIS knobs (accuracy.scf) -> the molecular facade's AcceleratorOptions (the Ladder's DIIS rung reads them too)
DIIS_KNOBS = {"diis_depth": ("diis_depth", 0), "diis_sv_tol": ("diis_sv_tol", 0.0), "diis_engage_fd": ("diis_engage_fd", 0.0)}

#: minDrho gates the density-change column whatever the Hamiltonian titles it (Δρ with a real-space quadrature, else ‖ΔD‖)
_GATES = (("minFD", "[F,D]"), ("minDrho", "density change"), ("minDE", "ΔE"), ("minDFD", "Δ[F,D]"))


def _gates_run(record: dict) -> Optional[str]:
    """The SCF gates the run USED, from its record's scf.standard (the SCF iterator writes it); None when absent.  The run's
    own SCF is the last filed: a +U crystal's pseudo-atom SCFs come first, each with its own standard."""
    for key, run in reversed((record or {}).items()):
        std = (run.get("scf") or {}).get("standard")
        if std and not key.startswith("gridAudit"):
            on = [f"{label} {std[k]:g}" for k, label in _GATES
                  if isinstance(std.get(k), (int, float))]                              # "off" / absent: not a gate
            return "run: " + ", ".join(on) + f"; at most {std.get('nMaxIter')} iterations"
    return None


def _calculator_args(n: dict) -> dict:
    """qchem_py.Calculator's keyword arguments from a molecule job's native dict (the state it saves is not one)."""
    return dict(numbers=n["numbers"], positions=n["positions"], basis=n["basis"], model=n["model"],
                symmetry=n["symmetry"], polarized=n["polarized"], multiplicity=n["multiplicity"], max_iter=n["max_iter"],
                pp=n["pp"], accelerator=n["accelerator"], mesh_radial=n["mesh_radial"], mesh_angular=n["mesh_angular"],
                relax_start=n.get("relax_start", 0.0), relax_max=n.get("relax_max", 0.0), diis_depth=n["diis_depth"],
                diis_sv_tol=n["diis_sv_tol"], mixer=n["mixer"], smearing_kT=n["smearing_kT"],
                diis_engage_fd=n["diis_engage_fd"], mom=n["mom"], mom_start=n["mom_start"], relax_min=n["relax_min"],
                ladder_switch_at=n["ladder_switch_at"], restart_from=n.get("restart_from", ""),
                basis_file=n.get("basis_file", ""), seed=n.get("seed", ""), spherical=n.get("spherical", False),
                **{k: n[k] for k, _ in solid2_deck.DECK_GATES})


def _native_key(n: dict) -> str:
    """What identifies a job for its kept Build: a crystal's deck text, a molecule's arguments."""
    return n["deck"] if "deck" in n else json.dumps(n, sort_keys=True)


def cell_box(lattice, n: int, reps=(1, 1, 1)) -> tuple[list[int], list[float], list[float]]:
    """The Cartesian box a crystal's field is sampled on: the bounding box of `reps` cells (rows of `lattice`, bohr), n points
    along its longest side and as many per bohr along the others, PERIODIC (no end point: the density repeats).  Returns
    (n3, origin, spacing) for SolidRun.field."""
    L = np.asarray(lattice, float) * np.asarray(reps, float)[:, None]
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ L
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    side = hi - lo
    h = side.max() / n
    n3 = [max(2, int(round(s / h))) for s in side]
    return n3, lo.tolist(), (side / np.asarray(n3)).tolist()


def _molecule_deck_ok(deck) -> bool:
    """A molecule's deck block: none, or only state.restartFrom (the molecular deck keys wait for V-FACADE-OPTS)."""
    return not deck or (set(deck) == {"state"} and set(deck["state"]) <= {"restartFrom"})


def restart_text(R: Optional[dict]) -> Optional[str]:
    """What a restart did (the facade's RestartedFrom, as Calculator / SolidRun.restarted_from return it), one line; None: fresh."""
    if not R:
        return None
    head = ("exact resume" if R["exact"] else "warm start") + f" from {os.path.basename(R['path'])}"
    more = list(R.get("warm") or [])
    if R.get("maxDisplacement", 0.0) > 1e-10:
        more.append(f"atoms moved up to {R['maxDisplacement']:.3g} bohr (density carried as AO coefficients)")
    more.append(f"restored density holds {R['electrons']:.6f} electrons here")
    return head + ": " + "; ".join(more)


def _as_run(resolved: dict, record: dict, restarted: Optional[dict] = None) -> dict:
    """A job's resolved settings with what the run's record states in place of what was planned."""
    out = dict(resolved)
    gates = _gates_run(record)
    if gates:
        out["SCF gates"] = gates
    if "starts from" in out:
        out["starts from"] = restart_text(restarted) or out["starts from"]
    return out


class Solid2Backend:
    name = NAME

    def __init__(self):
        self._qc, self._why_absent = _import_qchem_py()
        self._built = None          # (native key, built SolidRun / Calculator, its build record): the ONE Build kept for a Run of it
        self._prior: dict = {}      # the reused build's record, for the live snapshot of the run it feeds

    # -- build: every pre-SCF step, no SCF -----------------------------------
    def why_no_snapshot(self, spec: RunSpec) -> Optional[str]:
        """Why a RUNNING job of `spec` cannot hand out its current iterate's density (None: it can)."""
        return self._why_absent if self._qc is None else None   # a molecule's iterate_density, a crystal's field(iterate=True)

    def why_no_build(self, spec: RunSpec) -> Optional[str]:
        """Why `build` cannot run for `spec` (None: it can)."""
        if self._qc is None:
            return self._why_absent
        if spec.structure.kind == "molecule":
            return None                               # MoleculeCalculation::Prepare (the translate refuses what it cannot run)
        return solid2_deck.why_not(spec)

    def build(self, job: Job) -> dict:
        """Every pre-SCF step alone and the build run's record: a crystal's deck::Build (basis, grids, pseudopotentials, audits),
        a molecule's MoleculeCalculation::Prepare (basis per irrep with the overlap's conditioning, every mesh, the SCF iterator;
        its run "<ID> build").  The built calculation is KEPT: a Run of the same job converges it instead of building again (an
        edited spec's job differs: rebuilt).  Raises Refused (the library's text and record) on a refusal.  Call on the app's
        serial library thread."""
        self._built = None                                       # one kept build at a time (a crystal's can be large)
        self._qc.clear_report()
        self._prior = {}
        try:
            built = (self._qc.SolidRun(job.native["deck"]) if "deck" in job.native
                     else self._qc.Calculator(**_calculator_args(job.native), build_only=True))
        except Exception as e:                                   # noqa: BLE001 -- the library's refusal, in its own words
            raise Refused(str(e), self._filed()) from e
        record = json.loads(self._qc.report())
        self._built = (_native_key(job.native), built, record)
        return record

    def _kept_build(self, native: dict):
        """The kept Build of exactly this job (its calculation and record, consumed), else None."""
        built, self._built = self._built, None
        return built[1:] if built is not None and built[0] == _native_key(native) else None

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
            Capability("physics.occupations: Fermi-Dirac", molecule(lambda s: s.physics.occupations.smearing in ("none", "fermi-dirac")),
                       "the molecular facade smears Fermi-Dirac only"),
            Capability("accuracy.representation keys", molecule(lambda s: set(rep(s)) <= {"basis"}
                       and set(code_block(s, "solid2")) <= _REP_KEYS | {"deck"}),
                       "a stated representation is a shared basis, or SOLID² settings under representation['solid2'] "
                       "(mesh_radial, mesh_angular)"),
            Capability("molecular deck keys", molecule(lambda s: _molecule_deck_ok(code_block(s, "solid2").get("deck"))),
                       "a molecule has no input deck yet (V-FACADE-OPTS): its only deck key is state.restartFrom"),
            Capability("accuracy.representation.basis", molecule(lambda s: rep(s).get("basis") in (None, *BASES)
                                                                 or bool(code_block(s, "solid2").get("basis_file"))),
                       "the stated basis is not a registered Gaussian basis (nor a downloaded one: Basis ▸ Basis Set Exchange)"),
            Capability("a downloaded basis file exists", lambda s: not code_block(s, "solid2").get("basis_file")
                       or os.path.exists(code_block(s, "solid2")["basis_file"]),
                       "the downloaded basis file is gone (Basis ▸ Basis Set Exchange downloads it again)"),
            Capability("a downloaded basis is molecular", lambda s: s.structure.kind == "molecule"
                       or not code_block(s, "solid2").get("basis_file"),
                       "a downloaded basis is all-electron and molecular; a crystal runs on a valence basis with its pseudopotential"),
            Capability("accuracy.scf.mixing_alpha", lambda s: all(0.0 < float(scf(s)[k]) <= 1.0 for k in
                                                                 ("mixing_alpha", "mixing_alpha_max") if k in scf(s)),
                       "a linear mixing step α is in (0, 1]"),
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
            "mixer": str(scf.get("mixer", "Auto")),          # the facade refuses what a molecule cannot mix (Kerker: G space)
            "smearing_kT": float(ph.occupations.kT_Ha) if ph.occupations.smearing == "fermi-dirac" else 0.0,
            "basis_file": str(own.get("basis_file") or ""),
            "seed": str(scf.get("seed") or ""),
            "spherical": bool(rep.get("spherical")),
            "restart_from": solid2_deck.continues_from(spec) or "",
            **{key: -1.0 if scf.get(key) is None else 0.0 if scf[key] == "off" else float(scf[key])   # < 0: default, 0: off
               for key, _ in solid2_deck.DECK_GATES},
        }
        # the state this run SAVES: named for its arguments (where it starts from excluded: a continuation of the same
        # settings overwrites its own file, as a crystal deck's does)
        native["state_save"] = states.path_for(spec.label, json.dumps({k: v for k, v in native.items() if k != "restart_from"},
                                                                      sort_keys=True))
        resolved = {
            "starts from": (f"the saved state {native['restart_from']} (judged when it runs)" if native["restart_from"]
                            else f"scratch, seeded by {native['seed'] or 'Auto: ' + solid2_deck.seed_auto(spec)}"),
            "Gaussians": "spherical" if native["spherical"] else "Cartesian",
            "basis": (f"{basis} (downloaded: {native['basis_file']})" if native["basis_file"] else
                      f"{basis} ({'stated' if rep.get('basis') else 'from ε, provisional'})"),
            "model": native["model"],
            "pseudopotential": native["pp"] or "all-electron",
            "spin": ("polarized" if native["polarized"] or native["multiplicity"] > 1 else "unpolarized")
                    + (f", 2S+1={native['multiplicity']}" if native["multiplicity"] else ""),
            "point-group blocking": native["symmetry"],
            "accelerator": native["accelerator"] or "DIIS (Auto: the molecular facade's default)",
            "XC mesh": (f"{mesh[0]} radial x Lebedev {mesh[1]} per atom"
                        + (" stated" if {"mesh_radial", "mesh_angular"} & set(own) else " from ε, provisional")) if dft else "none (HF)",
            "SCF gates": "planned: " + (", ".join(f"{k[5:]} {scf[k]}" for k, _ in solid2_deck.DECK_GATES if k in scf)
                                        + "; the rest the facade's defaults" if any(k in scf for k, _ in solid2_deck.DECK_GATES)
                                        else "the facade's defaults (ε not yet mapped)") + "; the run states what it used",
            "max_iter": native["max_iter"],
            "linear mixing α": (f"start {native['relax_start'] or 'default (1.0)'}, "
                                f"min {native['relax_min'] or 'default (no floor)'}, "
                                f"max {native['relax_max'] or 'default (1.0)'}"),
            "Ladder → GDM": ("default (|ΔE/E| < 1e-6)" if native["ladder_switch_at"] < 0 else "on stall only"
                             if native["ladder_switch_at"] == 0 else f"|ΔE/E| < {native['ladder_switch_at']:g}"),
            "DIIS": (f"depth {native['diis_depth'] or 'default (4)'}, SV floor {native['diis_sv_tol'] or 'default (off; relative 1e-4)'}, "
                     f"engages below [F,D] {native['diis_engage_fd'] or 'default (Z-scaled)'}"),
            "MOM": (f"on, reference captured at iteration {native['mom_start'] or 'default (10)'}" if native["mom"] else "off"),
        }
        return Job(NAME, spec, native=native, resolved=resolved)

    # -- run + parse --------------------------------------------------------
    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result:
        """Converge `job`, streaming its rows.  A library throw raises Refused with the record filed up to it."""
        if self._qc is None:
            raise RuntimeError(self._why_absent)
        try:
            return self._solve(job, on_step)
        except (RunStopped, Refused):
            raise
        except Exception as e:                        # noqa: BLE001 -- the library's refusal, in its own words
            raise Refused(str(e), self._filed()) from e
        finally:
            self._prior = {}

    def _filed(self) -> dict:
        """The record so far: the runs the library filed, and the one it was in when it refused (still open).  A run refused
        before it wrote anything files as an empty json value, not an object: dropped, so every reader sees runs as objects."""
        record = self._prior | json.loads(self._qc.report())
        open_run = self._qc.current_report()
        if open_run:
            record["refused@open"] = json.loads(open_run)
        return {k: v for k, v in record.items() if isinstance(v, dict) and v}

    def _solve(self, job: Job, on_step: Optional[StepSink]) -> Result:
        n = job.native
        steps: list[SCFRow] = []

        live = {}                                     # "sample": (n, pad, spin) -> grid dict of the CURRENT iterate, once a handle exists

        def step(it, E, cells):
            # the lib's IterationRow, cell for cell: the console's columns, titles, printed text, gates and scales
            s = SCFRow(it, E, tuple(SCFCell(title, math.nan if title in _TABLE_ONLY else value, scale, text, tol)
                                    for title, value, text, tol, scale in cells))
            steps.append(s)
            while job.snapshots and "sample" in live:   # a live snapshot: the compute thread, between iterations (the safe point)
                spin, npts, pad, deliver = job.snapshots.pop(0)
                name = "Spin density (ρ↑−ρ↓)" if spin else "Electron density"
                try:
                    d = live["sample"](npts, pad, spin)
                except Exception as e:                # noqa: BLE001 -- told to the asker, never a stopped run
                    deliver(it, None, str(e))
                else:
                    deliver(it, ScalarField(f"{name}, iteration {it}", np.asarray(d["values"]), tuple(d["origin"]),
                                            tuple(d["spacing"]), signed=spin), None)
            if not job.live:                          # setup has filed its sections (grids, basis) by iteration 1
                snap = self._qc.current_report()
                if snap:                              # + the runs already filed: a crystal's grids are its BUILD run's
                    job.live.update(self._prior | json.loads(self._qc.report()))
                    job.live["running@live"] = json.loads(snap)
            if on_step:
                on_step(s)

        if "deck" in n:
            return self._run_deck(job, step, steps, live)

        self._qc.clear_report()
        kept = self._kept_build(n)
        if kept is not None:                          # this job's Build: converge its graph, do not rebuild
            calc, self._prior = kept
        else:                                         # Prepare, then the SCF: a handle exists while it runs (live snapshots)
            calc = self._qc.Calculator(**_calculator_args(n), build_only=True)
        live["sample"] = lambda npts, pad, spin: calc.iterate_density(npts, pad, spin)
        calc.run_scf(step, n["max_iter"])
        saved = n.get("state_save")
        if saved:                                     # converged or not: a later run continues from it (the file says which)
            states.state_dir().mkdir(parents=True, exist_ok=True)
            calc.save_state(saved)
        restarted = calc.restarted_from()
        # the lib's label is "<n><irrep>" -- n counts WITHIN the irrep (methane C2v: 1A1 2A1 1B2 3A1 1B1), the Mulliken
        # numbering; with no symmetry there is one irrep and n is the global count ("5No Symmetry")
        levels = [dict(L, n=int(L["label"].strip()[:len(L["label"].strip()) - len(lab)] or i + 1), label=lab)
                  for i, L in enumerate(calc.levels()) for lab in [L["label"].strip().lstrip("0123456789")]]
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
                      resolved=_as_run(job.resolved, record := self._prior | json.loads(self._qc.report()), restarted),
                      provenance={"code": NAME, "iterations": calc.iterations(), "restartedFrom": restarted},
                      record=record,                  # the closures above own `calc`
                      state_file=saved if saved and os.path.exists(saved) else None)

    def _run_deck(self, job: Job, step, steps: list[SCFRow], live: dict) -> Result:
        """A crystal: deck::Build (a refusal raises with the library's text), then the deck's schedule, streamed."""
        kept = self._kept_build(job.native)
        states.state_dir().mkdir(parents=True, exist_ok=True)      # the facade writes the state aside and renames it in
        self._qc.clear_report()
        if kept is not None:
            run, self._prior = kept                                # the Build of THIS deck: converge it, do not rebuild
        else:
            run, self._prior = self._qc.SolidRun(job.native["deck"]), {}
        struct = data_structure(job.spec)

        def sample(npts: int, iterate: bool, spin: bool) -> dict:
            box = cell_box(struct.lattice, npts)
            return run.field(*box, spin=spin, iterate=iterate)
        live["sample"] = lambda npts, pad, spin: sample(npts, True, spin)
        converged = run.converge(step)                             # run() clears self._prior, after a refusal has read it
        record = self._prior | json.loads(self._qc.report())
        saved = (json.loads(job.native["deck"]).get("state") or {}).get("save")
        restarted = run.restarted_from()
        polarized = bool(job.spec.physics.spin.polarized)

        def scalar(name: str, spin: bool):
            def f(npts: int, pad: float) -> ScalarField:
                d = sample(npts, False, spin)              # the CONVERGED density, over the cell's bounding box (pad: unused)
                return ScalarField(name, np.asarray(d["values"]), tuple(d["origin"]), tuple(d["spacing"]), signed=spin)
            return f
        fields = {}
        if converged:                                  # a crystal's fields: rho, and m when polarized (the library refuses an unconverged one)
            fields["density"] = FieldRef("Electron density", False, scalar("Electron density", False))
            if polarized:
                fields["spin_density"] = FieldRef("Spin density (ρ↑−ρ↓)", True, scalar("Spin density (ρ↑−ρ↓)", True))
        return Result(converged=converged, energy_Ha=run.total_energy(), failure=run.failure(),
                      structure=struct, scf=steps, fields=fields, polarized=polarized,
                      resolved=_as_run(job.resolved, record, restarted),
                      provenance={"code": NAME, "iterations": run.iterations(), "deck": json.loads(run.resolved()),
                                  "restartedFrom": restarted},
                      record=record, state_file=saved if saved and os.path.exists(saved) else None)
