"""
qviz.backends.qe -- Quantum ESPRESSO (pw.x, plane waves) as a back end (doc/RunSpecPlan.md §3.3).

translate(spec) writes one pw.x input in the shape of qchem's banked QE oracle decks (IntegrationTests/QE/*.in); run()
launches it through qviz.runner.MpiRunner -- ALWAYS `mpirun -np N pw.x -in <deck>`, never the bare binary -- in
~/Code/qe-runs/<Material>/, tails stdout into the live SCF trace, and parses the output into a neutral Result.

Rules the translator enforces (each cost a wrong number once):
  * the dense FFT grid nr1..3 is ALWAYS stated (qchem CLAUDE.md, 2026-10-26: pw.x evaluates E_xc on its minimal grid, and
    on β-MnO₂ at 40 Ry the default 36×36×24 was 107 μHa off the converged 72³).  The stated grid is QE's own minimal grid
    scaled by `grid_factor` (derived from ε, or stated), so a run's record says exactly what it was evaluated on;
  * the pseudopotentials are OUR GTH set as UPF files written by qchem's `gth2upf` (`<El>.pz-gth-q<q>.UPF`, q from the
    gth-lda-valence set), so QE runs the same pseudopotential -- and, through PP_CHI, the same Hubbard projector -- as
    SOLID² and CP2K.  A missing file is a capability refusal naming the gth2upf command, never a substitute;
  * LDA is `sla+vwn` (Dirac exchange + VWN, as SOLID² and CP2K), not QE's PZ default;
  * no SCF cap is accepted as a result: "convergence NOT achieved" is a failed run.

A STATED representation for QE lives under accuracy.representation["qe"]:
    {"pseudo_dir": path, "ecutwfc_Ry": 300, "ecutrho_Ry": 1200, "nr": [96, 96, 60] | "grid_factor": 1.5,
     "starting_magnetization": {"<species>": 0.8}, "nbnd": 30, "conv_thr": 1e-10, "mixing_beta": 0.3}
Every key is optional; an unstated one is derived (the PROVISIONAL table in `derive`).  The block is QE's alone.
"""
from __future__ import annotations

import functools
import json
import math
import os
import pathlib
import re
import shutil
from typing import Optional

from ..capability import Capabilities, Capability
from ..data import SCFStep
from ..runner import MpiRunner, next_prefix, run_dir
from ..spec import RunSpec
from .base import Job, Result, StepSink, code_block, data_structure, shared_representation, species_kinds
from .cp2k import _d_shell_n

NAME = "QE"
PW_EXE = os.environ.get("QE_PW", str(pathlib.Path.home() / "Code/q-e/bin/pw.x"))

_XC = {"LDA": "sla+vwn"}
_SMEARING = {"fermi-dirac": "fd", "gaussian": "gaussian", "marzari-vanderbilt": "mv"}
_PROJECTOR = {"ortho-atomic": "ortho-atomic", "atomic": "atomic"}
_REP_KEYS = {"pseudo_dir", "ecutwfc_Ry", "ecutrho_Ry", "nr", "grid_factor", "starting_magnetization", "nbnd", "conv_thr",
             "mixing_beta"}
# standard atomic weights through Kr (pw.x needs one per species; an SCF energy does not depend on it)
_MASS = (1.008, 4.003, 6.94, 9.012, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982, 28.085, 30.974,
         32.06, 35.45, 39.95, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,
         69.723, 72.630, 74.922, 78.971, 79.904, 83.798)


def qchem_src() -> pathlib.Path:
    return pathlib.Path(os.environ.get("QCHEM_SRC", pathlib.Path.home() / "Code/qchem-main"))


@functools.lru_cache(maxsize=None)
def pp_valence() -> dict[str, int]:
    """q per element of qchem's gth-lda-valence set (src/Pseudopotential/Data/pseudopotentials.json)."""
    d = json.loads((qchem_src() / "src/Pseudopotential/Data/pseudopotentials.json").read_text())
    return {k.capitalize(): int(v) for k, v in d["gth-lda-valence"]["valence"].items()}


def upf_name(element: str) -> str:
    return f"{element}.pz-gth-q{pp_valence()[element]}.UPF"


def good_fft(n: int) -> int:
    """The smallest m >= n whose only prime factors are 2, 3, 5 (QE's good_fft_order for FFTW)."""
    m = n
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1:
            return m
        m += 1


def fft_grid(lattice_bohr, ecutrho_Ry: float, factor: float = 1.0) -> list[int]:
    """pw.x's minimal dense grid for `ecutrho` (nr_i = int(sqrt(ecutrho)·|a_i|/π) + 1, made FFT-friendly), scaled by `factor`."""
    return [good_fft(int(factor * math.sqrt(ecutrho_Ry) * math.sqrt(sum(x * x for x in a)) / math.pi) + 1) for a in lattice_bohr]


def derive(eps: float) -> dict:
    """ε -> pw.x settings when none is stated.  PROVISIONAL (RunSpecPlan §1.2): the cutoff for GTH pseudopotentials, which
    are hard (Mn q7 converges at ~280 Ry, IntegrationTests/QE/README.md); ecutrho = 4·ecutwfc (norm-conserving); the grid
    factor from the β-MnO₂ measurement in doc/RunSpecPlan.md §3.3."""
    return {"ecutwfc_Ry": 120 if eps >= 1e-4 else 200 if eps >= 1e-6 else 300, "grid_factor": 1.0 if eps >= 1e-4 else 1.5}


def _shell(Z: int, l: int) -> Optional[str]:
    """The valence shell label QE's HUBBARD card names (3d, 2p, 4f), else None."""
    if l == 2 and _d_shell_n(Z):
        return f"{_d_shell_n(Z)}d"
    if l == 1 and 5 <= Z <= 36:
        return f"{2 if Z <= 10 else 3 if Z <= 18 else 4}p"
    return None


def _qe_rep(s: RunSpec) -> dict:
    return code_block(s, "qe")


def _missing_upf(s: RunSpec) -> Optional[str]:
    els = sorted({st.element for st in s.structure.sites})
    try:
        need = {e: upf_name(e) for e in els}
    except (OSError, KeyError) as e:
        return f"no gth-lda-valence pseudopotential for {e}"
    d = pathlib.Path(_qe_rep(s).get("pseudo_dir") or qchem_src() / "IntegrationTests/QE")
    missing = [e for e, f in need.items() if not (d / f).exists()]
    if missing:
        return (f"{', '.join(need[e] for e in missing)} not in {d}: write it with qchem's "
                + "; ".join(f"gth2upf --element {e} --q {pp_valence()[e]}" for e in missing))
    return None


class QeBackend:
    name = NAME

    def __init__(self, exe: str = PW_EXE, ranks: int = 1):
        self.exe = exe
        self.runner = MpiRunner(ranks=ranks)

    # -- capabilities ------------------------------------------------------
    def capabilities(self) -> Capabilities:
        installed = os.access(self.exe, os.X_OK) and shutil.which("mpirun") is not None
        return Capabilities(NAME, installed=installed, why_absent=None if installed else f"{self.exe} or mpirun not found", features=[
            Capability("structure.kind=crystal", lambda s: s.structure.kind == "crystal",
                       "periodic cells only (a molecule in a box is not translated yet)"),
            Capability("physics.theory=DFT", lambda s: s.physics.theory == "DFT", "pw.x runs DFT here (no HF translated)"),
            Capability("physics.functional", lambda s: s.physics.functional in _XC,
                       "LDA only: the GTH UPF files are the LDA set"),
            Capability("physics.core=pseudopotential", lambda s: s.physics.core == "pseudopotential", "plane waves need pseudopotentials"),
            Capability("physics.relativity", lambda s: s.physics.relativity in ("none", "scalar"), "no spin-orbit translated"),
            Capability("physics.occupations", lambda s: s.physics.occupations.smearing in _SMEARING or not s.physics.spin.polarized,
                       "a polarized run states a smearing (fixed occupations with nspin=2 need a fixed total moment)"),
            Capability("physics.hubbard", lambda s: all(h.form == "Dudarev" and not h.J_eV and h.projector in _PROJECTOR
                       and h.U_eV is not None and all(_shell(s.structure.sites[_tag_index(s, t)].Z, h.l) for t in h.sites)
                       for h in s.physics.hubbard),
                       "QE +U here: Dudarev, J = 0, U given, projector ortho-atomic or atomic, on a d or p valence shell"),
            Capability("accuracy.kmesh", lambda s: s.accuracy.kmesh is not None, "state accuracy.kmesh (no ε -> k-mesh rule yet)"),
            Capability("task.kind=energy", lambda s: s.task.kind == "energy", "single points only so far (hp.x for U is next)"),
            Capability("accuracy.representation", lambda s: not shared_representation(s) and set(_qe_rep(s)) <= _REP_KEYS,
                       "plane waves take no Gaussian basis; QE settings go under representation['qe'] with known keys"),
            Capability("pseudopotential files", lambda s: s.structure.kind != "crystal" or _missing_upf(s) is None,
                       lambda s: _missing_upf(s)),
        ])

    # -- translate ----------------------------------------------------------
    def translate(self, spec: RunSpec) -> Job:
        rep, st, ph, ac = _qe_rep(spec), spec.structure, spec.physics, spec.accuracy
        d = derive(ac.epsilon_Ha_per_atom)
        ecutwfc = rep.get("ecutwfc_Ry", d["ecutwfc_Ry"])
        ecutrho = rep.get("ecutrho_Ry", 4 * ecutwfc)
        factor = rep.get("grid_factor", d["grid_factor"])
        nr = list(rep["nr"]) if "nr" in rep else fft_grid(st.lattice_bohr, ecutrho, factor)
        nat = len(st.sites)
        conv_thr = rep.get("conv_thr", 2 * ac.epsilon_Ha_per_atom * nat / 100)     # Ry, two orders under ε for the cell
        kinds = species_kinds(spec)
        pseudo_dir = rep.get("pseudo_dir") or str(qchem_src() / "IntegrationTests/QE")
        sm = rep.get("starting_magnetization") or {}
        q = pp_valence()

        sysl = [f"ibrav = 0", f"nat = {nat}", f"ntyp = {len(kinds)}", f"ecutwfc = {ecutwfc}", f"ecutrho = {ecutrho}",
                f"input_dft = '{_XC[ph.functional]}'", f"nr1 = {nr[0]}", f"nr2 = {nr[1]}", f"nr3 = {nr[2]}"]
        occ = ph.occupations
        if occ.smearing in _SMEARING:
            sysl += ["occupations = 'smearing'", f"smearing = '{_SMEARING[occ.smearing]}'", f"degauss = {2 * occ.kT_Ha:g}"]
        if ph.spin.polarized:
            sysl.append("nspin = 2")
            for i, k in enumerate(kinds, 1):
                m = sm.get(k["name"], max(-1.0, min(1.0, k["moment"] / q[k["element"]])))
                if m:
                    sysl.append(f"starting_magnetization({i}) = {m:g}")
        if "nbnd" in rep:
            sysl.append(f"nbnd = {rep['nbnd']}")
        el = [f"conv_thr = {conv_thr:.3g}"] + ([f"mixing_beta = {rep['mixing_beta']}"] if "mixing_beta" in rep else [])

        name_of = {i: k["name"] for k in kinds for i in k["sites"]}
        lines = [" &control", "    calculation = 'scf'", "    prefix = 'job'", f"    pseudo_dir = '{pseudo_dir}/'",
                 "    outdir = './tmp/'", "    verbosity = 'high'", " /", " &system"]
        lines += [f"    {x}," for x in sysl] + [" /", " &electrons"] + [f"    {x}" for x in el] + [" /", "CELL_PARAMETERS {bohr}"]
        lines += [" " + " ".join(f"{x:.10f}" for x in a) for a in st.lattice_bohr]
        lines += ["ATOMIC_SPECIES"] + [f" {k['name']} {_MASS[k['Z'] - 1]} {upf_name(k['element'])}" for k in kinds]
        lines += ["ATOMIC_POSITIONS {crystal}"] + [f" {name_of[i]} " + " ".join(f"{x:.10f}" for x in s.frac)
                                                    for i, s in enumerate(st.sites)]
        lines += ["K_POINTS {automatic}", " " + " ".join(map(str, ac.kmesh)) + " 0 0 0"]
        if ph.hubbard:
            projs = {h.projector for h in ph.hubbard}
            assert len(projs) == 1, "one projector per run (QE's HUBBARD card)"
            lines.append(f"HUBBARD {{{_PROJECTOR[projs.pop()]}}}")
            for h in ph.hubbard:
                for t in h.sites:
                    i = _tag_index(spec, t)
                    lines.append(f"U {name_of[i]}-{_shell(st.sites[i].Z, h.l)} {h.U_eV:g}")
        deck = "\n".join(lines) + "\n"
        default = fft_grid(st.lattice_bohr, ecutrho)
        resolved = {
            "cutoff": f"ecutwfc {ecutwfc} Ry, ecutrho {ecutrho} Ry",
            "FFT grid": f"{nr[0]}×{nr[1]}×{nr[2]} (stated; pw.x's minimal grid is {default[0]}×{default[1]}×{default[2]})",
            "pseudopotentials": ", ".join(sorted({upf_name(k['element']) for k in kinds})) + " (qchem GTH, gth2upf)",
            "XC": _XC[ph.functional],
            "k-points": "×".join(map(str, ac.kmesh)),
            "conv_thr": f"{conv_thr:.3g} Ry",
        }
        material = (st.source or {}).get("materials") or spec.label
        return Job(NAME, spec, native={"material": material, "stem": spec.label}, resolved=resolved,
                   files={"deck": deck.encode()}, command=(self.exe,))

    # -- run + parse --------------------------------------------------------
    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result:
        d = run_dir("qe", job.native["material"])
        prefix = next_prefix(d, job.native["stem"])
        deck = job.files["deck"].decode().replace("    prefix = 'job'\n", f"    prefix = '{prefix}'\n", 1)
        tail = QeTail(on_step)
        rc, out = self.runner.run(job.command + ("-in", f"{prefix}.in"), d, prefix, {f"{prefix}.in": deck.encode()},
                                  record={"spec": job.spec.to_dict(), "resolved": job.resolved, "backend": NAME},
                                  on_line=tail.feed)
        for scratch in (d / "tmp").glob(f"{prefix}.*"):            # wavefunctions / save dir: scratch (D-RUNDATA)
            shutil.rmtree(scratch, ignore_errors=True) if scratch.is_dir() else scratch.unlink(missing_ok=True)
        parsed = parse_output(out.read_text())
        if rc != 0 and parsed["energy"] is None:
            raise RuntimeError(f"pw.x exited {rc}; see {out}")
        return Result(converged=parsed["converged"], energy_Ha=parsed["energy"], structure=data_structure(job.spec),
                      scf=tail.steps, polarized=job.spec.physics.spin.polarized, resolved=job.resolved,
                      provenance={"code": NAME, "version": parsed["version"], "output": str(out), "returncode": rc})


def _tag_index(spec: RunSpec, tag: str) -> int:
    return next(i for i, s in enumerate(spec.structure.sites) if s.tag == tag)


_E = re.compile(r"^[!\s]\s+total energy\s+=\s+(-?\d+\.\d+) Ry")
_ACC = re.compile(r"^\s+estimated scf accuracy\s+<\s+(\S+) Ry")


class QeTail:
    """pw.x's per-iteration `total energy` / `estimated scf accuracy` lines -> SCFSteps (energies in Ha)."""

    def __init__(self, on_step: Optional[StepSink] = None):
        self.on_step = on_step
        self.steps: list[SCFStep] = []
        self._E: Optional[float] = None

    def feed(self, line: str) -> None:
        if m := _E.match(line):
            self._E = float(m.group(1)) / 2
        elif (m := _ACC.match(line)) and self._E is not None:
            acc = float(m.group(1)) / 2
            prev = self.steps[-1].energy if self.steps else None
            s = SCFStep(len(self.steps) + 1, self._E, abs(self._E - prev) if prev is not None else abs(self._E), acc, acc)
            self.steps.append(s)
            self._E = None
            if self.on_step:
                self.on_step(s)


def parse_output(text: str) -> dict:
    """The neutral numbers out of a pw.x output: the final total energy (Ha), converged, version."""
    m = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+) Ry", text, re.M)
    v = re.search(r"Program PWSCF (v\.\S+)", text)
    return {"energy": float(m[-1]) / 2 if m else None,
            "converged": "convergence has been achieved" in text and "convergence NOT achieved" not in text,
            "version": v.group(1) if v else None}
