"""
qviz.backends.abinit -- ABINIT (plane waves) as a back end (doc/RunSpecPlan.md §3.4).

translate(spec) writes one ABINIT input; run() launches it through qviz.runner.MpiRunner -- ALWAYS
`mpirun -np N abinit <input>`, never the bare binary -- in ~/Code/abinit-runs/<Material>/, tails the log
into the live SCF trace (the `ETOT n E` lines), and parses the .abo into a neutral Result.

The same rules as the QE translator, because they are the same physics:
  * the FFT grid `ngfft` is ALWAYS stated (pw.x's minimal grid scaled by the same ε-derived factor -- an ABINIT run and a
    QE run of one spec are evaluated on the same grid);
  * the pseudopotentials are OUR GTH set, each entry of qchem's GTH database re-wrapped as ABINIT's native pspcod 10
    (qviz.backends.gth), so ABINIT, QE, CP2K and SOLID² run the same pseudopotential.  (Not the gth2upf UPFs QE reads:
    abinit 10.8's UPF2 reader requires a linear radial mesh from r = 0 and aborts on gth2upf's logarithmic one --
    measured 2026-10-31);
  * LDA is libxc's Dirac exchange + VWN (`ixc -1007`), as SOLID², CP2K and QE's `sla+vwn`;
  * a run that exhausts nstep is a failed run.
Not translated: +U (ABINIT's DFT+U is PAW-only; these are norm-conserving pseudopotentials) and molecules.

A STATED representation for ABINIT lives under accuracy.representation["abinit"]:
    {"ecut_Ha": 150, "ngfft": [96, 96, 60] | "grid_factor": 1.5, "nband": 30, "toldfe": 1e-10,
     "nstep": 100, "diemac": 12.0}
Every key is optional; an unstated one is derived from ε (the QE back end's PROVISIONAL table, in Ha).
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
from typing import Optional

from ..capability import Capabilities, Capability
from ..data import SCFCell, SCFRow
from ..runner import MpiRunner, next_prefix, run_dir
from ..spec import RunSpec
from .base import Job, Result, StepSink, code_block, data_structure, shared_representation
from . import gth
from .qe import derive, fft_grid

NAME = "ABINIT"
ABINIT_EXE = os.environ.get("ABINIT_EXE", str(pathlib.Path.home() / "Code/abinit/build/src/98_main/abinit"))

_IXC = {"LDA": -1007}                                  # libxc LDA_X + LDA_C_VWN
_OCCOPT = {"none": 1, "fermi-dirac": 3, "marzari-vanderbilt": 4, "gaussian": 7}
_REP_KEYS = {"ecut_Ha", "ngfft", "grid_factor", "nband", "toldfe", "nstep", "diemac"}


def _rep(s: RunSpec) -> dict:
    return code_block(s, "abinit")


def _missing_pp(s: RunSpec) -> Optional[str]:
    """None when every element has a gth-lda-valence entry in qchem's GTH database, else why not."""
    for e in sorted({st.element for st in s.structure.sites}):
        try:
            gth.gth_entry(e)
        except (OSError, KeyError) as x:
            return f"no gth-lda-valence GTH entry for {e}: {x}"
    return None


class AbinitBackend:
    name = NAME

    def __init__(self, exe: str = ABINIT_EXE, ranks: int = 1):
        self.exe = exe
        # NOT --force-mpirun: that is the test-suite driver's flag (runtests.py); abinit 10.8 rejects it as an
        # unsupported option and exits without running (measured 2026-10-31).  mpirun alone avoids the singleton hang.
        self.runner = MpiRunner(ranks=ranks)

    # -- capabilities ------------------------------------------------------
    def capabilities(self) -> Capabilities:
        installed = os.access(self.exe, os.X_OK) and shutil.which("mpirun") is not None
        return Capabilities(NAME, installed=installed, why_absent=None if installed else f"{self.exe} or mpirun not found", features=[
            Capability("structure.kind=crystal", lambda s: s.structure.kind == "crystal",
                       "periodic cells only (a molecule in a box is not translated yet)"),
            Capability("physics.theory=DFT", lambda s: s.physics.theory == "DFT", "ABINIT runs DFT here (no HF translated)"),
            Capability("physics.functional", lambda s: s.physics.functional in _IXC, "LDA only: the gth-lda-valence set is the LDA one"),
            Capability("physics.core=pseudopotential", lambda s: s.physics.core == "pseudopotential", "plane waves need pseudopotentials"),
            Capability("physics.relativity", lambda s: s.physics.relativity in ("none", "scalar"), "no spin-orbit translated"),
            Capability("physics.hubbard", lambda s: not s.physics.hubbard,
                       "ABINIT's DFT+U is PAW-only, and these are norm-conserving pseudopotentials"),
            Capability("physics.occupations", lambda s: s.physics.occupations.smearing != "none" or not s.physics.spin.polarized,
                       "a polarized run states a smearing"),
            Capability("accuracy.kmesh", lambda s: s.accuracy.kmesh is not None, "state accuracy.kmesh (no ε -> k-mesh rule yet)"),
            Capability("task.kind=energy", lambda s: s.task.kind == "energy", "single points only so far"),
            Capability("accuracy.representation", lambda s: not shared_representation(s) and set(_rep(s)) <= _REP_KEYS,
                       "plane waves take no Gaussian basis; ABINIT settings go under representation['abinit'] with known keys"),
            Capability("pseudopotentials", lambda s: s.structure.kind != "crystal" or _missing_pp(s) is None,
                       lambda s: _missing_pp(s)),
        ])

    # -- translate ----------------------------------------------------------
    def translate(self, spec: RunSpec) -> Job:
        rep, st, ph, ac = _rep(spec), spec.structure, spec.physics, spec.accuracy
        d = derive(ac.epsilon_Ha_per_atom)
        ecut = rep.get("ecut_Ha", d["ecutwfc_Ry"] / 2)
        ngfft = list(rep["ngfft"]) if "ngfft" in rep else fft_grid(st.lattice_bohr, 8 * ecut, rep.get("grid_factor", d["grid_factor"]))
        nat = len(st.sites)
        toldfe = rep.get("toldfe", ac.epsilon_Ha_per_atom * nat / 100)
        els = list(dict.fromkeys(s.element for s in st.sites))
        lines = [f"# {spec.label} -- written by qviz (doc/RunSpecPlan.md §3.4)", "acell 3*1.0", "rprim"]
        lines += ["   " + " ".join(f"{x:.10f}" for x in a) for a in st.lattice_bohr]
        lines += [f"natom {nat}", f"ntypat {len(els)}", "znucl " + " ".join(str(next(s.Z for s in st.sites if s.element == e)) for e in els),
                  "typat " + " ".join(str(els.index(s.element) + 1) for s in st.sites), "xred"]
        lines += ["   " + " ".join(f"{x:.10f}" for x in s.frac) for s in st.sites]
        pps = {gth.abinit_name(e): gth.to_abinit_psp10(e, next(x.Z for x in st.sites if x.element == e), _IXC[ph.functional]).encode()
               for e in els}                                    # written beside the input (pp_dirpath ".")
        lines += ['pp_dirpath "."', 'pseudos "' + ", ".join(pps) + '"',
                  f"ixc {_IXC[ph.functional]}", f"ecut {ecut}", "ngfft " + " ".join(map(str, ngfft))]
        occ = ph.occupations
        lines.append(f"occopt {_OCCOPT[occ.smearing]}")
        if occ.smearing != "none":
            lines.append(f"tsmear {occ.kT_Ha:g}")
        if ph.spin.polarized:
            lines.append("nsppol 2")
            m = ph.spin.site_moments or (0.0,) * nat
            lines += ["spinat"] + [f"   0 0 {x:g}" for x in m]
        if "nband" in rep:
            lines.append(f"nband {rep['nband']}")
        lines += ["kptopt 1", "ngkpt " + " ".join(map(str, ac.kmesh)), "nshiftk 1", "shiftk 0 0 0",
                  f"nstep {rep.get('nstep', (ac.scf or {}).get('max_iter', 100))}", f"toldfe {toldfe:.3g}",
                  "prtwf 0", "prtden 0", "prteig 0"]
        if "diemac" in rep:
            lines.append(f"diemac {rep['diemac']}")
        default = fft_grid(st.lattice_bohr, 8 * ecut)
        resolved = {
            "cutoff": f"ecut {ecut} Ha (= ecutwfc {2 * ecut} Ry)",
            "FFT grid": f"{ngfft[0]}×{ngfft[1]}×{ngfft[2]} (stated; the minimal grid is {default[0]}×{default[1]}×{default[2]})",
            "pseudopotentials": ", ".join(pps) + f" (qchem {gth.database().name}, re-wrapped as pspcod 10)",
            "XC": f"ixc {_IXC[ph.functional]} (libxc Dirac + VWN)",
            "k-points": "×".join(map(str, ac.kmesh)) + " Γ-centred",
            "toldfe": f"{toldfe:.3g} Ha",
        }
        material = (st.source or {}).get("materials") or spec.label
        return Job(NAME, spec, native={"material": material, "stem": spec.label}, resolved=resolved,
                   files={"deck": ("\n".join(lines) + "\n").encode()} | pps, command=(self.exe,))

    # -- run + parse --------------------------------------------------------
    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result:
        d = run_dir("abinit", job.native["material"])
        prefix = next_prefix(d, job.native["stem"])
        tail = AbinitTail(on_step, abinit_columns(job.spec.physics.spin.polarized))
        files = {f"{prefix}.abi": job.files["deck"]} | {k: v for k, v in job.files.items() if k != "deck"}
        rc, out = self.runner.run(job.command + (f"{prefix}.abi",), d, prefix, files,
                                  record={"spec": job.spec.to_dict(), "resolved": job.resolved, "backend": NAME},
                                  on_line=tail.feed)
        abo = d / f"{prefix}.abo"
        parsed = parse_output(abo.read_text() if abo.exists() else out.read_text())
        if rc != 0 and parsed["energy"] is None:
            raise RuntimeError(f"abinit exited {rc}; see {out}")
        return Result(converged=parsed["converged"], energy_Ha=parsed["energy"], structure=data_structure(job.spec),
                      scf=tail.steps, polarized=job.spec.physics.spin.polarized, resolved=job.resolved,
                      provenance={"code": NAME, "version": parsed["version"], "output": str(abo if abo.exists() else out), "returncode": rc})


_ETOT = re.compile(r"^ ETOT\s+(\d+)\s+(-?\d+\.\d+)((?:\s+\S+)*)\s*$")
_HEADER = re.compile(r"^\s+iter\s+Etot\(hartree\)")
# ABINIT's own column titles.  Its log (what the tail sees live) carries no header -- only the .abo does -- so the run
# says which columns to expect: vres2 for a norm-conserving SCF on the potential (iscf < 10), the spin-sphere pair
# difmagsph/maxmagsph in a polarized run.  A header line, when one passes by, overrides the expectation.
_LOG_SCALE = {"deltaE(h)", "residm", "vres2", "nres2", "difmagsph", "diffor"}


def abinit_columns(polarized: bool) -> tuple[str, ...]:
    return ("Etot(hartree)", "deltaE(h)", "residm", "vres2") + (("difmagsph", "maxmagsph") if polarized else ())


class AbinitTail:
    """ABINIT's `ETOT n Etot deltaE residm vres2 ...` lines -> SCFRows, one cell per column under ABINIT's own titles."""

    def __init__(self, on_step: Optional[StepSink] = None, columns: tuple[str, ...] = abinit_columns(False)):
        self.on_step = on_step
        self.columns = columns
        self.steps: list[SCFRow] = []

    def feed(self, line: str) -> None:
        if _HEADER.match(line):
            self.columns = tuple(line.split()[1:])
        elif m := _ETOT.match(line):
            words = [m.group(2)] + m.group(3).split()
            titles = self.columns + tuple(f"column {k + 1}" for k in range(len(self.columns), len(words)))
            s = SCFRow(len(self.steps) + 1, float(m.group(2)), tuple(
                SCFCell(title, float(w), "log" if title in _LOG_SCALE else "linear", w) for title, w in zip(titles, words)))
            self.steps.append(s)
            if self.on_step:
                self.on_step(s)


# m_common.F90 scprqt: tolvrs/tolwfr print `=>converged.`, toldfe prints `, etot is converged :` (no arrow).
_CONVERGED = re.compile(r"^ At SCF step\s+\d+.*(=>converged\.|etot is converged)", re.M)


def parse_output(text: str) -> dict:
    """The neutral numbers out of an ABINIT .abo: the final etotal (Ha, the end-of-run summary's full precision), converged,
    version.  In a multi-dataset run a dataset with no ground state prints etotal 0: the last NON-ZERO one is the energy."""
    m = [x for x in re.findall(r"^\s+etotal\d*\s+(-?\d\.\d+E[+-]\d+)", text, re.M) if float(x) != 0.0]
    v = re.search(r"^\.Version (\S+) of ABINIT", text, re.M)
    return {"energy": float(m[-1]) if m else None,
            "converged": bool(_CONVERGED.search(text)) and "not enough SCF cycles" not in text,
            "version": v.group(1) if v else None}
