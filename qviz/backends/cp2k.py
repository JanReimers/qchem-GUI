"""
qviz.backends.cp2k -- CP2K (Quickstep GPW) as a back end (doc/RunSpecPlan.md §3.2).

translate(spec) writes ONE &GLOBAL + &FORCE_EVAL deck in the shape of qchem's banked oracle decks
(IntegrationTests/CP2K/*.inp); run() launches it through qviz.runner.MpiRunner -- ALWAYS
`mpirun -np N cp2k.psmp -i <deck>`, never the bare binary -- in ~/Code/cp2k-runs/<Material>/, tails stdout
into the live SCF trace, and parses the output into a neutral Result.

Rules the translator enforces (each cost a wrong number once, qchem doc/OldPlans/GPWPlan.md, doc/Benchmark.md):
  * EPS_PGF_ORB is ALWAYS stated (CP2K's default once produced a 3.5 Ha "oracle" on a diffuse basis);
  * no IGNORE_CONVERGENCE_FAILURE: a capped SCF is a FAILED run here, as it is in qchem's tests;
  * U is Dudarev, shell-averaged, on Löwdin projectors (CP2K has no orbital-resolved U) -- anything else is refused.

A STATED representation for CP2K lives under accuracy.representation["cp2k"]:
    {"basis_file": path, "potential_file": path,
     "kinds": {"<tag or element>": {"basis": "SIPP-SR", "potential": "GTH-PADE-q4"}},
     "cutoff_Ry": 80, "rel_cutoff_Ry": 60, "ngrids": 4, "eps_default": 1e-12, "eps_pgf_orb": 1e-12, "added_mos": 20}
Every key is optional; an unstated one is derived from ε (the PROVISIONAL table in `derive`).  The block is CP2K's
alone: the other codes ignore it (base.CODE_BLOCKS), so a cross-code spec can tune each code.  A SHARED
representation.basis (a SOLID² Gaussian basis) is handed to CP2K converted from SOLID²'s own file (gaussian_basis).

Fields (density / orbitals on a grid) are not parsed yet: CP2K's cube files live on the cell's (generally
non-orthogonal) grid, and the frontends' ScalarField is axis-aligned.  The Result carries energies, the trace and
the resolved settings.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
from typing import Optional

from ..capability import Capabilities, Capability
from ..data import SCFStep
from ..runner import MpiRunner, next_prefix, run_dir
from ..spec import RunSpec
from . import gaussian_basis
from .base import Job, Result, StepSink, code_block, data_structure, shared_representation

NAME = "CP2K"
CP2K_EXE = os.environ.get("CP2K_EXE", "/usr/bin/cp2k.psmp")
CP2K_DATA = pathlib.Path(os.environ.get("CP2K_DATA", pathlib.Path.home() / "Code/cp2k/data"))

_XC = {"LDA": ("LDA_X", "LDA_C_VWN"), "PBE": ("PBE",)}
_DEFAULT_PP = {"LDA": "GTH-PADE", "PBE": "GTH-PBE"}
_REP_KEYS = {"basis_file", "potential_file", "kinds", "cutoff_Ry", "rel_cutoff_Ry", "ngrids", "eps_default", "eps_pgf_orb",
             "added_mos"}


def _d_shell_n(Z: int) -> Optional[int]:
    """The principal quantum number of a transition metal's open d shell (3d / 4d / 5d), else None."""
    return 3 if 21 <= Z <= 30 else 4 if 39 <= Z <= 48 else 5 if 72 <= Z <= 80 else None


def derive(eps: float) -> dict:
    """ε -> CP2K's accuracy settings when none is stated.  PROVISIONAL rules of thumb (RunSpecPlan §1.2), to be
    replaced by a measured ε-calibration curve: EPS_SCF two orders under ε (it is nearly free), the screening far
    under it, EPS_PGF_ORB stated (never CP2K's default), the cutoff stepped with ε."""
    eps_default = min(1e-12, eps * 1e-7)
    return {"cutoff_Ry": 80 if eps >= 1e-4 else 120 if eps >= 1e-6 else 200, "rel_cutoff_Ry": 60, "ngrids": 4,
            "eps_default": eps_default, "eps_pgf_orb": 100 * eps_default, "eps_scf": eps / 100}


def _cp2k_rep(s: RunSpec) -> dict:
    return code_block(s, "cp2k")


def _shared_basis(s: RunSpec) -> Optional[str]:
    """The SOLID² basis the spec states (representation.basis), which CP2K is handed converted (gaussian_basis)."""
    return shared_representation(s).get("basis")


def _shared_basis_problem(s: RunSpec) -> Optional[str]:
    name = _shared_basis(s)
    if name is None:
        return None
    if "basis_file" in _cp2k_rep(s) or any("basis" in k for k in (_cp2k_rep(s).get("kinds") or {}).values()):
        return "a shared basis and a CP2K basis are both stated"
    return gaussian_basis.why_not_cp2k(name, [st.element for st in s.structure.sites])


def _kinds(spec: RunSpec) -> list[dict]:
    """One &KIND per distinct (element, tag, moment): a tag names its kind (Mn1, Mn2); untagged sites of one element
    and one moment share the element's kind."""
    moments = spec.physics.spin.site_moments or (0.0,) * len(spec.structure.sites)
    kinds: dict[tuple, dict] = {}
    names: list[str] = []
    site_kind: list[str] = []
    for s, m in zip(spec.structure.sites, moments):
        key = (s.element, s.tag, float(m))
        if key not in kinds:
            name = s.tag or s.element
            if name in names:                               # untagged, same element, different moment
                name = f"{s.element}{sum(1 for n in names if n.startswith(s.element)) + 1}"
            names.append(name)
            kinds[key] = {"name": name, "element": s.element, "Z": s.Z, "moment": float(m), "tag": s.tag}
        site_kind.append(kinds[key]["name"])
    spec_kinds = list(kinds.values())
    for k in spec_kinds:
        k["sites"] = [i for i, n in enumerate(site_kind) if n == k["name"]]
    return spec_kinds


class Cp2kBackend:
    name = NAME

    def __init__(self, exe: str = CP2K_EXE, ranks: int = 1):
        self.exe = exe
        self.runner = MpiRunner(ranks=ranks)

    # -- capabilities ------------------------------------------------------
    def capabilities(self) -> Capabilities:
        installed = os.access(self.exe, os.X_OK) and shutil.which("mpirun") is not None
        moments = lambda s: s.physics.spin.site_moments or ()
        return Capabilities(NAME, installed=installed,
                            why_absent=None if installed else f"{self.exe} or mpirun not found", features=[
            Capability("physics.theory=DFT", lambda s: s.physics.theory == "DFT", "Quickstep runs DFT here (no HF translated)"),
            Capability("physics.functional", lambda s: s.physics.functional in _XC, "functionals translated: LDA, PBE"),
            Capability("physics.core=pseudopotential", lambda s: s.physics.core == "pseudopotential",
                       "GPW needs pseudopotentials (GAPW all-electron is not translated)"),
            Capability("physics.relativity", lambda s: s.physics.relativity in ("none", "scalar"), "no spin-orbit"),
            Capability("physics.hubbard: Dudarev on Löwdin", lambda s: all(h.form == "Dudarev" and h.projector == "lowdin"
                       and h.U_eV is not None for h in s.physics.hubbard),
                       "CP2K's +U is Dudarev, shell-averaged, on Löwdin projectors (projector: 'lowdin'), with U given"),
            Capability("physics.spin.site_moments on d shells", lambda s: all(m == 0 or _d_shell_n(st.Z)
                       for st, m in zip(s.structure.sites, moments(s))),
                       "the &BS initial-moment guess is translated for transition-metal d shells only"),
            Capability("physics.occupations", lambda s: s.physics.occupations.smearing in ("none", "fermi-dirac"),
                       "smearing translated: none, fermi-dirac"),
            Capability("task.kind=energy", lambda s: s.task.kind == "energy",
                       "single points only so far (GEO_OPT / CELL_OPT next; CP2K cannot compute U)"),
            Capability("accuracy.representation", lambda s: set(shared_representation(s)) <= {"basis"}
                       and set(_cp2k_rep(s)) <= _REP_KEYS,
                       "a stated representation is a shared basis, or CP2K settings under representation['cp2k'] with known keys"),
            Capability("accuracy.representation.basis converts", lambda s: _shared_basis_problem(s) is None,
                       lambda s: _shared_basis_problem(s)),
        ])

    # -- translate ----------------------------------------------------------
    def translate(self, spec: RunSpec) -> Job:
        st, ph, ac = spec.structure, spec.physics, spec.accuracy
        rep = _cp2k_rep(spec)
        d = derive(ac.epsilon_Ha_per_atom)
        shared = _shared_basis(spec)
        if shared:
            # a hard primitive needs the grid: F₂ valence_lowq_vb (α_max 40) is 1.8 mHa off at 200 Ry, 0.4 at 400, 2e-5 at
            # 800 and converged by 1200 (2026-10-30).  PROVISIONAL: 20·α_max Ry, never below the ε table.
            d["cutoff_Ry"] = max(d["cutoff_Ry"], 20 * gaussian_basis.max_exponent(shared, [s.element for s in st.sites]))
        d |= {k: v for k, v in rep.items() if k in derive(1.0)}
        scf = ac.scf or {}
        files = {}
        if shared:                                          # the SAME basis SOLID² reads, in CP2K's format, beside the deck
            files["basis"] = gaussian_basis.to_cp2k(shared, [s.element for s in st.sites]).encode()
            stated_basis = gaussian_basis.cp2k_name(shared)
        q = gaussian_basis.pp_valence(shared) if shared else {}

        def default_pp(el: str) -> str:
            """The GTH family for the functional; with a shared basis, the valence charge its blocks were made for."""
            return _DEFAULT_PP[ph.functional] + (f"-q{q[el.upper()]}" if el.upper() in q else "")
        basis_file = "job.basis" if shared else rep.get("basis_file") or str(CP2K_DATA / "BASIS_MOLOPT")
        pot_file = rep.get("potential_file") or str(CP2K_DATA / "GTH_POTENTIALS")
        kinds = _kinds(spec)
        stated_kinds = rep.get("kinds") or {}
        polarized = ph.spin.polarized
        if st.kind == "molecule":
            mult = ph.spin.multiplicity or (1 if spec.num_electrons % 2 == 0 else 2)
        else:
            mult = 1 + round(abs(sum(ph.spin.site_moments or ())))
        hub_sites = {t for h in ph.hubbard for t in h.sites}
        prefix = "job"                                      # renamed per run in run(); translate stays pure

        L = []
        w = L.append
        w("&GLOBAL"); w(f"  PROJECT {prefix}"); w("  RUN_TYPE ENERGY"); w("  PRINT_LEVEL MEDIUM"); w("&END GLOBAL")
        w("&FORCE_EVAL"); w("  METHOD Quickstep"); w("  &DFT")
        w(f"    BASIS_SET_FILE_NAME {basis_file}"); w(f"    POTENTIAL_FILE_NAME {pot_file}")
        if polarized:
            w("    UKS"); w(f"    MULTIPLICITY {mult}")
        if ph.hubbard:
            w("    PLUS_U_METHOD LOWDIN")
        w("    &QS"); w("      METHOD GPW"); w(f"      EPS_DEFAULT {d['eps_default']:.1E}")
        w(f"      EPS_PGF_ORB {d['eps_pgf_orb']:.1E}"); w("    &END QS")
        w("    &MGRID"); w(f"      CUTOFF {d['cutoff_Ry']:g}"); w(f"      REL_CUTOFF {d['rel_cutoff_Ry']:g}")
        w(f"      NGRIDS {int(d['ngrids'])}"); w("    &END MGRID")
        w("    &XC"); w("      &XC_FUNCTIONAL")
        for f in _XC[ph.functional]:
            w(f"        &{f}"); w(f"        &END {f}")
        w("      &END XC_FUNCTIONAL"); w("    &END XC")
        w("    &POISSON")
        if st.kind == "crystal":
            w("      PERIODIC XYZ")
        else:
            w("      PERIODIC NONE"); w("      PSOLVER MT")
        w("    &END POISSON")
        w("    &SCF"); w("      SCF_GUESS ATOMIC"); w(f"      EPS_SCF {d['eps_scf']:.1E}")
        w(f"      MAX_SCF {int(scf.get('max_iter', 100))}")
        # ADDED_MOS bounds the smeared occupations: too few TRUNCATES them and moves the energy (MnO AFM-II +U, kT 5e-3:
        # 16 instead of the oracle's 20 gave +15 μHa, 2026-10-30).  Generous by default; stated to reproduce an oracle.
        added_mos = int(rep.get("added_mos", 8 * len(st.sites)))
        if ph.occupations.smearing == "fermi-dirac" or "added_mos" in rep:
            w(f"      ADDED_MOS {added_mos}")
        if ph.occupations.smearing == "fermi-dirac":
            w("      &MIXING T"); w("        METHOD BROYDEN_MIXING"); w("        ALPHA 0.2"); w("        BETA 1.5")
            w("        NBUFFER 8"); w("      &END MIXING")
            w("      &SMEAR ON"); w("        METHOD FERMI_DIRAC")
            w(f"        ELECTRONIC_TEMPERATURE [au_temp] {ph.occupations.kT_Ha:.1E}"); w("      &END SMEAR")
        w("    &END SCF")
        if st.kind == "crystal" and ac.kmesh and tuple(ac.kmesh) != (1, 1, 1):
            w("    &KPOINTS"); w(f"      SCHEME MONKHORST-PACK {' '.join(str(n) for n in ac.kmesh)}"); w("    &END KPOINTS")
        w("  &END DFT")
        w("  &SUBSYS"); w("    &CELL")
        if st.kind == "crystal":
            for ax, row in zip("ABC", st.lattice_bohr):
                w(f"      {ax} [bohr] {' '.join(f'{x:.10g}' for x in row)}")
            w("      PERIODIC XYZ")
        else:
            xyz = [s.xyz for s in st.sites]
            box = max(max(p[i] for p in xyz) - min(p[i] for p in xyz) for i in range(3)) + 20.0   # 10 bohr vacuum a side
            w(f"      ABC [bohr] {box:.4f} {box:.4f} {box:.4f}"); w("      PERIODIC NONE")
        w("    &END CELL"); w("    &COORD")
        site_kind = {i: k["name"] for k in kinds for i in k["sites"]}
        if st.kind == "crystal":
            w("      SCALED")
            for i, s in enumerate(st.sites):
                w(f"      {site_kind[i]} {' '.join(f'{x:.10g}' for x in s.frac)}")
        else:
            w("      UNIT bohr")
            for i, s in enumerate(st.sites):
                w(f"      {site_kind[i]} {' '.join(f'{x:.10g}' for x in s.xyz)}")
        w("    &END COORD")
        if st.kind == "molecule":
            w("    &TOPOLOGY"); w("      &CENTER_COORDINATES"); w("      &END CENTER_COORDINATES"); w("    &END TOPOLOGY")
        for k in kinds:
            sk = stated_kinds.get(k["name"]) or stated_kinds.get(k["element"]) or {}
            w(f"    &KIND {k['name']}")
            if k["name"] != k["element"]:
                w(f"      ELEMENT {k['element']}")
            w(f"      BASIS_SET {stated_basis if shared else sk.get('basis', 'DZVP-MOLOPT-SR-GTH')}")
            w(f"      POTENTIAL {sk.get('potential') or default_pp(k['element'])}")
            for h in ph.hubbard:
                if k["tag"] in h.sites:
                    w("      &DFT_PLUS_U"); w(f"        L {h.l}"); w(f"        U_MINUS_J [eV] {h.U_eV - h.J_eV:g}")
                    w("      &END DFT_PLUS_U")
            if polarized and k["moment"]:
                n, m = _d_shell_n(k["Z"]), round(abs(k["moment"]))
                up, dn = (m, -m) if k["moment"] > 0 else (-m, m)
                w("      &BS")
                for chan, nel in (("ALPHA", up), ("BETA", dn)):
                    w(f"        &{chan}"); w(f"          NEL {nel}"); w("          L 2"); w(f"          N {n}"); w(f"        &END {chan}")
                w("      &END BS")
            w("    &END KIND")
        w("  &END SUBSYS"); w("&END FORCE_EVAL")
        deck = "\n".join(L) + "\n"

        resolved = {
            "cutoff": f"{d['cutoff_Ry']:g} Ry (rel {d['rel_cutoff_Ry']:g}, {int(d['ngrids'])} grids)"
                      + (" stated" if "cutoff_Ry" in rep else " from ε and the basis's hardest exponent, provisional"
                         if shared else " from ε, provisional"),
            "EPS_DEFAULT / EPS_PGF_ORB": f"{d['eps_default']:.1E} / {d['eps_pgf_orb']:.1E}",
            "EPS_SCF": f"{d['eps_scf']:.1E} (ε/100)",
            "spin": f"UKS, multiplicity {mult}" if polarized else "RKS",
            "smearing": (f"Fermi-Dirac kT {ph.occupations.kT_Ha:.1E} Ha, ADDED_MOS {added_mos}"
                         + (" stated" if "added_mos" in rep else " (8 per site)")) if ph.occupations.smearing == "fermi-dirac" else "none",
            "kinds": ", ".join(f"{k['name']}: {stated_basis + ' (SOLID² ' + shared + '.bsd, converted)' if shared else (stated_kinds.get(k['name']) or stated_kinds.get(k['element']) or {}).get('basis', 'DZVP-MOLOPT-SR-GTH')}"
                               f" / {(stated_kinds.get(k['name']) or stated_kinds.get(k['element']) or {}).get('potential') or default_pp(k['element'])}"
                               for k in kinds),
            "k-points": "MP " + "x".join(map(str, ac.kmesh)) if ac.kmesh and tuple(ac.kmesh) != (1, 1, 1) else "Γ",
            "launch": f"mpirun -np {self.runner.ranks} {self.exe} -i <deck>",
        }
        material = (st.source or {}).get("materials") or spec.label
        return Job(NAME, spec, native={"material": material, "stem": spec.label},
                   resolved=resolved, files={"deck": deck.encode()} | files, command=(self.exe,))

    # -- run + parse --------------------------------------------------------
    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result:
        d = run_dir("cp2k", job.native["material"])
        prefix = next_prefix(d, job.native["stem"])
        deck = job.files["deck"].decode().replace("  PROJECT job\n", f"  PROJECT {prefix}\n", 1)
        files = {f"{prefix}.inp": None}
        if "basis" in job.files:
            deck = deck.replace("BASIS_SET_FILE_NAME job.basis\n", f"BASIS_SET_FILE_NAME {prefix}.basis\n", 1)
            files[f"{prefix}.basis"] = job.files["basis"]
        files[f"{prefix}.inp"] = deck.encode()
        tail = Cp2kTail(on_step)
        rc, out = self.runner.run(job.command + ("-i", f"{prefix}.inp"), d, prefix, files,
                                  record={"spec": job.spec.to_dict(), "resolved": job.resolved, "backend": NAME},
                                  on_line=tail.feed)
        parsed = parse_output(out.read_text())
        if rc != 0 and parsed["energy"] is None:
            raise RuntimeError(f"cp2k exited {rc}; see {out}")
        return Result(converged=parsed["converged"], energy_Ha=parsed["energy"], structure=data_structure(job.spec),
                      scf=tail.steps, resolved=job.resolved,
                      provenance={"code": NAME, "version": parsed["version"], "output": str(out), "returncode": rc})


_STEP = re.compile(r"^\s+(\d+)\s+\S.*\s(\S+)\s+(-?\d+\.\d+)\s+(\S+)\s*$")


class Cp2kTail:
    """Turns CP2K's SCF table rows into SCFSteps as the lines arrive (the live trace).  Iterations are numbered
    across outer loops, so the trace stays monotone."""

    def __init__(self, on_step: Optional[StepSink] = None):
        self.on_step = on_step
        self.steps: list[SCFStep] = []
        self._in_table = False

    def feed(self, line: str) -> None:
        if "Step     Update method" in line:
            self._in_table = True
            return
        if self._in_table:
            m = _STEP.match(line)
            if m:
                try:
                    conv, E, dE = float(m.group(2)), float(m.group(3)), float(m.group(4))
                except ValueError:
                    return
                s = SCFStep(len(self.steps) + 1, E, abs(dE), conv, conv)   # CP2K's "Convergence" = max density change
                self.steps.append(s)
                if self.on_step:
                    self.on_step(s)
            elif line.strip().startswith(("***", "Leaving", "outer SCF", "Core Hamiltonian")) or "converged" in line:
                self._in_table = False


def parse_output(text: str) -> dict:
    """The neutral numbers out of a CP2K output: total energy (hartree), converged, version."""
    energy = None
    for m in re.finditer(r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[(?:a\.u\.|hartree)\]:?\s+(-?\d+\.\d+)", text):
        energy = float(m.group(1))
    converged = bool(re.search(r"SCF run converged in\s+\d+ steps", text)) and "SCF run NOT converged" not in text
    v = re.search(r"CP2K\| version string:\s+(.*)", text)
    return {"energy": energy, "converged": converged, "version": v.group(1).strip() if v else None}
