# RunSpec plan — a back-end-neutral seam, many back ends, one in-app agent

Status (2026-10-30): plan under execution.  **Built:** G1 bridge (qchem lane-g, landed on main), S1 spec +
capability filter + Workspace/Run/project on Backends, S2 SOLID² (molecules), S3 CP2K (crystals + molecules,
single points; its translator reproduces the banked Si Γ and MnO AFM-II +U oracle decks, and real runs give both
banked energies), S4 compare across codes (one spec on every ticked back end; CP2K is handed SOLID²'s own basis
file converted, `qviz/backends/gaussian_basis.py`; per-code `representation.<code>` blocks; the Codes panel;
F₂ runs on both).  **S4 findings, handed to qchem's tracker:** the molecular facade's default XC mesh is
0.52 Ha/atom off (the GUI now states a mesh from ε), and the molecular PP route sits 4 mHa/atom above CP2K on F₂
while the crystal route matches CP2K on NaF.  **SolidSetup BOUND (2026-10-31, §9):** a crystal is a qchem deck (`solid2_deck.py`) built by `deck::Build` in
process (`qchem_py.SolidRun`); the Si inline cell lands on `rundeck`'s energy to 1e-9 Ha.  **Next:** a crystal's density
over the cell, and its revision record in the run directory.  **S5 agent: the tool
layer, shared draft, loop, consent gate and Agent dock are built on molecules (`qviz/agent/`), against a scripted
client and an offline planner; no model client exists until the user answers Q4** (no API call, no key, nothing
sent).  The open questions for the user are in §10; the code takes the
recommended option for each and marks it PROVISIONAL.

## 0. Why this note exists

Two requirements arrived on 2026-10-30:

1. **An AI agent inside the app** that helps the user build the structure and choose the
   Hamiltonian, the basis set, U (including *computing* U) and the convergence criteria / methods.
2. **The app is a front end for several codes**: SOLID² (the qchem rename, pending), CP2K,
   Quantum ESPRESSO, ABINIT and VASP.  The user picked **option 2.2**: the user says *what* they
   want, and the back ends that can do it light up.  Option 2.1 (pick a code first, then the
   features toggle to match it) was rejected.

The ruling that follows: today's seam (`ComputeBackend` in `qviz/data.py` + `RunSpec` in
`qviz/workspace.py`) is shaped like one code's molecular SCF.  It turns into two things:

- a **`RunSpec`** that names no code: structure, physics, task, and **one accuracy target ε**
  (the same single knob the core ruled for its grids, `doc/QueueCampaignPlan.md` §7 in qchem:
  one user-facing target, every internal tolerance derived from it by a stated formula);
- a **`Backend`** interface with four verbs: `capabilities` / `translate` / `run` / `parse`.

The agent's product is a `RunSpec`, nothing else.  So the build order is
**spec → back ends (SOLID² + CP2K first) → agent**.

Design rules carried over from qchem's own `doc/InputDeck.md`, because they have already been
paid for there: a reader is **strict** (an unknown key throws, naming the legal keys); **the
file you wrote is never the record of what ran** (a resolved record is written before the run);
**units: eV in files where a person quotes them, atomic units in memory**.

---

## 1. The RunSpec schema

A `RunSpec` is a frozen, hashable, JSON-serializable value.  Its four blocks answer four
questions; nothing in it names a code.  A field the user leaves unset stays `None` and means
"the back end's resolved default", which is *recorded* after resolution (§4.3), never silently
assumed.

```jsonc
{
  "schema": "solid2-runspec/1",
  "label": "MnO AFM-II +U 4 eV",

  // ---- 1. STRUCTURE: what atoms, where ---------------------------------------------------
  "structure": {
    "kind": "crystal",                       // "molecule" | "crystal" (slab/wire later: a crystal with vacuum + periodicity flags)
    "lattice_bohr": [[8.4,4.2,4.2],[4.2,8.4,4.2],[4.2,4.2,8.4]],   // absent for a molecule
    "sites": [                               // one per atom; fractional for a crystal, Cartesian bohr for a molecule
      {"element":"Mn", "frac":[0,0,0],         "tag":"Mn1"},
      {"element":"Mn", "frac":[0.5,0.5,0.5],   "tag":"Mn2"},
      {"element":"O",  "frac":[0.25,0.25,0.25]},
      {"element":"O",  "frac":[0.75,0.75,0.75]}
    ],
    "source": {"materials": "MnO_AFM2"}      // provenance only: a materials.json key, a CIF path, "agent", "drawn"
  },

  // ---- 2. PHYSICS: which Hamiltonian, which state -----------------------------------------
  "physics": {
    "theory": "DFT",                         // "HF" | "DFT"   (MP2/RPA/GW later -- not a token today)
    "functional": "LDA",                     // "LDA" | "PBE" | "PBEsol" | "SCAN" | "HSE06" | ...  (libxc-style names; DFT only)
    "core": "pseudopotential",               // "all-electron" | "pseudopotential"  (WHAT, not which file -- see "representation")
    "spin": {
      "polarized": true,                     // the DEFAULT is true (user bias: Pol first, UnPol is the efficiency special case)
      "multiplicity": null,                  // molecules: 2S+1; crystals: net moment via "site_moments"
      "site_moments": [5, -5, 0, 0]          // the stated magnetic order (one per site, μB, sign = sublattice); required for an AFM
    },
    "hubbard": [                             // DFT+U; U is an INPUT or a QUANTITY TO COMPUTE (task.compute_U)
      {"sites":["Mn1","Mn2"], "l":2, "U_eV":4.0, "J_eV":0.0, "form":"Dudarev", "projector":"ortho-atomic"}
    ],
    "occupations": {"smearing":"fermi-dirac", "kT_Ha":5e-3},   // or {"smearing":"none"} for an insulator / molecule
    "relativity": "scalar"                   // "none" | "scalar" | "spin-orbit"
  },

  // ---- 3. TASK: what to compute -----------------------------------------------------------
  "task": {
    "kind": "energy",                        // "energy" | "relax" | "cell-relax" | "bands" | "dos" | "compute_U" | "md" (later)
    "properties": ["density", "spin_density", "orbitals", "forces"],   // what parse() must return
    "compute_U": null,                       // e.g. {"method":"linear-response", "sites":["Mn1"], "l":2, "self_consistent":false}
    "relax": null,                           // e.g. {"fmax_eV_A":0.01, "stress_kbar":0.1}
    "kpath": null                            // bands: "auto" (Setyawan-Curtarolo) or explicit
  },

  // ---- 4. ACCURACY: ONE target, plus optional stated method choices ------------------------
  "accuracy": {
    "epsilon_Ha_per_atom": 1e-5,             // THE knob.  Every back end derives cutoffs, k-mesh, grids, SCF tolerance from it.
    "representation": null,                  // OPTIONAL stated choice: {"basis":"GTH-DZVP-MOLOPT"} or {"pw_cutoff_Ry":80} or {"pp":"GTH-LDA-q7"}
    "kmesh": null,                           // OPTIONAL stated: [2,2,2] -- otherwise derived from ε and the cell
    "scf": null                              // OPTIONAL stated method: {"mixer":"kerker+pulay", "depth":8, "accelerator":"GDM", "max_iter":200}
  }
}
```

### 1.1 Why ONE ε, and what "stated" means

The user's ruling for the core, verbatim in spirit: one user-facing accuracy target, consistent
error across terms — there is no point making one term 1e-10 and another 1e-4.  The GUI extends
the same principle across codes: **ε is the only accuracy knob a novice ever sees**.  Each back
end owns a `derive(spec) → native settings` table that maps ε to its own cutoffs, grids, k-mesh
and SCF tolerance, and prints that table (the GUI shows it as the "resolved settings" panel).

The `representation`, `kmesh` and `scf` blocks are **stated overrides**, for the expert and for
reproducing a banked oracle run byte-for-byte (e.g. the CP2K decks in qchem's
`IntegrationTests/CP2K/`, which pin `CUTOFF 144`, `REL_CUTOFF 60`, `EPS_DEFAULT 1e-14`).  A
stated value outranks the ε-derived one and is recorded as stated, the same way the core's deck
records a stated CP2K-deviation route.  A stated `representation` is also the one place where a
spec can become code-specific (a GTH basis name only CP2K and SOLID² read) — and the capability
model (§2) handles that honestly: such a spec simply enables fewer back ends.

### 1.2 What ε means per term, precisely

`epsilon_Ha_per_atom` is the target error in the total energy per atom, against the complete
basis / dense-k / tight-SCF limit of *that code's* Hamiltonian.  It does **not** promise
agreement between codes (their pseudopotentials differ by more than 1e-5 Ha; that is what the
compare view is for).  Each back end splits ε across its error sources (basis, k, grid, SCF) by a
stated rule — first version: ε/4 each, with the SCF tolerance one order tighter because it is
nearly free.  The ε→setting tables start as rules of thumb (§3) and are replaced by measured
convergence curves as we bank them (an "ε calibration" record per back end, same discipline as
qchem's `doc/Benchmark.md`).

### 1.3 The structure is inline

The core's deck takes a structure **name** (a `materials.json` key, `doc/InputDeck.md` rule 2).
The GUI cannot live with that: the agent and the editor make structures that are in no registry.
So the spec carries the atoms inline and keeps the name only as provenance.  Consequence for the
SOLID² translator, flagged as a lib ask in §10 Q3.

### 1.4 Validation

`RunSpec.validate()` is strict and back-end-free: unknown keys throw; parity of electrons vs
multiplicity; `site_moments` length = number of sites; an AFM (moments summing to zero with
nonzero entries) must state its moments (mirrors the core's `RequireStatedMagneticOrdering`);
`hubbard.sites` must name existing tags; `task.compute_U` requires a `hubbard`-able site.  A
spec that validates may still be one no back end can run — that is §2's job, not an error.

---

## 2. The capability model and how option 2.2 filters back ends

### 2.1 Capabilities are DATA, declared per back end

Each back end publishes a `Capabilities` value — a set of **features**, each with an optional
**constraint** — that `can_run(spec)` checks.  It is data (a JSON file per back end, versioned
with the adapter), so the agent can read it and the user can see why a back end is greyed out.

```python
@dataclass(frozen=True)
class Capability:
    feature: str                 # dotted key into the spec: "physics.functional=HSE06", "task.kind=compute_U", ...
    supported: bool
    note: str = ""               # shown on hover: "linear response via hp.x", "Dudarev only", "Γ-only"
    constraint: Callable | None = None   # e.g. "hubbard.form == Dudarev"; "structure.kind == crystal"

class Backend(Protocol):
    name: str                                  # "SOLID²", "CP2K", "QE", "ABINIT", "VASP"
    def capabilities(self) -> Capabilities: ...
    def can_run(self, spec: RunSpec) -> Verdict: ...   # Verdict = ok | refused(list[Reason])
```

`can_run` walks the spec and collects **every** reason it refuses (not the first), so the UI can
say "CP2K: no orbital-resolved U; no SCAN in this build".

### 2.2 The first matrix (to be verified against each code's own docs + our builds)

| feature | SOLID² | CP2K | QE | ABINIT | VASP |
|---|---|---|---|---|---|
| molecule (isolated) | ✅ | ✅ | box only | box only | box only |
| crystal | ✅ GPW / PW | ✅ GPW | ✅ | ✅ | ✅ |
| all-electron | ✅ molecules | GAPW | ✗ (PAW ≈) | PAW ≈ | PAW ≈ |
| LDA / PBE | ✅ LDA; PBE pending | ✅ | ✅ | ✅ | ✅ |
| hybrids (HSE06) | ✗ | ✅ | ✅ | ✅ | ✅ |
| spin-polarized, stated AFM | ✅ | ✅ (BS kinds) | ✅ (species split) | ✅ (spinat) | ✅ (MAGMOM) |
| DFT+U Dudarev, given U | ✅ | ✅ (shell-averaged, LOWDIN) | ✅ | ✅ | ✅ |
| orbital-resolved U | ✅ | ✗ | ✅ (orbital_resolved) | ✗ | ✗ |
| **compute U: linear response** | ✅ (`hubbardLinearResponse`) | ✗ | ✅ (hp.x) | ✅ (lruj) | manual (LDAUTYPE scan) |
| **compute U: ACBN0** | ✅ (`estimateHubbardU`, loop) | ✗ | ✗ (separate tools) | ✗ | ✗ |
| relax / cell-relax | 🟡 (PES `ConvergeAt`; forces seam pending) | ✅ | ✅ | ✅ | ✅ |
| bands / DOS | 🟡 | ✅ | ✅ | ✅ | ✅ |
| live per-iteration stream | ✅ (in-process observer) | tail stdout | tail stdout | tail stdout | tail OSZICAR |
| density / orbital on a grid | ✅ in-process | cube file | pp.x cube | cut3d | CHGCAR / PARCHG |

The table is a starting hypothesis; each row becomes a capability entry only when an adapter test
has actually run it (rule: **a capability is earned by a parse-tested run**, not by a manual page).

### 2.3 How 2.2 shows up in the UI

- The user (or the agent) edits **one spec**.  Beside it sits a back-end strip: one chip per code,
  green (can run), grey (cannot — hover lists the reasons), or absent (not installed / not
  licensed on this box; detected at start-up, §4.1).
- Nothing in the spec editor ever disappears because of a back end.  An option no code supports
  is still shown, marked "no installed back end can do this".  This is the 2.2 rule: the **spec**
  is primary, back ends are filtered *by* it, never the reverse.
- When several back ends are green the default is SOLID²; the user may tick more than one, which
  makes N runs of the **same spec** — the existing split-view compare (Pillar 2) becomes the
  cross-code compare for free.
- A *stated* representation narrows the strip (a GTH basis name → only SOLID² and CP2K); the UI
  says so on the field ("stating this disables QE, ABINIT, VASP").  What narrows is the SHARED part;
  a `representation.<code>` block (`cp2k`, `solid2`, ...) is that code's own tuning and the other
  codes ignore it, so one spec carries every leg's settings (S4, 2026-10-30).

---

## 3. Translation per back end

`translate(spec) → Job`: a pure function, no side effects, no run.  A `Job` is a directory plan:
`{files: {name: bytes}, command: argv, env: dict, resolved: dict}` where `resolved` is the
ε-derived settings table the GUI shows and the record keeps.  Pure so it is unit-testable by
golden files (§8 stage 2) and previewable before anything runs ("show me the input").

### 3.1 SOLID² (in-process first, deck second)

Two routes, one translator:

- **In-process** (the GUI default): the spec maps onto `MoleculeSetup::Resolve(structure,
  CalcOptions, AcceleratorOptions)` for a molecule, and `SolidSetup::Resolve(lattice,
  GaussianBasisRecipe, SolidCalcOptions, SolidAcceleratorOptions)` for a crystal, through the
  nanobind bridge (lane G step 1).  `Calculation::OnIteration` already exists, so live streaming
  needs no new facade hook.  `MoleculeSession` / `SolidSession` give relax via
  `PotentialEnergySurface::ConvergeAt(cell, R)`.
- **Batch**: the same translation emits a `rundeck` JSON deck (qchem's `doc/InputDeck.md` §3
  schema — `structure`, `kmesh`, `basis`, `solid`, `scf`, `postSCF`).  `task.compute_U` maps to
  `postSCF: [{"hubbardLinearResponse": {...}}]` or `[{"estimateHubbardU": {}}, {"hubbardLoop":
  {...}}]`.  The deck writes its own `rNNN.json` record, which `parse` reads directly.

Field map (molecule): `physics.theory/functional` → `CalcOptions.model` (HF | LDA | Xalpha);
`physics.core = pseudopotential` → `CalcOptions.pp = "gth-lda-valence"` (+ `ppOverride` from a
stated `representation.pp`); `spin.multiplicity` → `multiplicity`; `accuracy.scf.accelerator` →
`AcceleratorOptions.type`.  ε → `basis` choice (sto-3g / dzvp / larger), mesh `nRadial` /
`angularDegree`, SCF `minFD`.  Crystal: ε → the ONE accuracy target the core is about to grow
(`SolidCalcOptions.tolerances`, lane A chunk C2); until C2 lands, the translator fills
`screenEps`/`densityEps`/`vlocEps`/`cutoffFactor` from a table and marks them "derived (pre-C2)".

### 3.2 CP2K (first external back end)

Real shape: qchem's `IntegrationTests/CP2K/mno_afm2_gpw_va_plusU.inp` (banked oracle, -61.303 Ha
without U).  The translator emits one `&GLOBAL` + `&FORCE_EVAL/&DFT/&SUBSYS` deck:

| spec | CP2K |
|---|---|
| `task.kind` energy / relax / cell-relax | `RUN_TYPE ENERGY / GEO_OPT / CELL_OPT` (+ `&MOTION`) |
| `physics.spin.polarized` | `UKS`; `MULTIPLICITY` from the moments |
| `site_moments` (AFM) | one `&KIND` per sublattice (`Mn1`, `Mn2`), each with `&BS` α/β occupations |
| `hubbard` | `PLUS_U_METHOD LOWDIN`; `&DFT_PLUS_U L 2  U_MINUS_J [eV] 4.0` per `&KIND` (shell-averaged only → `can_run` refuses orbital-resolved U) |
| `functional` | `&XC_FUNCTIONAL` `&LDA_X`+`&LDA_C_VWN` / `PBE` / … |
| `occupations.smearing` | `&SMEAR ON  METHOD FERMI_DIRAC  ELECTRONIC_TEMPERATURE [au_temp] kT`, `ADDED_MOS` |
| ε (grid) | `&MGRID CUTOFF / REL_CUTOFF / NGRIDS` |
| ε (screening) | `EPS_DEFAULT`, **and `EPS_PGF_ORB` stated tight** (CP2K's default once produced a 3.5 Ha "oracle" that was retracted — qchem `doc/OldPlans/GPWPlan.md`; the translator never leaves it at default) |
| ε (SCF) | `EPS_SCF`, `MAX_SCF`; mixer from `accuracy.scf` (`BROYDEN_MIXING ALPHA/BETA/NBUFFER`) |
| `representation.basis/pp` | `BASIS_SET_FILE_NAME`, `POTENTIAL_FILE_NAME`, per-kind `BASIS_SET` / `POTENTIAL` (`GTH-PADE-q7`) |
| k-mesh | `&KPOINTS SCHEME MONKHORST-PACK` (Γ-only omits it) |
| structure | `&CELL A/B/C [bohr]`, `&COORD SCALED` (crystal) or Cartesian (molecule in a box, `PERIODIC NONE` + `&POISSON PSOLVER MT`) |

**No `IGNORE_CONVERGENCE_FAILURE`** in a translated deck (the banked oracle decks have it for
historical reasons): a capped SCF must surface as a failed run, the same rule the core enforces
on its tests (`qchem.SCFCap`).

### 3.3 Quantum ESPRESSO

`pw.x` namelists `&CONTROL / &SYSTEM / &ELECTRONS` + `ATOMIC_SPECIES / CELL_PARAMETERS /
ATOMIC_POSITIONS crystal / K_POINTS automatic`.  AFM: split the species (`Mn1`, `Mn2`) with
`starting_magnetization(i) = ±`.  DFT+U: the QE ≥ 7 `HUBBARD {ortho-atomic}` card.  compute_U:
a two-step job, `pw.x` then `hp.x` (`&INPUTHP nq1..3`).  ε → `ecutwfc`, `ecutrho`, `conv_thr`,
k-mesh, **and the FFT grid `nr1..3` stated** — pw.x evaluates E_xc on its minimal FFT grid, and on
β-MnO₂ at 40 Ry the default grid is 107 μHa off the grid-converged value (qchem CLAUDE.md,
2026-10-26), so a μHa-level ε must state it.

**Built (S7, `qviz/backends/qe.py`).**  Pseudopotentials are qchem's own GTH set as `gth2upf` UPFs
(`<El>.pz-gth-q<q>.UPF`, q from gth-lda-valence); a missing file is a refusal naming the gth2upf
command.  LDA is `sla+vwn`.  The grid is always stated: pw.x's own minimal grid
(nr_i = int(√ecutrho·|a_i|/π)+1, FFT-friendly -- it reproduces the banked run's 96×96×60) times a
`grid_factor`, 1.0 for ε ≥ 1e-4 and 1.5 below (PROVISIONAL).  **Measured 2026-10-31** on the gate deck
(β-MnO₂ Γ, 300/1200 Ry, `~/Code/qe-runs/MnO2/gridscan_*`): E(Ry) at 96×96×60 / 120×120×75 /
144×144×90 / 192×192×120 = −186.93546944 / −186.93546189 / −186.93546514 / −186.93546375 -- the
default is 2.8 μHa (0.5 μHa/atom) off and every finer grid agrees to < 1 μHa.  So at a hard-PP
cutoff the minimal grid is nearly enough; the 107 μHa of CLAUDE.md is a 40 Ry effect.  The 1.5 factor
is the margin until a low-cutoff point is measured.  Gate: the translator reproduces the banked oracle
deck `IntegrationTests/QE/mno2b.scf.in`, and a real run at the banked grid reproduces −186.93546944 Ry.

### 3.4 ABINIT

One `.abi` file: `acell`/`rprim`, `ntypat`/`typat`/`znucl`, `xred`, `ecut` (+`pawecutdg`),
`ngkpt`/`shiftk`, `nsppol 2` + `spinat`, DFT+U `usepawu 1 lpawu dmatpuopt upawu jpawu`, compute_U
via the `lruj` post-processing utility (two perturbed runs).  Pseudos via `pseudos` + `pp_dirpath`.

**Built (S7, `qviz/backends/abinit.py`).**  Pseudopotentials: each element's gth-lda-valence entry of qchem's
GTH database (`BasisSetData/GTH_POTENTIALS.txt`) re-wrapped as ABINIT's native **pspcod 10** (HGH, full h
matrix; `qviz/backends/gth.py`) and written beside the input -- one pseudopotential in a second format,
as CP2K gets SOLID²'s basis.  Not the gth2upf UPFs QE reads: abinit 10.8 aborts on them twice over (its
UPF→ixc map does not know gth2upf's label `SLA PZ NOGX NOGC`, and its UPF2 reader requires a linear
radial mesh from r = 0, while gth2upf writes QE's logarithmic one; measured 2026-10-31).  LDA = `ixc -1007`
(libxc Dirac + VWN).  `ngfft` always stated, by the QE rule.  +U refused (ABINIT's DFT+U is PAW-only).
Gate: β-MnO₂ Γ at the QE gate's cutoff and grid, against the banked QE energy -- **passed 2026-10-31:
ABINIT −93.467734513 Ha vs QE −93.46773472 Ha, +0.2 μHa** (38 SCF steps, toldfe 1e-10; QE's U = 1e-8 eV
left out, energetically nil): two codes agreeing on one pseudopotential in two formats.  Parser note: a
toldfe stop prints `etot is converged` with no `=>converged.` arrow (m_common.F90 scprqt), so convergence
is read from either `At SCF step` line.

### 3.5 VASP — licensed

`INCAR` (`ENCUT`, `EDIFF`, `ISPIN 2`, `MAGMOM`, `LDAU LDAUTYPE 2 LDAUL LDAUU LDAUJ`, `ISMEAR`
`SIGMA`), `POSCAR`, `KPOINTS`; `POTCAR` is **concatenated locally from the user's licensed PAW
library and never written into a project file, a shared record, a commit or an agent transcript**.

**VASP licence rules, enforced in code, not in prose:**
- A VASP job directory lives only under `~/Code/vasp-runs/…` (the D-RUNDATA home) and is marked
  `licensed=true` in its record; the project file stores a **reference** (path + checksum), never
  the bytes of an input or output.
- Export / share / "publish compare" refuses any run whose back end is VASP, naming the reason.
- The agent may *read* a VASP result to answer the user (energy, moments) but its tools never
  return raw VASP file contents (POTCAR, OUTCAR) into the model context — only parsed numbers.
- Golden-file tests for the VASP translator compare structure-only (INCAR keys, POSCAR geometry)
  and the fixtures are generated at test time, not checked in.

---

## 4. Run and parse — local first

### 4.1 Discovery

At start-up each adapter probes for its executable **without running it bare**.  Paths default to
the box's built oracles (qchem CLAUDE.md, verified 2026-09-09) and are overridable in settings:

| back end | binary | how it is launched |
|---|---|---|
| SOLID² | the `qchem_py` extension (in-process) / `rundeck` (batch) | in-process / plain exe |
| CP2K | `/usr/bin/cp2k.psmp` | `mpirun -np N cp2k.psmp -i job.inp -o job.out` |
| QE | `~/Code/q-e/PW/src/pw.x` (+ `hp.x`) | `mpirun -np N pw.x -in job.in` |
| ABINIT | `~/Code/abinit/build/src/98_main/abinit` | `mpirun -np N abinit job.abi` (no `--force-mpirun`: see below) |
| VASP | `~/Code/vasp.6.3.0/vasp.6.3.0/bin/vasp_std` | `mpirun -np N vasp_std` (in the job dir) |

⛔ **All four oracles are MPI builds: the runner ALWAYS launches through `mpirun`, even for one
rank.  A bare binary hangs on this box (the OpenMPI
singleton-init hang).**  This is enforced structurally: the `Job.command` an external translator
returns is `[...]` *without* the launcher, and the one `MpiRunner` prepends
`mpirun -np N`.  There is no code path that executes an oracle binary
directly, and discovery checks presence (`os.access`), never `--version`.  **Measured 2026-10-31:
`--force-mpirun` is the ABINIT TEST-SUITE driver's flag (`runtests.py`); `abinit` 10.8 itself rejects it
("Unsupported option") and exits without running, so the ABINIT argv carries no flag** (qchem CLAUDE.md's
"ABINIT additionally needs `--force-mpirun`" conflates the two -- flagged to the controller).

### 4.2 The runner

`Runner.start(job) → RunHandle` creates the run directory with qchem's convention
(`~/Code/<app>-runs/<Material>/<label>.rNNN/`, via `scripts/rundir` semantics), writes the files,
writes the **resolved record first** (spec + resolved table + command + code version), then starts
the process with `subprocess.Popen` on a worker thread.  `RunHandle` exposes `status`, `cancel()`
(SIGTERM to the mpirun process group), and a **progress stream**: an adapter-provided incremental
parser tails stdout / OSZICAR and emits the same `SCFStep(iteration, energy, dE, residual, drho)`
the live trace already plots.  So the existing streaming SCF panel works for every back end, at
the granularity each code prints.  Thread count: `QCHEM_OPENMP_THREADS` / `OMP_NUM_THREADS`
set explicitly in `job.env` (a batch run's default is serial otherwise).

Local first means: one machine, a bounded queue (default 1 concurrent heavy run, respecting the
box's memory discipline — the runner wraps external runs in `scripts/memsafe` when available).
Remote (ssh / a queue system) is the same `Runner` interface later; nothing in the spec or the
adapters changes.

### 4.3 Parse → one neutral `Result`

`parse(run_dir) → Result`:

```python
@dataclass(frozen=True)
class Result:
    converged: bool                 # a capped SCF is converged=False, and the UI shows it red
    energy_Ha: float | None
    breakdown: dict[str, float]     # where the code prints it (Hartree, XC, +U, ...)
    scf: list[SCFStep]
    site_moments: list[float] | None
    hubbard_U_eV: dict | None       # task.compute_U results, per site/manifold
    forces: np.ndarray | None
    fields: dict[str, FieldRef]     # lazily loadable grids: density, spin density, orbitals (cube/CHGCAR/in-process)
    resolved: dict                  # the ε-derived native settings that actually ran
    provenance: dict                # code, version, command, input checksums, host, wall time
```

`FieldRef` is lazy (a path + reader, or an in-process sampler), so the existing `Run`'s
lazy-memoized field cache keeps its shape: SOLID² samples in-process; CP2K reads cube files
(requested by `&PRINT &E_DENSITY_CUBE` in the translated deck when `properties` asks for it).

Parsers are tested against **real outputs of runs on this box**, stored as fixtures under
`tests/fixtures/<code>/` — except VASP (§3.5).

---

## 5. The in-app agent

### 5.1 Where it sits

```
 user ──chat──▶ Agent ──tools──▶ SpecDraft (the one editable spec)  ──▶ Backend strip (can_run)
                  │                    ▲                                   │
                  │                    └── user edits the same spec ───────┤
                  └── read-only tools ──▶ results, capabilities, structure DB, docs
                                                         Run button (USER) ─┘
```

The agent is a panel beside the spec editor.  It and the user edit **the same `SpecDraft`**; every
agent change is a diff the user sees highlighted and can revert (an undo stack shared by both).
The agent never owns hidden state that is not in the spec or in the chat.

Implementation: Claude via the Anthropic API with tool use (model id configurable, default the
current Opus), run from the GUI process.  API key from the environment / the OS keyring, never in
a project file.  The agent works without a network back end being configured — it only plans.

### 5.2 Its tools

Read / reason (no side effects):
- `get_spec()`, `validate_spec(spec)`, `list_backends()`, `can_run(spec)` per back end;
- `explain_resolved(spec, backend)` — the ε → native table, so it can say *why* a cutoff is 80 Ry;
- `lookup_structure(query)` — materials.json, molecules.json, the bundled examples, a CIF file
  the user opened (later: COD / Materials Project, with the user's consent per query);
- `analyze_structure(spec)` — space group, Wyckoff sites, inequivalent magnetic sublattices,
  suggested AFM orderings (spglib), coordination, nearest distances;
- `get_result(run_id)` / `compare_results(ids)` — parsed `Result`s only (never raw licensed files);
- `search_docs(query)` — the app's own help + curated notes (e.g. "MnO AFM-II converges free with
  Kerker+Pulay, recipe `decks/MnO_AFM2_free.json`"), so advice cites banked experience.

Write (only the draft):
- `propose_spec_patch(patch, rationale)` — a JSON-merge patch applied to the draft and shown as a
  diff with the rationale; the user accepts, edits or reverts;
- `build_structure(...)` — generate a structure into the draft: a molecule from a name / SMILES,
  a crystal from a prototype (rocksalt, perovskite, …) + lattice constant, a supercell, an AFM
  decoration, a slab.

Run (gated, see 5.3):
- `request_run(spec, backends, purpose)` — puts a **run request card** in the chat ("compute U for
  Mn by linear response on SOLID²: 2 SCFs, est. 6 min"); only the user's click starts it.

### 5.3 What it may and may not do

May:
- draft and revise the spec, explain every choice, and point out when a choice disables a back
  end (2.2 made visible);
- propose a **U workflow**: a literature/default U as a starting value, *or* a `task.compute_U`
  spec (linear response / ACBN0) the user then runs; and after the run, propose patching the
  computed U into the production spec;
- propose convergence methods from banked experience (e.g. GDM for a restart; Kerker+Pulay for a
  free TMO run) and an ε appropriate to the question asked;
- read results and diagnose a failed SCF, proposing a revised spec.

May not:
- **start, cancel or delete a run** — those are the user's clicks, always (runs cost CPU-hours
  and the box has memory limits);
- write files outside the draft, edit the project file, or touch the qchem / oracle trees;
- invoke any binary directly (it has no shell tool at all; the runner is the only launcher, and
  it always goes through `mpirun`);
- see or transmit licensed VASP inputs/outputs, the user's API key, or anything not in its
  tools' outputs;
- query an external database or web service without the user's per-query consent;
- state a number it did not get from a tool (a computed U, an energy) — its prose must cite the
  run id.

### 5.4 Why the agent comes last

Its only product is a spec, and its value is bounded by what `can_run` / `explain_resolved` /
`get_result` can tell it.  With the spec and two back ends in place, the agent is a thin layer
of tools over existing functions; built first, it would be guessing at an interface that moves.

---

## 6. How today's `ComputeBackend` migrates

No shims (the user's rule: replace, don't keep compatibility layers).

| today | becomes |
|---|---|
| `qviz.workspace.RunSpec(label, numbers, positions, method, basis, symmetry, multiplicity)` | `qviz.spec.RunSpec` (§1).  `method`→`physics.theory/functional`; `basis`→`accuracy.representation.basis` (stated); `symmetry`→`accuracy.scf.impose_symmetry` (a method choice, SOLID²-only capability); `multiplicity`→`physics.spin.multiplicity`.  Old `.qproj.h5` files: a one-off converter script, then the old reader is deleted. |
| `ComputeBackend` protocol (structure / density / orbital / gradient / run_scf) | split in two: `Backend` (capabilities / translate / run / parse — the *how to compute*) and `Result` + `FieldRef` (the *what came out*).  The frontends keep consuming `Structure` / `ScalarField` / `VectorField` / `SCFStep` from `qviz.data` unchanged. |
| `QChemBackend` (ctor runs the SCF) | `Solid2Backend`: `translate` builds the bridge call arguments; `run` calls the bridge on the worker thread with the `OnIteration` observer feeding the progress stream; `parse` wraps the live calculation handle as a `Result` whose `FieldRef`s sample in-process. |
| `AnalyticBackend` | `FakeBackend` for tests and demos: declares a small capability set, `run` emits the synthetic SCF trace.  Kept because it makes the UI testable without a build. |
| `BackendFactory(spec, on_iteration)` injected into `Workspace` | `Workspace` holds a `BackendRegistry`; `add_run(spec, backend_name)`.  `Run` keeps its lifecycle and observer shape; `Run.compute()` becomes `Runner.start(backend.translate(spec))`. |
| `Run.stream_scf` re-runs the SCF to replay | gone: the trace is recorded during the one run (`Result.scf`); replay is a UI animation over the record. |
| `project.save_workspace` stores spec attrs | stores the `RunSpec` JSON + a pointer to each run directory's record (results are not duplicated into the project; VASP is reference-only). |

---

## 7. Module layout (qchem-GUI)

```
qviz/spec.py          RunSpec + validate + JSON (no back-end imports)
qviz/capability.py    Capability, Capabilities, Verdict
qviz/backends/        base.py (Backend protocol, Job, Result, FieldRef)
                      solid2.py, cp2k.py, qe.py, abinit.py, vasp.py, fake.py
                      <code>/capabilities.json
qviz/runner.py        Runner, MpiRunner, RunHandle, progress tailers
qviz/accuracy.py      the per-back-end ε → native tables (one function per back end, printed)
qviz/agent/           tools.py (the §5.2 tool set; RunRequest cards), draft.py (SpecDraft: one undo stack, both
                      authors), client.py (ModelClient interface, ScriptedClient, OfflinePlanner -- the real
                      client lands after Q4), consent.py (ConsentGate: every model call passes it; Q4_UNANSWERED
                      refuses any off-box call; licensed results never leave), session.py (the loop), panel.py (dock)
```

`qviz/spec.py` imports nothing from `backends/`; `backends/` imports `spec`; the agent imports
both but nothing imports the agent.  (Same no-cycles discipline as the core's module hierarchy.)

---

## 8. Staged build order

Each stage ends in something runnable and tested; nothing merges ahead of its stage's tests.

| stage | content | done when |
|---|---|---|
| **G1 bridge** (lane G step 1, already queued) | fresh nanobind bridge on `lane-g`: `MoleculeSetup::Resolve` + `Calculation` (later `MoleculeSession`), opaque handle, module-unit / C-header split; `QChemBackend` runs on it | water HF/dzvp in the app, live trace via `OnIteration` |
| **S1 spec** | `qviz/spec.py` + validation + JSON round-trip; `FakeBackend`; the Workspace/Run/project migration of §6; spec editor panel (structure / physics / task / ε) with the back-end strip | the app runs a `FakeBackend` from a spec; old projects converted |
| **S2 SOLID² back end** | `Solid2Backend` over the G1 bridge (molecules), then crystals via `SolidSetup` once bound; the deck route for batch | water/benzene molecules and Si / MnO crystal from the editor, live trace |
| **S3 CP2K back end** | translator (golden-file tests against `IntegrationTests/CP2K/` shapes), `MpiRunner`, stdout tailer, parser, cube fields | the MnO AFM-II +U spec reproduces the banked CP2K oracle energy when its representation is stated to match the oracle deck |
| **S4 compare across codes** | one spec → SOLID² + CP2K runs side by side in split view; the resolved-settings panel | the user can see the two codes' energies, moments and densities for one spec |
| **S5 agent v1** | read tools + `propose_spec_patch` + `build_structure` + `request_run` cards | the agent builds MnO AFM-II from "MnO, antiferromagnetic, with U", the user runs it |
| **S6 compute U** | `task.compute_U` on SOLID² (linear response, ACBN0) wired end to end; the agent's U workflow | the agent proposes, the user runs, the computed U patches into the spec |
| **S7 QE, ABINIT** | translators + parsers + capability files | each earns its matrix rows by a parse-tested run |
| **S8 VASP** | translator + parser under the §3.5 licence rules | a VASP run is visible in the app and refused by export |
| later | relax / bands / DOS tasks; remote runners; external structure DBs | — |

Per the qchem repo's tracker convention, residual items from these stages go as rows to qchem's
`doc/OpenWork.md` (lane G section), not into "suggested task" boxes.

---

## 9. Lib-side dependencies (flagged, not done here)

- **The SolidSetup bind (design, 2026-10-30, against lane-b 2acddd04's observer-in-ctor API).**
  Two routes:
  (a) the bridge builds `Lattice_3D` + `GaussianBasisRecipe` + `SolidCalcOptions` from a C request,
  calls `SolidSetup::Resolve` and constructs `SolidCalculation(setup, params, observer)`.  This works
  with no lib change, but it duplicates in pybind/ what `qchem::deck` already knows (species -> Nelec,
  decoration -> siteSpins, eV -> Ha for U, stages, the basis registry, defaults), and the two
  translators drift.
  (b) **APPROVED (controller, 2026-10-31; provisional until the user answers Q3): the deck is the seam.**  The GUI's SOLID² translator writes a deck JSON (the way
  the CP2K translator writes a `.inp`), and the bridge hands it to the library's own reader, so there
  is one translator and every GUI run is a reproducible `rundeck` deck.  It needs two lib pieces
  (lane B):
  1. an **inline `structure`** beside the named one (Q3).  The shape the GUI translator writes today
     (PROVISIONAL, for lane B to accept or amend): `{"lattice": {"vectors_bohr": [[a],[b],[c]]},
     "atoms": [{"el": "Mn", "frac": [0,0,0]}, ...]}` -- a materials.json entry with explicit lattice
     vectors in place of `type`/`a`/`T`; the pp set and the decoration stay where the deck already
     says them (`solid.pp`, `solid.siteSpins`).
  2. **`deck::Build(const json& deck, Observer) -> unique_ptr<SolidCalculation>`**, i.e. `deck::Run` split
     (the library never sees the GUI's RunSpec: RunSpec -> deck JSON is the GUI's translator)
     at "built, not yet converged", so the bridge converges with the GIL released and streams.
     The revision record is written as `Run` writes it, into the GUI's
     `~/Code/qchem6-runs/<Material>/`.

  What the bridge then exposes: energy, converged, iterations, the `RunDiagnostics` summary, site
  moments, and the density from `LastIterateDensity()` sampled over one cell (an orthogonal box
  around the cell's bounding box for the GUI's axis-aligned ScalarField, until the viewer learns
  non-orthogonal grids).  It does NOT expose levels (k-resolved bands are a later panel).
  Molecules keep `MoleculeSetup` (`deck::Resolve` refuses molecule names today).

  **The translator is BUILT (2026-10-31): `qviz/backends/solid2_deck.py`.**  A crystal spec becomes a
  deck: a materials.json NAME when `structure.source.materials` names an entry whose geometry the spec
  matches (elements, fractional sites, cell volume), else the inline form above; k-mesh and basis
  stated (no ε rule banked for either); spin as `multiplicity` (|Σ moments| + 1) + `siteSpins` (the
  signs); +U per site with the projector mapped (lowdin -> siteShells); Fermi-Dirac kT; compute_U ->
  a postSCF action; `representation.solid2.deck` deep-merged last.  ε is NOT mapped (the deck's
  defaults stand, and the resolved rows say so).  Golden tests (`tests/test_solid2_deck.py`): specs
  stating the banked bench decks translate to exactly `decks/bench/Si_Gamma_parity.json` and
  `NaF_SR2_Gamma_parity.json`; those and an MnO AFM-II +U deck pass `rundeck --check`; the Si Γ deck
  runs end to end (−7.1150674 Ha, 7 μHa from CP2K) and its revision record holds every value the GUI
  stated.

  **BOUND (2026-10-31, qchem main a0f6ce1a: lane B's inline structure + `deck::Build(RunSpec, Observer)` +
  `ScheduleToRun`).**  `qchem_py.SolidRun(deck_json)`: `deck::FromJson` + `deck::Build` (no SCF; a refusal raises with the
  library's text: an unknown key, an unknown structure, a setup refusal), `converge(on_iteration)` runs
  `Converge(ScheduleToRun(spec))` with the GIL released and streams the periodic IterationRow cells (Etotal, [F,D], ΔE/E,
  Δρ, ...), `failure()` is the SCFFailure line, `total_energy()` is None unless converged, `resolved()` is the resolved
  deck (Nelec / species derived).  The observer is attached per `converge`, not at Build.  The inline shape the
  translator writes IS lane B's (`lattice.vectors_bohr` rows, `atoms` el + frac; `decks/Si_diamond_inline.json`).
  Gates (`tests/test_solid2_deck.py::TheBridgeRunsThem`): Si as a built cell (inline) converges through the bridge to
  rundeck's energy on the same deck within 1e-9 Ha (both −7.1150676382, 10 iterations); a 2-iteration cap returns no
  energy and the "iteration limit" reason; a misspelt key raises the deck reader's text; compute_U is refused (Build stops
  at the SCF, postSCF actions are `rundeck`'s).
  **Found on the way:** the lib's console tables (submodules/tabulate) set `std::locale::global(en_US.UTF-8)` and restore
  the C++ global, "C", around each border, so every SCF left the HOST process in LC_ALL=C (a later `subprocess.run(...,
  text=True)` decoded rundeck's UTF-8 as ASCII).  The bridge restores the host's locale on every entry that prints
  (`HostLocale`); the defect itself is reported to the lib.
  **Not yet exposed:** the density over the cell, site moments, `RunDiagnostics`; the revision record (Build writes
  none -- the GUI will write `resolved()` into `~/Code/qchem6-runs/<Material>/`).  Lane B's electron-count refusal
  (cfdd1e04) reaches the GUI through the same RuntimeError path once it is on main.
- **The ONE accuracy target for solids** is lane A chunk C2; until then the SOLID² ε table fills
  today's separate tolerances and says so.
- **Forces / stress** for relax: the `GeometryGradient` capability (lane B review); until it
  lands, SOLID² declares `relax` unsupported for crystals.
- **PBE and other functionals** in SOLID²: today the PP front door is LSDA; the capability file
  says exactly that.

---

## 10. Open questions for the user

1. **ε units and default.** Proposed: `epsilon_Ha_per_atom`, default 1e-5 (≈ 0.3 meV/atom).  The
   core ruled "Ha per matrix element" for its internal target — should the user-facing GUI knob
   be per atom (what a user quotes) with the core's per-element target derived from it, or the
   same unit end to end?
2. **Stated overrides in a novice view.** Should `representation` / `kmesh` / `scf` be hidden
   behind an "expert" disclosure, or always visible?
3. **Inline structures in the core deck** (§9): add an inline `structure` object to `rundeck`
   (a lib change, lane B's territory), or keep SOLID² batch runs on temporary materials entries?
4. **The agent's model and key.** Anthropic API from the GUI process with the user's key — OK?
   Any limit on what the agent may send off the box (structures are fine? results?).
5. **External structure databases** (COD, Materials Project): wanted in agent v1, or later?
6. **Default back end when several are green**: SOLID² always, or "the last one used"?
7. ~~ABINIT launch line~~ -- settled by measurement (§4.1): `mpirun -np N abinit job.abi`.
8. **Multiplicity vs site moments for crystals**: the spec states per-site moments and derives the
   net multiplicity; the core's `SolidCalcOptions.multiplicity` + decoration fit that — confirm
   this is the representation you want users to see.
9. **Name**: write "SOLID²" in the UI already, or keep "qchem" until the rename lands?
