"""
qviz.backends.base -- the Backend interface every code implements (doc/RunSpecPlan.md §2-4).

Four verbs, and the order a Run calls them in:

    capabilities()            -> Capabilities   what this code can do (option 2.2: the spec filters back ends)
    translate(spec)           -> Job            PURE: the native input + the resolved settings table; nothing runs
    run(job, on_step)         -> Result         blocking (a Run calls it on a worker thread); streams SCFRow live (the code's own columns)
    (parse is inside run for an in-process code; an external code's run = launch + parse(run_dir))

A Result is NEUTRAL: the frontends read energies, the SCF trace, levels and lazily
sampled fields from it without knowing which code produced it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

from ..capability import Capabilities
from ..data import ScalarField, SCFRow, Structure, VectorField
from ..spec import RunSpec

StepSink = Callable[[SCFRow], None]


@dataclass(frozen=True)
class Job:
    """What `translate` produced: enough to run (and to SHOW the user before running)."""
    backend: str
    spec: RunSpec
    native: dict                          # the code's own input, as data (SOLID²: the bridge's arguments)
    resolved: dict                        # the ε-derived / stated native settings table, shown and recorded
    files: dict[str, bytes] = field(default_factory=dict)   # an external code's input deck(s)
    command: tuple[str, ...] = ()         # an external code's argv WITHOUT the mpirun launcher (the runner adds it)
    live: dict = field(default_factory=dict)   # the code's run record SO FAR, while it runs (SOLID²: taken at iteration 1)


@dataclass(frozen=True)
class FieldRef:
    """A lazily sampled field: `sample(n, pad)` builds the grid only when a panel asks."""
    name: str
    signed: bool
    sample: Callable[[int, float], ScalarField | VectorField]


@dataclass
class Result:
    converged: bool
    energy_Ha: Optional[float]            # None for an unconverged run: a failed SCF hands back no plausible number
    structure: Structure
    scf: list[SCFRow] = field(default_factory=list)
    levels: list[dict] = field(default_factory=list)      # ascending: {"e","occ","spin" -1/0/+1,"label","n"}
    polarized: bool = False
    fields: dict[str, FieldRef] = field(default_factory=dict)   # "density", "spin_density", "gradient", "orbital:<i>"
    orbital: Optional[Callable[[int, int, float], ScalarField]] = None   # level index -> sampled MO
    resolved: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    failure: Optional[str] = None         # why the SCF did not converge, in the code's own words (SOLID²: its SCFFailure)
    record: dict = field(default_factory=dict)   # the code's own run record (SOLID²: report::GlobalReport -- grids, basis, scf)
    state_file: Optional[str] = None      # the saved SCF state a later run can CONTINUE from (qviz.states); None: none saved

    @property
    def num_occupied(self) -> int:
        return sum(1 for L in self.levels if L["occ"] > 0.5)


@runtime_checkable
class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...
    def translate(self, spec: RunSpec) -> Job: ...
    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result: ...


class BackendRegistry:
    """The back ends this app knows, by name; the Workspace asks it, never a concrete class (DIP)."""

    def __init__(self, backends: list[Backend] = ()):
        self._by_name: dict[str, Backend] = {}
        for b in backends:
            self.add(b)

    def add(self, b: Backend) -> None:
        self._by_name[b.name] = b

    def get(self, name: str) -> Backend:
        if name not in self._by_name:
            raise KeyError(f"no back end {name!r}; known: {sorted(self._by_name)}")
        return self._by_name[name]

    def names(self) -> list[str]:
        return list(self._by_name)

    def verdicts(self, spec: RunSpec) -> dict:
        """The back-end strip for one spec (option 2.2)."""
        return {n: b.capabilities().can_run(spec) for n, b in self._by_name.items()}

    def strip(self, spec: RunSpec, ticked=()) -> dict[str, tuple[bool, bool, str]]:
        """The back-end strip's state for `spec`: {name: (enabled, ticked, hover text)}.  Enabled = can run the spec as
        stated (option 2.2); a tick survives while its back end stays able; with none left, the first able REAL back
        end is ticked (a synthetic one only when no real one can)."""
        v = self.verdicts(spec)
        keep = [n for n in ticked if v.get(n)]
        if not keep:
            able = [n for n in v if v[n]]
            real = [n for n in able if not self.get(n).capabilities().synthetic]
            keep = (real or able)[:1]
        return {n: (bool(v[n]), n in keep, "can run this spec" if v[n] else "cannot: " + "; ".join(v[n].reasons))
                for n in v}

    def default_for(self, spec: RunSpec) -> Optional[str]:
        """The first back end that can run `spec`, in registration order (SOLID² is registered first --
        PROVISIONAL, RunSpecPlan §10 Q6)."""
        return next((n for n, v in self.verdicts(spec).items() if v), None)


#: accuracy.representation keys that hold ONE code's stated settings ("if this runs on CP2K, use these").  A code reads its
#: own block and ignores the others', so one spec can carry tuning for every code it is compared across; what a code
#: must understand -- or refuse -- is the SHARED part (a stated Gaussian basis, today).
CODE_BLOCKS = ("solid2", "cp2k", "qe", "abinit", "vasp")


def shared_representation(spec: RunSpec) -> dict:
    """representation minus the per-code blocks: the stated choices EVERY code that runs the spec must honour."""
    return {k: v for k, v in (spec.accuracy.representation or {}).items() if k not in CODE_BLOCKS}


def code_block(spec: RunSpec, code: str) -> dict:
    return (spec.accuracy.representation or {}).get(code) or {}


def data_structure(spec: RunSpec) -> Structure:
    """The frontends' Structure (Cartesian bohr) for a spec's atoms; a crystal's fractional sites go through its lattice."""
    import numpy as np
    st = spec.structure
    numbers = np.array([s.Z for s in st.sites], int)
    if st.kind == "crystal":
        L = np.asarray(st.lattice_bohr, float)
        pos = np.array([np.asarray(s.frac, float) @ L for s in st.sites])
        return Structure([s.element for s in st.sites], pos, numbers, lattice=L)
    pos = np.array([s.xyz for s in st.sites], float)
    return Structure([s.element for s in st.sites], pos, numbers)


def species_kinds(spec: RunSpec) -> list[dict]:
    """One species (CP2K &KIND, QE ATOMIC_SPECIES) per distinct (element, tag, moment): a tag names its kind (Mn1, Mn2); untagged sites of one element
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
