"""
qviz.spec -- the back-end-NEUTRAL run spec (doc/RunSpecPlan.md §1).

A RunSpec says WHAT is to be computed -- structure, physics, task, and ONE accuracy
target ε -- and names no code.  Back ends (qviz.backends, stage S2+) are filtered BY a
spec (option 2.2), translate it, run it and parse the result; the in-app agent's only
product is a spec.  So this module imports nothing from any back end.

Rules carried over from qchem's doc/InputDeck.md, already paid for there:
  * the reader is STRICT: an unknown key raises, naming the path and the legal keys;
    a missing key keeps the default (so an old spec survives a new field);
  * a key beginning with "_" is a comment, ignored and not kept;
  * units: bohr / hartree in the spec (atomic units in memory); eV only where a person
    quotes it (U_eV, J_eV), as the key name says.
RULED 2026-11-01 (doc/RunSpecPlan.md §10):
  * Q1: the accuracy knob is `epsilon_Ha_per_atom`, default 1e-5.  NOTE the core has since
    named its own target ε per Hamiltonian MATRIX ELEMENT (qchem doc/GridEpsilonPlan.md
    §1, deck `solid.accuracy`); the SOLID² translator derives that from this one.
  * Q8: a crystal's magnetic state is stated per site (`site_moments`); its net
    multiplicity is derived, never stated beside it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from typing import Any, Optional

SCHEMA = "solid2-runspec/1"

# symbol <-> Z through Kr (the same table qviz.molecules uses)
_ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
             "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
             "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th "
             "Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()   # the whole table: what a basis or a pseudopotential covers is the library's question
Z_OF = {s: i + 1 for i, s in enumerate(_ELEMENTS)}

THEORIES     = ("HF", "DFT")
CORES        = ("all-electron", "pseudopotential")
SMEARINGS    = ("none", "fermi-dirac", "gaussian", "marzari-vanderbilt")
RELATIVITY   = ("none", "scalar", "spin-orbit")
TASKS        = ("energy", "relax", "cell-relax", "bands", "dos", "compute_U")
U_FORMS      = ("Dudarev", "Liechtenstein")
U_PROJECTORS = ("ortho-atomic", "atomic", "pseudo-atomic", "lowdin")
U_METHODS    = ("linear-response", "ACBN0")


class SpecError(ValueError):
    """A spec that is malformed or contradicts itself (never: one no back end can run --
    that is a capability verdict, not an error)."""


# --------------------------------------------------------------------------
# The four blocks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Site:
    element: str
    frac: Optional[tuple[float, float, float]] = None   # crystal: fractional coordinates
    xyz:  Optional[tuple[float, float, float]] = None   # molecule: Cartesian bohr
    tag:  Optional[str] = None                          # names the site for hubbard / BS kinds

    @property
    def Z(self) -> int:
        return Z_OF[self.element]


@dataclass(frozen=True)
class Structure:
    kind: str                                           # "molecule" | "crystal"
    sites: tuple[Site, ...]
    lattice_bohr: Optional[tuple[tuple[float, float, float], ...]] = None   # rows a, b, c
    source: Optional[dict] = None                       # provenance only: {"materials": ...}, {"cif": path}, {"agent": ...}

    def __hash__(self) -> int:                          # `source` is a dict: hash the rest
        return hash((self.kind, self.sites, self.lattice_bohr))


@dataclass(frozen=True)
class Spin:
    polarized: bool = True                              # Pol is the general case (CLAUDE.md bias); UnPol the special one
    multiplicity: Optional[int] = None                  # molecules: 2S+1; None = minimal
    site_moments: Optional[tuple[float, ...]] = None    # crystals: one per site, μB, sign = sublattice (PROVISIONAL Q8)


@dataclass(frozen=True)
class Hubbard:
    sites: tuple[str, ...]                              # site tags
    l: int
    U_eV: Optional[float] = None                        # None = to be COMPUTED (task.compute_U)
    J_eV: float = 0.0
    form: str = "Dudarev"
    projector: str = "ortho-atomic"


@dataclass(frozen=True)
class Occupations:
    smearing: str = "none"
    kT_Ha: Optional[float] = None


@dataclass(frozen=True)
class Physics:
    theory: str = "DFT"
    functional: Optional[str] = "LDA"                   # libxc-style name; None for HF
    core: str = "pseudopotential"
    spin: Spin = field(default_factory=Spin)
    hubbard: tuple[Hubbard, ...] = ()
    occupations: Occupations = field(default_factory=Occupations)
    relativity: str = "scalar"


@dataclass(frozen=True)
class ComputeU:
    method: str = "linear-response"
    sites: tuple[str, ...] = ()
    l: int = 2
    self_consistent: bool = False


@dataclass(frozen=True)
class Task:
    kind: str = "energy"
    properties: tuple[str, ...] = ("density",)
    compute_U: Optional[ComputeU] = None


@dataclass(frozen=True)
class Accuracy:
    epsilon_Ha_per_atom: float = 1e-5                   # THE knob (Q1, ruled 2026-11-01)
    representation: Optional[dict] = None               # STATED: {"basis": ...} / {"pw_cutoff_Ry": ...} / {"pp": ...}
    kmesh: Optional[tuple[int, int, int]] = None        # STATED, else derived from ε and the cell
    scf: Optional[dict] = None                          # STATED method: {"mixer": ..., "accelerator": ..., "max_iter": ...}

    def __hash__(self) -> int:
        return hash((self.epsilon_Ha_per_atom, self.kmesh,
                     json.dumps(self.representation, sort_keys=True), json.dumps(self.scf, sort_keys=True)))


@dataclass(frozen=True)
class RunSpec:
    label: str
    structure: Structure
    physics: Physics = field(default_factory=Physics)
    task: Task = field(default_factory=Task)
    accuracy: Accuracy = field(default_factory=Accuracy)

    # -- derived -----------------------------------------------------------
    @property
    def num_electrons(self) -> int:
        """All-electron count (a back end's pseudopotential decides its valence count)."""
        return sum(s.Z for s in self.structure.sites)

    @property
    def summary(self) -> str:
        p = self.physics
        s = f"{self.label} · {p.theory}" + (f"/{p.functional}" if p.theory == "DFT" and p.functional else "")
        if p.hubbard:
            s += " +U"
        if p.spin.polarized and (p.spin.multiplicity or 0) > 1:
            s += f" · 2S+1={p.spin.multiplicity}"
        return s + f" · ε={self.accuracy.epsilon_Ha_per_atom:g}"

    # -- validation (back-end-free) ------------------------------------------
    def validate(self) -> "RunSpec":
        """Raise SpecError listing EVERY defect; return self when clean."""
        problems: list[str] = []
        st, ph, tk, ac = self.structure, self.physics, self.task, self.accuracy
        _check(problems, st.kind in ("molecule", "crystal"), f"structure.kind {st.kind!r} not in ('molecule', 'crystal')")
        _check(problems, len(st.sites) > 0, "structure.sites is empty")
        for i, s in enumerate(st.sites):
            _check(problems, s.element in Z_OF, f"structure.sites[{i}].element {s.element!r} is not an element")
            if st.kind == "crystal":
                _check(problems, s.frac is not None and s.xyz is None, f"structure.sites[{i}]: a crystal site states frac, not xyz")
            else:
                _check(problems, s.xyz is not None and s.frac is None, f"structure.sites[{i}]: a molecule site states xyz, not frac")
        if st.kind == "crystal":
            _check(problems, st.lattice_bohr is not None and len(st.lattice_bohr) == 3
                   and all(len(r) == 3 for r in st.lattice_bohr), "structure.lattice_bohr must be 3x3 for a crystal")
        else:
            _check(problems, st.lattice_bohr is None, "a molecule has no lattice_bohr")
        tags = [s.tag for s in st.sites if s.tag]
        _check(problems, len(tags) == len(set(tags)), "structure.sites: tags must be unique")

        _check(problems, ph.theory in THEORIES, f"physics.theory {ph.theory!r} not in {THEORIES}")
        _check(problems, ph.theory != "DFT" or bool(ph.functional), "physics.functional is required for DFT")
        _check(problems, ph.core in CORES, f"physics.core {ph.core!r} not in {CORES}")
        _check(problems, ph.relativity in RELATIVITY, f"physics.relativity {ph.relativity!r} not in {RELATIVITY}")
        _check(problems, ph.occupations.smearing in SMEARINGS, f"physics.occupations.smearing {ph.occupations.smearing!r} not in {SMEARINGS}")
        _check(problems, (ph.occupations.smearing == "none") == (ph.occupations.kT_Ha is None),
               "physics.occupations: kT_Ha is stated exactly when smearing is not 'none'")

        sp = ph.spin
        if sp.multiplicity is not None:
            _check(problems, st.kind == "molecule", "physics.spin.multiplicity is for molecules; a crystal states site_moments")
            _check(problems, sp.multiplicity >= 1, "physics.spin.multiplicity must be >= 1")
            _check(problems, (self.num_electrons - (sp.multiplicity - 1)) % 2 == 0,
                   f"multiplicity {sp.multiplicity} has the wrong parity for {self.num_electrons} electrons")
            _check(problems, sp.polarized or sp.multiplicity == 1, "an open shell (multiplicity > 1) must be polarized")
        if sp.site_moments is not None:
            _check(problems, st.kind == "crystal", "physics.spin.site_moments is for crystals")
            _check(problems, len(sp.site_moments) == len(st.sites), "physics.spin.site_moments needs one entry per site")
            _check(problems, sp.polarized, "site_moments on an unpolarized run")

        for k, h in enumerate(ph.hubbard):
            _check(problems, all(t in tags for t in h.sites), f"physics.hubbard[{k}].sites name unknown tags {h.sites}")
            _check(problems, h.form in U_FORMS, f"physics.hubbard[{k}].form {h.form!r} not in {U_FORMS}")
            _check(problems, h.projector in U_PROJECTORS, f"physics.hubbard[{k}].projector {h.projector!r} not in {U_PROJECTORS}")
            _check(problems, 0 <= h.l <= 3, f"physics.hubbard[{k}].l must be 0..3")
            _check(problems, h.U_eV is not None or tk.kind == "compute_U",
                   f"physics.hubbard[{k}].U_eV is unset and the task does not compute it")
        _check(problems, ph.theory == "DFT" or not ph.hubbard, "DFT+U needs theory DFT")

        _check(problems, tk.kind in TASKS, f"task.kind {tk.kind!r} not in {TASKS}")
        _check(problems, (tk.kind == "compute_U") == (tk.compute_U is not None),
               "task.compute_U is stated exactly when task.kind is 'compute_U'")
        if tk.compute_U is not None:
            cu = tk.compute_U
            _check(problems, cu.method in U_METHODS, f"task.compute_U.method {cu.method!r} not in {U_METHODS}")
            _check(problems, bool(cu.sites) and all(t in tags for t in cu.sites), "task.compute_U.sites must name existing site tags")

        _check(problems, 0 < ac.epsilon_Ha_per_atom < 1, "accuracy.epsilon_Ha_per_atom must be in (0, 1)")
        _check(problems, ac.kmesh is None or (st.kind == "crystal" and all(n >= 1 for n in ac.kmesh)),
               "accuracy.kmesh is for a crystal, every division >= 1")
        if problems:
            raise SpecError(f"RunSpec '{self.label}':\n  " + "\n  ".join(problems))
        return self

    # -- JSON ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **_plain(asdict(self))}

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        d = dict(d)
        schema = d.pop("schema", SCHEMA)
        if schema != SCHEMA:
            raise SpecError(f"schema {schema!r}: this reader knows {SCHEMA!r}")
        return _read(cls, d, "")

    @classmethod
    def from_json(cls, text: str) -> "RunSpec":
        return cls.from_dict(json.loads(text))


def _check(problems: list[str], ok: bool, msg: str) -> None:
    if not ok:
        problems.append(msg)


def _plain(x: Any) -> Any:
    """asdict leaves tuples; JSON wants lists.  None is WRITTEN (null): a field whose default is not None (functional)
    must read back as None, not as its default."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


# The nested dataclass each field reads into (a tuple-of-dataclass field names its element type).
_NESTED: dict[tuple[type, str], tuple[type, bool]] = {
    (RunSpec, "structure"):    (Structure, False),
    (RunSpec, "physics"):      (Physics, False),
    (RunSpec, "task"):         (Task, False),
    (RunSpec, "accuracy"):     (Accuracy, False),
    (Structure, "sites"):      (Site, True),
    (Physics, "spin"):         (Spin, False),
    (Physics, "hubbard"):      (Hubbard, True),
    (Physics, "occupations"):  (Occupations, False),
    (Task, "compute_U"):       (ComputeU, False),
}
_DICT_FIELDS = {(Structure, "source"), (Accuracy, "representation"), (Accuracy, "scf")}


def _tuplify(v: Any) -> Any:
    return tuple(_tuplify(x) for x in v) if isinstance(v, list) else v


def _read(cls: type, d: Any, path: str):
    if not isinstance(d, dict):
        raise SpecError(f"{path or 'spec'}: expected an object, got {type(d).__name__}")
    legal = {f.name for f in fields(cls)}
    kw = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue                                     # a comment
        p = f"{path}.{k}" if path else k
        if k not in legal:
            raise SpecError(f"unknown key '{p}'; legal keys here: {sorted(legal)}")
        if (cls, k) in _NESTED and v is not None:
            sub, many = _NESTED[(cls, k)]
            kw[k] = tuple(_read(sub, x, f"{p}[{i}]") for i, x in enumerate(v)) if many else _read(sub, v, p)
        elif (cls, k) in _DICT_FIELDS:
            kw[k] = v
        else:
            kw[k] = _tuplify(v)
    try:
        return cls(**kw)
    except TypeError as e:                               # a REQUIRED key missing
        raise SpecError(f"{path or 'spec'}: {e}") from None


# --------------------------------------------------------------------------
# Constructors the app (and the agent) use
# --------------------------------------------------------------------------
def molecule(label: str, numbers, positions_bohr, *, theory: str = "HF", functional: Optional[str] = None,
             multiplicity: Optional[int] = None, core: str = "all-electron", **kw) -> RunSpec:
    """A molecule spec from Z + flat 3N bohr positions (the shape the picker and .xyz reader give)."""
    numbers = list(numbers); pos = list(positions_bohr)
    if len(pos) != 3 * len(numbers):
        raise SpecError("positions must be flat 3N (bohr)")
    sites = tuple(Site(_ELEMENTS[z - 1], xyz=(float(pos[3*i]), float(pos[3*i+1]), float(pos[3*i+2])))
                  for i, z in enumerate(numbers))
    open_shell = multiplicity is not None and multiplicity > 1
    physics = Physics(theory=theory, functional=functional if theory == "DFT" else None, core=core,
                      spin=Spin(polarized=open_shell, multiplicity=multiplicity), relativity="none")
    return RunSpec(label, Structure("molecule", sites), physics, **kw).validate()
