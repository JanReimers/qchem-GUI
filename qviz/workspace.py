"""
qviz.workspace -- the Workspace{Run} model (Pillar 2).

The load-bearing foundation: the app is multi-run from line one. Everything
(layouts, compare, optimizer, papermill) hangs off this.

Deliberately **Qt-agnostic** -- pure Python, observer callbacks.  It obeys DIP: a Run
depends on the `Backend` abstraction (qviz.backends.base), never on a concrete code; the
Workspace holds a `BackendRegistry` and asks it by name.

A Run is ONE spec on ONE back end: `compute()` translates the spec (a Job: the native input
and the resolved settings, inspectable before anything runs), runs it -- streaming each SCF
iteration to the caller -- and keeps the neutral Result.  Lifecycle pending → running →
ready / failed, so the GUI never assumes compute is instant; the Qt layer calls
`compute()` on a worker thread.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from .backends.base import Backend, BackendRegistry, Job, Result
from .data import ScalarField, SCFStep, Structure, VectorField
from .spec import RunSpec


class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    READY   = "ready"
    FAILED  = "failed"


class Run:
    """One calculation: a spec + the back end it runs on + a lifecycle + (when ready) the Result.
    Fields are sampled lazily on access and memoized -- a 64^3 grid is built only when a panel asks."""

    def __init__(self, spec: RunSpec, backend: Backend):
        self.spec = spec
        self.backend = backend
        self.status: RunStatus = RunStatus.PENDING
        self.error: Optional[str] = None
        self.job: Optional[Job] = None
        self.result: Optional[Result] = None
        self._cache: dict = {}
        self._listeners: list[Callable[["Run"], None]] = []

    @property
    def backend_name(self) -> str:
        return self.backend.name

    @property
    def summary(self) -> str:
        return f"{self.spec.summary} · {self.backend_name}"

    @property
    def energy(self) -> Optional[float]:
        return self.result.energy_Ha if self.result else None

    # -- observer (Qt-agnostic) -------------------------------------------
    def on_change(self, cb: Callable[["Run"], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            cb(self)

    # -- lifecycle --------------------------------------------------------
    def compute(self, on_step: Optional[Callable[[SCFStep], None]] = None) -> None:
        """Translate + run.  Blocking: the Qt layer calls it on a worker thread.  `on_step(SCFStep)` streams the
        SCF live.  Any failure (a refused capability, a throw in the code) becomes the FAILED state."""
        self.status = RunStatus.RUNNING
        self.error = None
        self._notify()
        try:
            verdict = self.backend.capabilities().can_run(self.spec)
            if not verdict:
                raise RuntimeError(f"{self.backend_name} cannot run this spec: " + "; ".join(verdict.reasons))
            self.job = self.backend.translate(self.spec)
            self.result = self.backend.run(self.job, on_step)
            self._cache.clear()
            if not self.result.converged:
                raise RuntimeError(f"SCF did not converge in {len(self.result.scf)} iterations")
            self.status = RunStatus.READY
        except Exception as e:           # noqa: BLE001 -- surface any back-end failure as a state
            self.error = str(e)
            self.status = RunStatus.FAILED
        self._notify()

    @property
    def is_ready(self) -> bool:
        return self.status is RunStatus.READY

    # -- results (valid only when READY) ----------------------------------
    def structure(self) -> Structure:
        self._require_ready()
        return self.result.structure

    @property
    def scf(self) -> list[SCFStep]:
        """The trace recorded during the one run (no re-run: a replay is a UI animation over this)."""
        self._require_ready()
        return self.result.scf

    def density(self, n: int = 64, pad: float = 5.0) -> ScalarField:
        return self._field("density", n, pad)

    def density_gradient(self, n: int = 22, pad: float = 4.0) -> VectorField:
        return self._field("gradient", n, pad)

    def spin_density(self, n: int = 64, pad: float = 5.0) -> Optional[ScalarField]:
        """Signed ρ↑−ρ↓, or None when the run is unpolarized (or the back end cannot sample it)."""
        self._require_ready()
        return self._field("spin_density", n, pad) if "spin_density" in self.result.fields else None

    def orbital(self, index: int = 0, n: int = 64, pad: float = 5.0) -> ScalarField:
        """Level `index` in report('orbitals') order (0 = lowest)."""
        self._require_ready()
        if self.result.orbital is None:
            raise RuntimeError(f"{self.backend_name} does not sample orbitals")
        key = ("orbital", index, n, pad)
        if key not in self._cache:
            self._cache[key] = self.result.orbital(index, n, pad)
        return self._cache[key]

    def report(self, kind: str) -> dict:
        """'orbitals': every level (lowest first) + counts, the shape the orbital panel reads; {} otherwise."""
        self._require_ready()
        if kind != "orbitals":
            return {}
        r = self.result
        return {"energy": r.energy_Ha, "levels": r.levels, "num_occupied": r.num_occupied,
                "num_orbitals": len(r.levels)}

    def _field(self, name: str, n: int, pad: float):
        self._require_ready()
        key = (name, n, pad)
        if key not in self._cache:
            ref = self.result.fields.get(name)
            if ref is None:
                raise RuntimeError(f"{self.backend_name} gives no '{name}' field")
            self._cache[key] = ref.sample(n, pad)
        return self._cache[key]

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(f"Run '{self.summary}' is {self.status.value}, not ready")


class Workspace:
    """A collection of Runs + selection. The unit saved to a .qproj.h5 (see project.py)."""

    def __init__(self, title: str = "Untitled", registry: Optional[BackendRegistry] = None):
        self.title = title
        self.registry = registry or BackendRegistry()
        self.runs: list[Run] = []
        self.selected: Optional[Run] = None
        self._listeners: list[Callable[["Workspace"], None]] = []

    def on_change(self, cb: Callable[["Workspace"], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            cb(self)

    def add_run(self, spec: RunSpec, backend: Optional[str] = None, compute: bool = True) -> Run:
        """Add `spec` on the back end named `backend`, else on the registry's default for it (the first that can run
        it).  With no back end able to run it, the Run is still added -- and fails with the strip's reasons."""
        name = backend or self.registry.default_for(spec) or self.registry.names()[0]
        run = Run(spec, self.registry.get(name))
        run.on_change(lambda _r: self._notify())     # run state changes bubble up to the workspace
        self.runs.append(run)
        if self.selected is None:
            self.selected = run
        self._notify()
        if compute:
            run.compute()
        return run

    def select(self, run: Optional[Run]) -> None:
        if run is not None and run not in self.runs:
            raise ValueError("run is not in this workspace")
        self.selected = run
        self._notify()

    def remove(self, run: Run) -> None:
        self.runs.remove(run)
        if self.selected is run:
            self.selected = self.runs[0] if self.runs else None
        self._notify()
