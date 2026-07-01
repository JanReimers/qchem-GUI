"""
qviz.workspace -- the Workspace{Run} model (Pillar 2).

The load-bearing foundation: the app is multi-run from line one. Everything
(layouts, compare, optimizer, papermill) hangs off this.

Deliberately **Qt-agnostic** -- pure Python, observer callbacks, and the backend
injected by a factory. So the model obeys DIP just like the seam it sits on: it
depends on the `ComputeBackend` abstraction, never on a concrete backend or on Qt.

A `Run` carries a lifecycle (pending → running → ready / failed) so the GUI never
assumes compute is instant. Local backends resolve synchronously today; a future
RemoteBackend resolves over the network through the *same* interface. The Qt layer
can run `Run.compute()` on a worker thread later -- the status + notify shape is
already here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import numpy as np

from .data import Structure, ScalarField, VectorField, ComputeBackend


class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    READY   = "ready"
    FAILED  = "failed"


@dataclass(frozen=True)
class RunSpec:
    """What DEFINES a calculation -- the inputs. Hashable + serializable, so it can
    be a dict key, written to a project file, or shipped to a remote job."""
    label: str
    numbers: tuple[int, ...]
    positions: tuple[float, ...]    # flat 3N, bohr
    method: str = "HF"             # HF | LDA | ...  (only HF wired in the binding today)
    basis: str = "dzvp"

    @property
    def summary(self) -> str:
        return f"{self.label} · {self.method}/{self.basis}"


# A factory maps a spec -> a ComputeBackend. THIS is the DIP seam: the Workspace
# never names QChemBackend/AnalyticBackend/RemoteBackend -- local vs remote is just
# which factory you inject.
BackendFactory = Callable[[RunSpec], ComputeBackend]


class Run:
    """One calculation instance: a spec + a lifecycle + (when ready) sampled fields.
    Fields are sampled lazily on access -- creating the Run runs the SCF (energy),
    but a 64^3 density grid is only built when a panel actually asks for it."""

    def __init__(self, spec: RunSpec, make_backend: BackendFactory):
        self.spec = spec
        self.status: RunStatus = RunStatus.PENDING
        self.error: Optional[str] = None
        self.energy: Optional[float] = None
        self._make_backend = make_backend
        self._be: Optional[ComputeBackend] = None
        self._structure: Optional[Structure] = None
        self._cache: dict = {}      # memoized sampled fields, keyed by (kind, args)
        self._listeners: list[Callable[["Run"], None]] = []

    # -- observer (Qt-agnostic) -------------------------------------------
    def on_change(self, cb: Callable[["Run"], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            cb(self)

    # -- lifecycle --------------------------------------------------------
    def compute(self) -> None:
        """Run the calculation. Synchronous for local backends; the status + notify
        shape lets the Qt layer push this onto a worker thread later unchanged."""
        self.status = RunStatus.RUNNING
        self._notify()
        try:
            self._be = self._make_backend(self.spec)
            self._structure = self._be.structure()
            self._cache.clear()      # a fresh converge invalidates any memoized grids
            te = getattr(self._be, "total_energy", None)
            self.energy = te() if callable(te) else None
            self.status = RunStatus.READY
        except Exception as e:           # noqa: BLE001 -- surface any backend failure as a state
            self.error = str(e)
            self.status = RunStatus.FAILED
        self._notify()

    @property
    def is_ready(self) -> bool:
        return self.status is RunStatus.READY

    # -- results (valid only when READY; delegate to the backend) ---------
    def structure(self) -> Structure:
        self._require_ready()
        return self._structure

    def density(self, n: int = 64, pad: float = 5.0) -> ScalarField:
        return self._sampled(("density", n, pad), lambda: self._be.density(n, pad))

    def orbital(self, index: int = 0, n: int = 64, pad: float = 5.0) -> ScalarField:
        return self._sampled(("orbital", index, n, pad), lambda: self._be.orbital(index, n, pad))

    def density_gradient(self, n: int = 22, pad: float = 4.0) -> VectorField:
        return self._sampled(("grad", n, pad), lambda: self._be.density_gradient(n, pad))

    def _sampled(self, key, sample):
        """Lazy + memoized: sample the grid on first ask, reuse thereafter (the
        cache is cleared on each Converge)."""
        self._require_ready()
        if key not in self._cache:
            self._cache[key] = sample()
        return self._cache[key]

    def run_scf(self):
        self._require_ready(); return self._be.run_scf()

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(f"Run '{self.spec.summary}' is {self.status.value}, not ready")


class Workspace:
    """A collection of Runs + selection. The unit saved to a .qproj.h5 (see project.py).
    Project-level multiplicity is handled by opening multiple Workspaces in separate
    windows; run-level multiplicity lives here, as data."""

    def __init__(self, title: str = "Untitled", make_backend: Optional[BackendFactory] = None):
        self.title = title
        self.make_backend = make_backend     # default factory for add_run()
        self.runs: list[Run] = []
        self.selected: Optional[Run] = None
        self._listeners: list[Callable[["Workspace"], None]] = []

    def on_change(self, cb: Callable[["Workspace"], None]) -> None:
        self._listeners.append(cb)

    def _notify(self) -> None:
        for cb in self._listeners:
            cb(self)

    def add_run(self, spec: RunSpec, make_backend: Optional[BackendFactory] = None,
                compute: bool = True) -> Run:
        factory = make_backend or self.make_backend
        if factory is None:
            raise ValueError("no backend factory: pass make_backend to add_run or the Workspace")
        run = Run(spec, factory)
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
