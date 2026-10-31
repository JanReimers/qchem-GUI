"""qviz.backends -- the codes a RunSpec can run on (doc/RunSpecPlan.md §2-4)."""
from .base import Backend, BackendRegistry, FieldRef, Job, Result, data_structure
from .fake import FakeBackend
from .solid2 import Solid2Backend
from .cp2k import Cp2kBackend


def default_registry() -> BackendRegistry:
    """SOLID² first (the default when several can run a spec -- PROVISIONAL, RunSpecPlan §10 Q6), then CP2K, then the
    analytic stand-in."""
    return BackendRegistry([Solid2Backend(), Cp2kBackend(), FakeBackend()])


__all__ = ["Backend", "BackendRegistry", "FieldRef", "Job", "Result", "data_structure",
           "FakeBackend", "Solid2Backend", "Cp2kBackend", "default_registry"]
