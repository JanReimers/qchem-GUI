"""qviz.backends -- the codes a RunSpec can run on (doc/RunSpecPlan.md §2-4)."""
from .base import Backend, BackendRegistry, FieldRef, Job, Result, data_structure
from .fake import FakeBackend
from .solid2 import Solid2Backend


def default_registry() -> BackendRegistry:
    """SOLID² first (the default when several can run a spec -- PROVISIONAL, RunSpecPlan §10 Q6), then the fake."""
    return BackendRegistry([Solid2Backend(), FakeBackend()])


__all__ = ["Backend", "BackendRegistry", "FieldRef", "Job", "Result", "data_structure",
           "FakeBackend", "Solid2Backend", "default_registry"]
