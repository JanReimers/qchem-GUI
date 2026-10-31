"""qviz -- the qchem (SOLID²) visualization app's model layer.

Pipeline:  RunSpec (what)  -->  Backend (how: SOLID² in process, later CP2K / QE / ABINIT / VASP)
           -->  Result (neutral)  -->  PyVista / pyqtgraph

See doc/RunSpecPlan.md.  qviz.data holds the frontend-agnostic records the views draw.
"""
from .data import Structure, ScalarField, VectorField, SCFCell, SCFRow
from .spec import RunSpec, SpecError, molecule
from .backends import Backend, BackendRegistry, FakeBackend, Solid2Backend, default_registry

__all__ = ["Structure", "ScalarField", "VectorField", "SCFCell", "SCFRow", "RunSpec", "SpecError", "molecule",
           "Backend", "BackendRegistry", "FakeBackend", "Solid2Backend", "default_registry"]
