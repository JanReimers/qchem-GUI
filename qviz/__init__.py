"""qviz -- a feasibility spike for the qchem6 visualization layer.

Pipeline:  qchem (C++)  --nanobind-->  ComputeBackend  -->  PyVista / pyqtgraph

See qviz.data for the (frontend-agnostic) boundary every backend implements.
"""
from .data import (Structure, ScalarField, VectorField, SCFStep,
                   ComputeBackend)
from .backend_analytic import AnalyticBackend, water

__all__ = ["Structure", "ScalarField", "VectorField", "SCFStep",
           "ComputeBackend", "AnalyticBackend", "water"]
