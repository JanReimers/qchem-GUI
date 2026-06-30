"""
qviz.data -- the C++ <-> Python boundary.

This module defines the *only* thing the GUI knows about the numerical core:
a handful of plain, frontend-agnostic data records plus a `ComputeBackend`
Protocol that emits them.

The whole architecture rests on this seam:

    qchem (C++)  --nanobind-->  ComputeBackend  -->  qviz frontends (PyVista, pyqtgraph)

Today `backend_analytic.AnalyticBackend` implements `ComputeBackend` with
closed-form chemistry so we can build and admire the plots. Tomorrow you write
`backend_qchem.QChemBackend` as a nanobind module exposing the real density /
structure / SCF trace from src/. Nothing in the frontends changes -- they only
ever see these records.

Design rules (mirrors CLAUDE.md "expose high-level answers, not internals"):
  * arrays are contiguous NumPy (zero-copy views of Blaze storage via nanobind),
  * grids are described, never re-derived (origin, spacing, dims),
  * no plotting type ever leaks across this line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Iterator, runtime_checkable
import numpy as np


# --------------------------------------------------------------------------
# Geometry / structure
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Structure:
    """Atoms (and, for solids, lattice vectors). Coordinates in bohr."""
    symbols: list[str]
    positions: np.ndarray          # (natom, 3) float64, bohr
    numbers: np.ndarray            # (natom,) int   atomic numbers (Z)
    lattice: np.ndarray | None = None   # (3,3) or None for molecules/atoms

    def __post_init__(self) -> None:
        assert self.positions.shape == (len(self.symbols), 3)
        assert self.numbers.shape == (len(self.symbols),)

    @property
    def is_periodic(self) -> bool:
        return self.lattice is not None


# --------------------------------------------------------------------------
# Scalar field on a regular grid (density, orbital, potential, ELF, ...)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScalarField:
    """A scalar sampled on an axis-aligned regular grid.

    `values` has shape `dims` (nx, ny, nz), C-order. `origin` is the position of
    voxel (0,0,0); `spacing` is the per-axis step (bohr). This maps 1:1 onto a
    VTK ImageData with zero copies, which is why isosurfaces/slices are instant.
    """
    name: str
    values: np.ndarray             # (nx, ny, nz) float64
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    signed: bool = False           # True for orbitals (+/- lobes), False for densities

    @property
    def dims(self) -> tuple[int, int, int]:
        return self.values.shape  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Vector field on a regular grid (gradient, current density, forces field, ...)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VectorField:
    name: str
    vectors: np.ndarray            # (nx, ny, nz, 3) float64
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]


# --------------------------------------------------------------------------
# Live SCF telemetry -- one row per iteration, streamed during a run
# --------------------------------------------------------------------------
@dataclass
class SCFStep:
    iteration: int
    energy: float                  # total energy (hartree)
    dE: float                      # |E_n - E_{n-1}|
    commutator: float              # ||[F, D]||  (DIIS error)
    drho: float                    # ||rho_n - rho_{n-1}||


# --------------------------------------------------------------------------
# The backend contract
# --------------------------------------------------------------------------
@runtime_checkable
class ComputeBackend(Protocol):
    """What every numerical backend must offer the GUI. nanobind fills this."""

    def structure(self) -> Structure: ...

    def density(self, n: int = 64, pad: float = 5.0) -> ScalarField:
        """Total electron density on an n^3 grid padded `pad` bohr past the atoms."""

    def orbital(self, index: int, n: int = 64, pad: float = 5.0) -> ScalarField:
        """A (signed) molecular orbital -- e.g. the HOMO."""

    def density_gradient(self, n: int = 24, pad: float = 5.0) -> VectorField: ...

    def run_scf(self) -> Iterator[SCFStep]:
        """Drive an SCF, yielding one SCFStep per iteration so the GUI can
        stream convergence live. The real backend yields from inside the C++
        iterate loop via a callback; the frontend never blocks."""
