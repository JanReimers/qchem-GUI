"""
qviz.data -- the records the views draw.

Plain, frontend-agnostic data: Structure, ScalarField, VectorField, SCFStep.  Every back end's
Result (qviz.backends.base) hands these out, so nothing in the frontends knows which code ran.

Design rules (mirrors CLAUDE.md "expose high-level answers, not internals"):
  * arrays are contiguous NumPy (zero-copy views of Blaze storage via nanobind),
  * grids are described, never re-derived (origin, spacing, dims),
  * no plotting type ever leaks across this line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
