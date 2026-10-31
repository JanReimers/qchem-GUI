"""
qviz.data -- the records the views draw.

Plain, frontend-agnostic data: Structure, ScalarField, VectorField, SCFRow (of SCFCells).  Every back end's
Result (qviz.backends.base) hands these out, so nothing in the frontends knows which code ran.

Design rules (mirrors CLAUDE.md "expose high-level answers, not internals"):
  * arrays are contiguous NumPy (zero-copy views of Blaze storage via nanobind),
  * grids are described, never re-derived (origin, spacing, dims),
  * no plotting type ever leaks across this line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
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
@dataclass(frozen=True)
class SCFCell:
    """One column of a code's SCF iteration table, under that code's OWN title (SOLID²'s console headers, CP2K's
    `Convergence`, pw.x's `estimated scf accuracy`, ABINIT's `residm`) -- never a cross-code synonym.  Mirrors the
    lib's IterationCell (SCFIterator.C), so SOLID²'s row crosses the bridge one-to-one."""
    title: str
    value: float                   # NaN: no number this iteration (the console's "----"), or a text-only column
    scale: str = "linear"          # "log": a convergence measure, drawn as |value| on a log axis
    text: str = ""                 # the cell as the code printed it ("" -> formatted from value)
    tolerance: Optional[float] = None   # the gate this column is held to, when the code states one
    unit: str = ""                 # the code's own unit for the column ("Ry", "Bohr mag/cell"); "" = hartree / none

    def shown(self) -> str:
        if self.text:
            return self.text
        if math.isnan(self.value):
            return "----"
        return f"{self.value:.2e}" if self.scale == "log" else f"{self.value:.10g}"


@dataclass(frozen=True)
class SCFRow:
    """One SCF iteration: the code's own cells in display order (the iteration number is not a cell), plus the
    total energy in hartree -- the one neutral number every code's row carries, whatever its own column's units."""
    iteration: int
    energy: float                  # total energy (hartree)
    cells: tuple[SCFCell, ...]

    def value(self, title: str) -> float:
        """The cell titled `title`'s value; NaN when this row has no such column."""
        return next((c.value for c in self.cells if c.title == title), math.nan)

    @property
    def titles(self) -> tuple[str, ...]:
        return tuple(c.title for c in self.cells)
