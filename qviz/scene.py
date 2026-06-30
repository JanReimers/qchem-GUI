"""
qviz.scene -- turn qviz.data records into PyVista actors.

This is the *only* place that knows about VTK/PyVista. Frontends (the headless
artifact renderer and the desktop app) both build their scenes from here, so the
look-and-feel stays consistent.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from .data import Structure, ScalarField, VectorField

# CPK-ish colours and covalent radii (bohr, scaled for display)
_CPK = {"H": "#e6e6e6", "O": "#ff4d3d", "C": "#3a3a3a", "N": "#4d6dff",
        "Si": "#f0c33c", "Li": "#9b59b6", "Na": "#a0c4ff"}
_RAD = {"H": 0.45, "O": 0.72, "C": 0.70, "N": 0.68, "Si": 1.05,
        "Li": 1.20, "Na": 1.40}
_DEFAULT = ("#b36bff", 0.7)


def field_to_imagedata(f: ScalarField) -> pv.ImageData:
    """ScalarField -> VTK ImageData (zero-copy on the value array)."""
    grid = pv.ImageData(dimensions=f.dims, spacing=f.spacing, origin=f.origin)
    # VTK is Fortran-ordered point data; ravel(order="F") keeps it a view.
    grid.point_data[f.name] = f.values.ravel(order="F")
    grid.set_active_scalars(f.name)
    return grid


def add_structure(p: pv.Plotter, s: Structure, *, bond_scale: float = 1.3):
    """Ball-and-stick: spheres for atoms, tubes for bonds within range."""
    for sym, R in zip(s.symbols, s.positions):
        colour, rad = (_CPK.get(sym, _DEFAULT[0]), _RAD.get(sym, _DEFAULT[1]))
        p.add_mesh(pv.Sphere(radius=rad, center=R),
                   color=colour, smooth_shading=True,
                   specular=0.6, specular_power=20, name=f"atom_{sym}_{R}")
    # naive bond detection
    P = s.positions
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = np.linalg.norm(P[i] - P[j])
            rcut = bond_scale * (_RAD.get(s.symbols[i], 0.7) +
                                 _RAD.get(s.symbols[j], 0.7))
            if d < rcut:
                p.add_mesh(pv.Tube(P[i], P[j], radius=0.12, n_sides=20),
                           color="#9aa0aa", smooth_shading=True)


def add_isosurface(p: pv.Plotter, f: ScalarField, *, levels=None,
                   cmap="viridis", opacity=1.0):
    """Shaded isosurface(s). For signed fields draws +/- lobes in two colours."""
    grid = field_to_imagedata(f)
    if f.signed:
        v = float(np.abs(f.values).max())
        lo, hi = -0.18 * v, 0.18 * v
        for lvl, col in ((hi, "#2b7bff"), (lo, "#ff5a4d")):
            iso = grid.contour([lvl])
            if iso.n_points:
                p.add_mesh(iso, color=col, smooth_shading=True,
                           opacity=opacity, specular=0.5, specular_power=15)
    else:
        if levels is None:
            vmax = float(f.values.max())
            levels = [0.02 * vmax, 0.002 * vmax]
        for lvl, op in zip(levels, (1.0, 0.35)):
            iso = grid.contour([lvl])
            if iso.n_points:
                p.add_mesh(iso, scalars=f.name, cmap=cmap, opacity=op,
                           smooth_shading=True, show_scalar_bar=False)


def add_vector_glyphs(p: pv.Plotter, vf: VectorField, *, factor=0.6, stride=2,
                      normalize=True):
    """Arrow glyphs. Density gradients span orders of magnitude near nuclei, so
    by default we draw *uniform-length* arrows coloured by |v| (log) -- you see
    direction everywhere instead of a few giant spikes."""
    grid = pv.ImageData(dimensions=vf.vectors.shape[:3],
                        spacing=vf.spacing, origin=vf.origin)
    vecs = vf.vectors.reshape(-1, 3, order="F")
    mag = np.linalg.norm(vecs, axis=1)
    grid["mag"] = np.log10(mag + 1e-6)
    if normalize:
        unit = vecs / (mag[:, None] + 1e-12)
        grid["vec"] = unit
        glyphs = grid.glyph(orient="vec", scale=False, factor=factor)
    else:
        grid["vec"] = vecs
        glyphs = grid.glyph(orient="vec", scale="mag", factor=factor)
    p.add_mesh(glyphs, scalars="mag", cmap="plasma", show_scalar_bar=False)


def style(p: pv.Plotter):
    """House dark theme."""
    p.set_background("#101216")
    try:
        p.enable_anti_aliasing("ssaa")   # skipped gracefully on GPU-less boxes
    except Exception:
        pass
