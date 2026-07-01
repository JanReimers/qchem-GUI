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
                   cmap="viridis", opacity=1.0, signed_frac=0.18, style="surface"):
    """Shaded isosurface(s). For signed fields (orbitals, Δρ) draws +/- lobes in
    two colours at ±signed_frac·max|f| -- lower signed_frac reveals weaker lobes
    (e.g. the diffuse side of a difference density). style='surface'|'wireframe'."""
    grid = field_to_imagedata(f)
    wire = (style == "wireframe")
    if f.signed:
        v = float(np.abs(f.values).max())
        lo, hi = -signed_frac * v, signed_frac * v
        for lvl, col in ((hi, "#2b7bff"), (lo, "#ff5a4d")):
            iso = grid.contour([lvl])
            if iso.n_points:
                p.add_mesh(iso, color=col, style=style, smooth_shading=not wire,
                           opacity=opacity, specular=0.5, specular_power=15)
    else:
        if levels is None:
            vmax = float(f.values.max())
            levels = [0.02 * vmax, 0.002 * vmax]
        for lvl, op in zip(levels, (1.0, 0.35)):
            iso = grid.contour([lvl])
            if iso.n_points:
                p.add_mesh(iso, scalars=f.name, cmap=cmap, opacity=(1.0 if wire else op),
                           style=style, smooth_shading=not wire, show_scalar_bar=False)


def add_height_surface(p: pv.Plotter, f: ScalarField, *, z_frac=0.5,
                       cmap=None, relief=0.5) -> int:
    """Lift a single z-plane of the field into a 3D SURFACE: height = f(x,y) on the
    selected plane (the classic `plot_surface` view). Peaks/valleys of ρ or an MO
    on a slice, complementing the isosurface. z_frac ∈ [0,1] picks the plane;
    returns the chosen k index. Height is scaled so max|f| ≈ relief·(in-plane extent)."""
    vals = np.asarray(f.values)
    nx, ny, nz = f.dims
    ox, oy, _ = f.origin
    dx, dy, _ = f.spacing
    k = int(round(z_frac * (nz - 1)))
    slab = vals[:, :, k]                              # (nx, ny)
    X, Y = np.meshgrid(ox + dx * np.arange(nx),
                       oy + dy * np.arange(ny), indexing="ij")
    span = max(dx * (nx - 1), dy * (ny - 1))
    peak = float(np.abs(slab).max()) or 1e-9
    H = relief * span                                # scaled height of max|f|
    grid = pv.StructuredGrid(X, Y, slab * (H / peak))
    grid["value"] = grid.points[:, 2]                # colour by height (== scaled f), point-aligned
    if cmap is None:
        cmap = "coolwarm" if f.signed else "hot"
    # signed field -> symmetric clim so zero maps to the diverging map's white midpoint
    clim = [-H, H] if f.signed else None
    p.add_mesh(grid, scalars="value", cmap=cmap, clim=clim, show_scalar_bar=False,
               show_edges=True, edge_color="#404040", line_width=1,
               smooth_shading=True, specular=0.25)
    return k


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
