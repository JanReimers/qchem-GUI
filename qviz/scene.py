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


def add_height_surface(p: pv.Plotter, f: ScalarField, *, frac=0.5, axis=2,
                       cmap=None, relief=0.5, log=False, log_floor=1e-6) -> int:
    """Lift a single plane of the field into a 3D SURFACE: height = f on the plane
    normal to `axis` (0=x,1=y,2=z) at position `frac`∈[0,1] (the classic
    `plot_surface` view). The two in-plane axes become the surface's u,v; the field
    value is the height. Returns the chosen slab index. Height is scaled so max|f|
    ≈ relief·(in-plane extent).

    log=True plots log₁₀(f) (density only -- f must be >=0), which compresses the
    nuclear spike and brings out the shell structure as shoulders on the profile;
    values below log_floor are clamped and the baseline is shifted to that floor."""
    vals = np.asarray(f.values)
    dims, org, sp = f.dims, f.origin, f.spacing
    k = int(round(frac * (dims[axis] - 1)))
    slab = np.take(vals, k, axis=axis)               # 2D over the two in-plane axes
    a0, a1 = (i for i in (0, 1, 2) if i != axis)     # in-plane axes, natural order
    logged = log and not f.signed
    if logged:
        slab = np.log10(np.maximum(slab, log_floor)) - np.log10(log_floor)  # >=0, floor->0
    U, V = np.meshgrid(org[a0] + sp[a0] * np.arange(dims[a0]),
                       org[a1] + sp[a1] * np.arange(dims[a1]), indexing="ij")
    span = max(sp[a0] * (dims[a0] - 1), sp[a1] * (dims[a1] - 1))
    peak = float(np.abs(slab).max()) or 1e-9
    H = relief * span                                # scaled height of max|f|
    grid = pv.StructuredGrid(U, V, slab * (H / peak))
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
