"""
Render every demo artifact headlessly (off-screen VTK + matplotlib), so the
plots can be inspected without a display. Run:

    python scripts/make_artifacts.py

Produces in out/:
    density_iso.png      shaded density isosurface + ball-and-stick
    orbital_homo.png     two-lobe signed MO isosurface
    gradient_field.png   grad(rho) vector glyphs
    slice_sweep.gif      a slider sweeping a 2D slice through the 3D density
    scf_convergence.gif  live-streamed dE / ||[F,D]|| / drho during SCF
    water.qproj.h5       a saved project (workspace round-trip)
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

from qviz import AnalyticBackend
from qviz import scene, project
from qviz.scene import field_to_imagedata

OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)
be = AnalyticBackend()
struct = be.structure()


def _cam(p):
    p.camera_position = "yz"
    p.camera.azimuth = 35
    p.camera.elevation = 20
    p.reset_camera()
    p.camera.zoom(1.3)


def density_iso():
    f = be.density(n=96)
    p = pv.Plotter(off_screen=True, window_size=(1100, 850))
    scene.style(p)
    scene.add_isosurface(p, f, cmap="viridis")
    scene.add_structure(p, struct)
    p.add_text("Electron density  (rho)  -  water", font_size=12, color="w")
    _cam(p)
    p.screenshot(str(OUT / "density_iso.png"))
    print("  density_iso.png")


def orbital_homo():
    f = be.orbital(0, n=96)
    p = pv.Plotter(off_screen=True, window_size=(1100, 850))
    scene.style(p)
    scene.add_isosurface(p, f)            # signed -> blue/red lobes
    scene.add_structure(p, struct)
    p.add_text("HOMO  (O 2p_z lone pair)  -  signed isosurface",
               font_size=12, color="w")
    _cam(p)
    p.screenshot(str(OUT / "orbital_homo.png"))
    print("  orbital_homo.png")


def gradient_field():
    from qviz.data import VectorField
    vf = be.density_gradient(n=21, pad=3.0)
    # show the field on the molecular plane (central y slab) for legibility
    j = vf.vectors.shape[1] // 2
    plane = VectorField(vf.name, vf.vectors[:, j:j+1, :, :],
                        (vf.origin[0], vf.origin[1] + j*vf.spacing[1], vf.origin[2]),
                        vf.spacing)
    p = pv.Plotter(off_screen=True, window_size=(1100, 850))
    scene.style(p)
    scene.add_vector_glyphs(p, plane, factor=0.7, normalize=True)
    scene.add_structure(p, struct)
    p.add_text("grad(rho)  on the molecular plane", font_size=12, color="w")
    p.camera_position = "xz"; p.camera.zoom(1.4)
    p.screenshot(str(OUT / "gradient_field.png"))
    print("  gradient_field.png")


def slice_sweep():
    """The 'knob-driven 2D slice of a 3D field' feature, baked to a GIF."""
    f = be.density(n=96)
    grid = field_to_imagedata(f)
    lo = f.origin[2]
    n = f.dims[2]
    p = pv.Plotter(off_screen=True, window_size=(900, 800))
    scene.style(p)
    scene.add_structure(p, struct)
    p.add_text("2D slice  (slider sweeps z)", font_size=12, color="w")
    _cam(p)
    p.open_gif(str(OUT / "slice_sweep.gif"), fps=12)
    zs = np.concatenate([np.linspace(0.25, 0.75, 26),
                         np.linspace(0.75, 0.25, 26)])
    for frac in zs:
        zpos = lo + frac * (n - 1) * f.spacing[2]
        sl = grid.slice(normal="z", origin=(0, 0, zpos))
        p.add_mesh(sl, name="slice", cmap="inferno", show_scalar_bar=False,
                   clim=[0, 0.06 * float(f.values.max())])
        p.write_frame()
    p.close()
    print("  slice_sweep.gif")


def scf_convergence():
    """Stream the SCF trace into a growing semilog plot -> GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    steps = list(be.run_scf())
    it = [s.iteration for s in steps]
    dE = [s.dE for s in steps]
    comm = [s.commutator for s in steps]
    drho = [s.drho for s in steps]

    plt.style.use("dark_background")
    frames = []
    for k in range(1, len(steps) + 1):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=120)
        ax.semilogy(it[:k], dE[:k], "o-", label=r"$|\Delta E|$", color="#4dd0e1")
        ax.semilogy(it[:k], comm[:k], "s-", label=r"$\|[F,D]\|$", color="#ff8a65")
        ax.semilogy(it[:k], drho[:k], "^-", label=r"$\|\Delta\rho\|$", color="#aed581")
        ax.set_xlim(0, len(steps) - 1)
        ax.set_ylim(1e-7, 1e1)
        ax.set_xlabel("SCF iteration"); ax.set_ylabel("residual")
        ax.set_title(f"Live SCF convergence  -  E = {steps[k-1].energy:.6f} Ha")
        ax.legend(loc="upper right"); ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(buf)
        plt.close(fig)
    frames += [frames[-1]] * 8
    imageio.mimsave(OUT / "scf_convergence.gif", frames, fps=4)
    print("  scf_convergence.gif")
    return steps


def save_project(steps):
    project.save(str(OUT / "water.qproj.h5"),
                 title="Water / LDA demo", backend="AnalyticBackend",
                 structure=struct,
                 fields={"density": be.density(n=64),
                         "homo": be.orbital(0, n=64)},
                 scf=steps,
                 viewstate={"iso_level": 0.02, "slice_axis": "z",
                            "camera": "yz+35+20"})
    # round-trip check
    p = project.load(str(OUT / "water.qproj.h5"))
    assert p["fields"]["density"].values.shape == (64, 64, 64)
    assert p["scf"].shape[0] == len(steps)
    print(f"  water.qproj.h5  (reloaded: {p['meta']['title']!r}, "
          f"{len(p['fields'])} fields, {p['scf'].shape[0]} scf rows)")


if __name__ == "__main__":
    print("rendering artifacts -> out/")
    density_iso()
    orbital_homo()
    gradient_field()
    slice_sweep()
    steps = scf_convergence()
    save_project(steps)
    print("done.")
