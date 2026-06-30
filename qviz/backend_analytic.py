"""
qviz.backend_analytic -- a closed-form ComputeBackend stand-in.

>>> REPLACE-ME SEAM <<<
This is exactly the surface your nanobind module (`backend_qchem.QChemBackend`)
will re-implement against src/. Every method returns one of the records from
qviz.data; the frontends can't tell an analytic water from a converged HF water.

To keep the demo honest the numbers are *physically shaped*, not random:
  * water at its real geometry (r_OH = 1.81 bohr, angle 104.5 deg),
  * density = sum of atom-centred Slater-like clouds (Z electrons each),
  * a HOMO modelled as an O 2p_z lone pair (the classic two-lobe orbital),
  * an SCF "run" that decays like a real DIIS-accelerated HF: monotone-ish
    energy, commutator and drho falling a few orders of magnitude.
"""
from __future__ import annotations

import numpy as np
from .data import (Structure, ScalarField, VectorField, SCFStep,
                   ComputeBackend)

# Slater-ish parameters: (n_electrons, exponent zeta) per element, tuned so the
# isosurfaces look like the textbook picture rather than to be quantitative.
_ATOM = {
    "H": (1.0, 1.0),
    "O": (8.0, 2.25),
}


def water() -> Structure:
    r, half = 1.81, np.deg2rad(104.5) / 2.0
    pos = np.array([
        [0.0, 0.0, 0.0],                          # O
        [r * np.sin(half), 0.0, r * np.cos(half)],   # H
        [-r * np.sin(half), 0.0, r * np.cos(half)],  # H
    ])
    return Structure(symbols=["O", "H", "H"],
                     positions=pos,
                     numbers=np.array([8, 1, 1]))


class AnalyticBackend(ComputeBackend):
    def __init__(self, struct: Structure | None = None):
        self._struct = struct or water()

    # -- geometry ----------------------------------------------------------
    def structure(self) -> Structure:
        return self._struct

    # -- grid helper -------------------------------------------------------
    def _grid(self, n: int, pad: float):
        p = self._struct.positions
        lo = p.min(0) - pad
        hi = p.max(0) + pad
        spacing = (hi - lo) / (n - 1)
        xs = [lo[i] + spacing[i] * np.arange(n) for i in range(3)]
        X, Y, Z = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
        return X, Y, Z, tuple(lo), tuple(spacing)

    # -- total electron density -------------------------------------------
    def density(self, n: int = 64, pad: float = 5.0) -> ScalarField:
        X, Y, Z, origin, spacing = self._grid(n, pad)
        rho = np.zeros_like(X)
        for sym, R in zip(self._struct.symbols, self._struct.positions):
            ne, zeta = _ATOM[sym]
            d = np.sqrt((X - R[0])**2 + (Y - R[1])**2 + (Z - R[2])**2)
            # |1s|^2-like normalised cloud carrying `ne` electrons
            rho += ne * (zeta**3 / np.pi) * np.exp(-2.0 * zeta * d)
        return ScalarField("Electron density", rho, origin, spacing, signed=False)

    # -- a frontier orbital: O 2p_z lone pair -----------------------------
    def orbital(self, index: int = 0, n: int = 64, pad: float = 5.0) -> ScalarField:
        X, Y, Z, origin, spacing = self._grid(n, pad)
        Ro = self._struct.positions[0]
        x, y, z = X - Ro[0], Y - Ro[1], Z - Ro[2]
        r = np.sqrt(x*x + y*y + z*z)
        zeta = 1.6
        psi = z * np.exp(-zeta * r)                # 2p_z: + lobe above, - below
        psi /= np.abs(psi).max()
        return ScalarField(f"MO #{index} (O 2p_z lone pair)", psi,
                           origin, spacing, signed=True)

    # -- density gradient field -------------------------------------------
    def density_gradient(self, n: int = 22, pad: float = 4.0) -> VectorField:
        f = self.density(n=n, pad=pad)
        gx, gy, gz = np.gradient(f.values, *f.spacing)
        vec = np.stack([gx, gy, gz], axis=-1)
        return VectorField("grad(rho)", vec, f.origin, f.spacing)

    # -- streamed SCF telemetry -------------------------------------------
    def run_scf(self, n_iter: int = 22):
        rng = np.random.default_rng(7)
        E = -75.5
        E_final = -76.0241  # near the real HF water energy, for flavour
        comm, drho = 3.0, 1.0
        prevE = E
        for k in range(n_iter):
            # geometric-ish decay with a little DIIS jitter
            frac = 0.62 ** k
            E = E_final + (E - E_final) * 0.55 + (-75.5 - E_final) * frac * 0.45
            comm = comm * (0.55 + 0.05 * rng.standard_normal()) ** 1
            drho = drho * (0.6 + 0.04 * rng.standard_normal())
            comm = max(comm, 1e-7)
            drho = max(drho, 1e-7)
            yield SCFStep(iteration=k,
                          energy=E,
                          dE=abs(E - prevE) + 1e-12,
                          commutator=comm,
                          drho=drho)
            prevE = E
