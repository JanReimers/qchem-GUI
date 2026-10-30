"""
qviz.backends.fake -- a closed-form Backend for tests and demos (no compiled code needed).

It makes the whole UI testable without a qchem build.  The numbers are PHYSICALLY SHAPED,
not computed: the density is a sum of atom-centred Slater-like clouds carrying Z electrons
each, the "HOMO" is a 2p_z lobe on the first heavy atom, and the SCF trace decays like a
DIIS-accelerated HF.  It declares a small capability set -- molecules, no +U, no computed U --
so it also exercises the option-2.2 filter.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..capability import Capabilities, Capability
from ..data import ScalarField, SCFStep, VectorField
from ..spec import RunSpec
from .base import FieldRef, Job, Result, StepSink, data_structure

NAME = "Fake"


def _zeta(Z: int) -> float:
    """A Slater-ish exponent that grows with Z (tuned for the textbook look, not for numbers)."""
    return 1.0 if Z == 1 else 0.3 * Z ** 0.95


class FakeBackend:
    name = NAME

    def capabilities(self) -> Capabilities:
        return Capabilities(NAME, [
            Capability("structure.kind=molecule", lambda s: s.structure.kind == "molecule", "molecules only"),
            Capability("physics.hubbard=()", lambda s: not s.physics.hubbard, "no DFT+U"),
            Capability("task.kind=energy", lambda s: s.task.kind == "energy", "single-point energy only"),
        ])

    def translate(self, spec: RunSpec) -> Job:
        n_iter = int((spec.accuracy.scf or {}).get("max_iter", 22))
        return Job(NAME, spec, native={"n_iter": min(n_iter, 22)}, resolved={"model": "analytic Slater clouds"})

    def run(self, job: Job, on_step: Optional[StepSink] = None) -> Result:
        spec = job.spec
        st = data_structure(spec)
        ne = spec.num_electrons
        E_final = -0.5 * sum(int(z) ** 2.4 for z in st.numbers) / 2.0   # a rough total-energy scale
        steps = list(_trace(E_final, job.native["n_iter"]))
        for s in steps:
            if on_step:
                on_step(s)
        nocc = (ne + 1) // 2
        levels = [{"e": -0.5 * (nocc - i) ** 1.3, "occ": 2.0 if 2 * (i + 1) <= ne else 1.0, "spin": 0,
                   "label": "", "n": i + 1} for i in range(nocc)]
        levels += [{"e": 0.15 * (k + 1), "occ": 0.0, "spin": 0, "label": "", "n": nocc + k + 1} for k in range(3)]

        def density(n: int, pad: float) -> ScalarField:
            X, Y, Z, origin, spacing = _grid(st.positions, n, pad)
            rho = np.zeros_like(X)
            for z, R in zip(st.numbers, st.positions):
                zeta = _zeta(int(z))
                d = np.sqrt((X - R[0])**2 + (Y - R[1])**2 + (Z - R[2])**2)
                rho += int(z) * (zeta**3 / np.pi) * np.exp(-2.0 * zeta * d)
            return ScalarField("Electron density", rho, origin, spacing, signed=False)

        def gradient(n: int, pad: float) -> VectorField:
            f = density(n, pad)
            g = np.stack(np.gradient(f.values, *f.spacing), axis=-1)
            return VectorField("grad(rho)", g, f.origin, f.spacing)

        heavy = int(np.argmax(st.numbers))

        def orbital(index: int, n: int, pad: float) -> ScalarField:
            X, Y, Z, origin, spacing = _grid(st.positions, n, pad)
            R = st.positions[heavy]
            x, y, z = X - R[0], Y - R[1], Z - R[2]
            psi = [z, x, y][index % 3] * np.exp(-1.6 * np.sqrt(x*x + y*y + z*z))
            return ScalarField(f"MO #{index}", psi / np.abs(psi).max(), origin, spacing, signed=True)

        return Result(converged=True, energy_Ha=steps[-1].energy, structure=st, scf=steps, levels=levels,
                      fields={"density": FieldRef("Electron density", False, density),
                              "gradient": FieldRef("grad(rho)", False, gradient)},
                      orbital=orbital, resolved=job.resolved, provenance={"code": NAME})


def _grid(p: np.ndarray, n: int, pad: float):
    lo = p.min(0) - pad
    hi = p.max(0) + pad
    spacing = (hi - lo) / (n - 1)
    xs = [lo[i] + spacing[i] * np.arange(n) for i in range(3)]
    X, Y, Z = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
    return X, Y, Z, tuple(lo), tuple(spacing)


def _trace(E_final: float, n_iter: int):
    """A geometric-ish decay with a little DIIS jitter."""
    rng = np.random.default_rng(7)
    E0 = E_final * 0.95
    E, comm, drho, prevE = E0, 3.0, 1.0, E0
    for k in range(n_iter):
        E = E_final + (E - E_final) * 0.55 + (E0 - E_final) * 0.62 ** k * 0.45
        comm = max(comm * (0.55 + 0.05 * rng.standard_normal()), 1e-7)
        drho = max(drho * (0.6 + 0.04 * rng.standard_normal()), 1e-7)
        yield SCFStep(iteration=k + 1, energy=E, dE=abs(E - prevE) + 1e-12, commutator=comm, drho=drho)
        prevE = E
