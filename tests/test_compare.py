"""S4, one spec on several codes: the SAME Gaussian basis reaches CP2K converted from SOLID²'s file (never a
look-alike by name), the back-end strip enables exactly the codes that can run a spec, and the Codes panel lays
the legs' energies beside their resolved settings.

    .venv/bin/python -m unittest tests.test_compare
"""
import os
import pathlib
import unittest

from qviz import compare
from qviz.backends import BackendRegistry, Cp2kBackend, FakeBackend, Solid2Backend, default_registry
from qviz.backends import gaussian_basis
from qviz.backends.base import Result
from qviz.spec import Accuracy, molecule
from qviz.workspace import RunStatus, Workspace

ORACLES = gaussian_basis.bsd_dir().parent / "IntegrationTests/CP2K"
F2 = ([9, 9], [0, 0, -1.3343, 0, 0, 1.3343])


def f2(representation=None, **acc):
    return molecule("F2", *F2, theory="DFT", functional="LDA", core="pseudopotential",
                    accuracy=Accuracy(epsilon_Ha_per_atom=1e-7,
                                      representation=representation or {"basis": "valence_lowq_vb"}, **acc))


def cp2k_blocks(text: str) -> dict:
    """A CP2K basis file as {(element, name): [(l, ((alpha, coef), ...)), ...]}."""
    lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    out, i = {}, 0
    while i < len(lines):
        el, name = lines[i].split()[:2]
        nsets = int(lines[i + 1]); i += 2
        shells = []
        for _ in range(nsets):
            n, lmin, lmax, nexp, *_rest = map(int, lines[i].split()); i += 1
            shells.append((lmin, tuple(tuple(map(float, lines[i + k].split()[:2])) for k in range(nexp)))); i += nexp
        out[(el.upper(), name)] = shells
    return out


@unittest.skipUnless((gaussian_basis.bsd_dir() / "valence_lowq_vb.bsd").exists(), "qchem's BasisSetData not found (QCHEM_SRC)")
class OneBasisTwoFormats(unittest.TestCase):
    def test_the_converted_O_block_is_the_oracles_hand_written_one(self):
        ours = cp2k_blocks(gaussian_basis.to_cp2k("valence_lowq_vb", ["O"]))[("O", "SOLID2-VALENCE_LOWQ_VB")]
        theirs = cp2k_blocks((ORACLES / "VALENCE-LOWQ-BASIS").read_text().split("# --- VB")[1].split("\n", 1)[1]
                             .split("Mn VALENCE-LOWQ-VA")[0])[("O", "VALENCE-LOWQ-VB")]
        self.assertEqual(ours, theirs)

    def test_the_pp_valence_comes_from_the_files_provenance(self):
        self.assertEqual(gaussian_basis.pp_valence("valence_lowq_vb")["F"], 7)
        deck = Cp2kBackend().translate(f2()).files["deck"].decode()
        self.assertIn("POTENTIAL GTH-PADE-q7", deck)
        self.assertIn("BASIS_SET SOLID2-VALENCE_LOWQ_VB", deck)
        self.assertIn("BASIS_SET_FILE_NAME job.basis", deck)

    def test_d_shells_and_missing_elements_are_refused_with_the_reason(self):
        self.assertIn("spherical", gaussian_basis.why_not_cp2k("valence_lowq_vb", ["Mn", "O"]))
        self.assertIn("no block for H", gaussian_basis.why_not_cp2k("valence_lowq_vb", ["O", "H"]))
        v = Cp2kBackend().capabilities().can_run(molecule("MnO", [25, 8], [0, 0, 0, 0, 0, 3.0], theory="DFT",
                                                         functional="LDA", core="pseudopotential", multiplicity=6,
                                                         accuracy=Accuracy(representation={"basis": "valence_lowq_vb"})))
        self.assertFalse(v)
        self.assertTrue(any("spherical" in r for r in v.reasons), v.reasons)


class PerCodeBlocks(unittest.TestCase):
    def tuned(self):
        return f2(representation={"basis": "valence_lowq_vb", "cp2k": {"cutoff_Ry": 400},
                                  "solid2": {"mesh_radial": 120, "mesh_angular": 35}})

    def test_each_code_reads_its_own_block_and_ignores_the_others(self):
        s = self.tuned()
        self.assertTrue(all(c.check(s) for c in Cp2kBackend().capabilities().features))
        self.assertTrue(all(c.check(s) for c in Solid2Backend().capabilities().features))
        self.assertIn("CUTOFF 400", Cp2kBackend().translate(s).files["deck"].decode())
        n = Solid2Backend().translate(s).native
        self.assertEqual((n["mesh_radial"], n["mesh_angular"]), (120, 35))

    def test_an_unknown_shared_key_is_refused_by_both(self):
        s = f2(representation={"basis": "valence_lowq_vb", "pw_cutoff_Ry": 80})
        self.assertFalse(all(c.check(s) for c in Cp2kBackend().capabilities().features))
        self.assertFalse(all(c.check(s) for c in Solid2Backend().capabilities().features))

    def test_a_dft_run_never_uses_the_facades_12_direction_mesh(self):
        n = Solid2Backend().translate(f2()).native
        self.assertGreaterEqual((n["mesh_radial"], n["mesh_angular"]), (80, 29))
        hf = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.43, 1.1, 0, -1.43, 1.1])
        self.assertEqual(Solid2Backend().translate(hf).native["mesh_radial"], 0, "HF has no XC mesh")


class TheStrip(unittest.TestCase):
    def test_a_shared_spec_enables_both_codes_and_ticks_the_default(self):
        reg = default_registry()
        st = reg.strip(f2())
        self.assertTrue(st["CP2K"][0])
        if Solid2Backend().capabilities().installed:
            self.assertEqual({n for n, (_e, t, _h) in st.items() if t}, {"SOLID²"})
            self.assertTrue(reg.strip(f2(), ticked=["SOLID²", "CP2K"])["CP2K"][1], "a tick survives while able")

    def test_an_all_electron_spec_greys_cp2k_with_its_reason(self):
        able, ticked, hover = default_registry().strip(molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.43, 1.1, 0, -1.43, 1.1]))["CP2K"]
        self.assertFalse(able or ticked)
        self.assertIn("pseudopotential", hover)

    def test_a_synthetic_back_end_is_ticked_only_when_nothing_real_can_run(self):
        reg = BackendRegistry([Cp2kBackend(), FakeBackend()])
        self.assertTrue(reg.strip(molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.43, 1.1, 0, -1.43, 1.1]))["Fake"][1])
        if Cp2kBackend().capabilities().installed:
            self.assertFalse(reg.strip(f2())["Fake"][1])


class _Offset(FakeBackend):
    """A non-synthetic stand-in: Fake's numbers shifted by a known amount, to check the table's arithmetic."""
    name = "Offset"

    def capabilities(self):
        c = super().capabilities(); c.backend, c.synthetic = self.name, False
        return c

    def run(self, job, on_step=None) -> Result:
        r = super().run(job, on_step)
        return Result(**{**r.__dict__, "energy_Ha": r.energy_Ha + 2e-6, "fields": {}})


class _Plain(_Offset):
    name = "Plain"

    def run(self, job, on_step=None) -> Result:
        return FakeBackend.run(self, job, on_step)


class TheCodesPanel(unittest.TestCase):
    def ws(self):
        return Workspace("t", registry=BackendRegistry([_Plain(), _Offset(), FakeBackend()]))

    def test_one_spec_lands_on_every_real_code_and_the_table_lines_them_up(self):
        ws = self.ws()
        runs = compare.cross_code(ws, f2())
        self.assertEqual([r.backend_name for r in runs], ["Plain", "Offset"], "Fake is never a comparison leg")
        for r in runs:
            r.compute()
            self.assertIs(r.status, RunStatus.READY, r.error)
        heads, rows = compare.code_table(compare.legs(ws, runs[0]))
        t = dict(rows)
        self.assertEqual(heads, ["Plain", "Offset"])
        self.assertEqual(t["ΔE vs Plain (μHa)"], ["+0.0", "+2.0"])
        self.assertEqual(t["ΔE per atom (μHa)"], ["+0.0", "+1.0"])
        self.assertIn("model", t, "the resolved settings sit beside the numbers")

    def test_a_run_without_a_density_is_never_a_delta_rho_partner(self):
        ws = self.ws()
        a, b = compare.cross_code(ws, f2())
        a.compute(); b.compute()
        self.assertTrue(a.offers("density"))
        self.assertFalse(b.offers("density"))
        with self.assertRaisesRegex(ValueError, "density"):
            compare.difference_density(a, b, n=8)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.environ.get("QVIZ_LONG") and Solid2Backend().capabilities().installed
                     and Cp2kBackend().capabilities().installed, "set QVIZ_LONG=1 (needs SOLID² and CP2K; ~1 min)")
class F2OnBothCodes(unittest.TestCase):
    def test_one_spec_two_codes_through_the_workspace(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.environ["RUNDIR_ROOT"] = d
            try:
                ws = Workspace("t", registry=BackendRegistry([Solid2Backend(), Cp2kBackend(), FakeBackend()]))
                runs = compare.cross_code(ws, f2())
                for r in runs:
                    r.compute()
            finally:
                del os.environ["RUNDIR_ROOT"]
        self.assertEqual([r.backend_name for r in runs], ["SOLID²", "CP2K"])
        for r in runs:
            self.assertIs(r.status, RunStatus.READY, r.error)
        # each leg against its own converged value (2026-10-30): SOLID² mesh-converged from 80/29 up; CP2K 800 Ry in its
        # 22-bohr MT box (1200 Ry and a 40-bohr box move it < 40 μHa).  The 7.9 mHa BETWEEN them is an OPEN defect,
        # localized to SOLID²'s molecular PP route (the crystal route matches CP2K on NaF to 0.1 mHa) -- not pinned here.
        self.assertAlmostEqual(runs[0].energy, -48.19259, delta=1e-4)
        self.assertAlmostEqual(runs[1].energy, -48.20058, delta=1e-5)
        heads, rows = compare.code_table(runs)
        self.assertIn("XC mesh", dict(rows))
        self.assertIn("cutoff", dict(rows))
