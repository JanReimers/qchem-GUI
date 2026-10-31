"""The ABINIT back end: the input it writes, the launch (always `mpirun`, and no `--force-mpirun`: abinit 10.8 rejects it), the tailer and parser on a
banked ABINIT run, and a real run gated CROSS-CODE: β-MnO₂ at Γ on the same GTH pseudopotentials (qchem's database, as pspcod 10), cutoff and FFT grid as the banked
QE run (~/Code/qe-runs/MnO2/mno2b.scf.out, -186.93546944 Ry; its U = 1e-8 eV is the hp.x reference state, energetically
nil, and ABINIT's +U is PAW-only, so the ABINIT spec states none).

    .venv/bin/python -m unittest tests.test_abinit
    QVIZ_LONG=1 ~/Code/qchem6-runs/heavy .venv/bin/python -m unittest tests.test_abinit   # + the real run
"""
import os
import pathlib
import tempfile
import unittest

from qviz.backends.abinit import AbinitBackend, AbinitTail, parse_output
from qviz.spec import RunSpec
from qviz.backends import gth
from tests.test_qe import mno2b

AB = AbinitBackend()
NIO = pathlib.Path.home() / "Code/abinit-runs/NiO"
QE_BANKED_HA = -186.93546944 / 2


def mno2b_abinit(**ab) -> RunSpec:
    d = mno2b().to_dict()
    d["physics"]["hubbard"] = []
    d["accuracy"]["representation"] = {"abinit": {"ecut_Ha": 150.0, "ngfft": [96, 96, 60],
                                                  "nband": 30, "toldfe": 1e-10} | ab}
    return RunSpec.from_dict(d).validate()


@unittest.skipUnless(gth.database().exists(), "qchem's GTH database is absent (QCHEM_SRC)")
class TheInput(unittest.TestCase):
    def test_the_pseudopotential_is_the_database_entry_rewrapped(self):
        text = AB.translate(mno2b_abinit()).files["Mn.gth-q7.psp10"].decode().splitlines()
        self.assertEqual(text[1].split()[:2], ["25", "7"])
        self.assertEqual(text[2].split()[:4], ["10", "-1007", "2", "0"])
        self.assertEqual(text[5].split(), ["0.48124608", "3", "2.79903057", "-0.96286281", "0.62594958"])   # Mn GTH-PADE-q7 s-channel
        self.assertEqual(text[10].split(), ["0.0", "0.0"], "a zero k triangle for l > 0: no spin-orbit")

    def test_the_same_physics_as_the_qe_deck(self):
        deck = AB.translate(mno2b_abinit()).files["deck"].decode()
        for line in ("ixc -1007", "ecut 150.0", "ngfft 96 96 60", "occopt 7", "tsmear 0.001", "nsppol 2", "nband 30",
                     "ngkpt 1 1 1", "shiftk 0 0 0", 'pseudos "Mn.gth-q7.psp10, O.gth-q6.psp10"', 'pp_dirpath "."', "typat 1 1 2 2 2 2"):
            self.assertIn(line, deck)
        self.assertIn("spinat\n   0 0 3\n   0 0 -3\n", deck)

    def test_an_unstated_grid_is_pw_x_s_minimal_grid_scaled_and_stated(self):
        d = mno2b_abinit().to_dict()
        d["accuracy"]["representation"]["abinit"].pop("ngfft")
        d["accuracy"]["epsilon_Ha_per_atom"] = 1e-3                 # grid_factor 1: the minimal grid
        self.assertIn("ngfft 96 96 60", AB.translate(RunSpec.from_dict(d)).files["deck"].decode())

    def test_plus_u_is_refused(self):
        v = AB.capabilities().can_run(mno2b())
        if AB.capabilities().installed:
            self.assertFalse(v.ok)
            self.assertIn("PAW", " ".join(v.reasons))

    def test_the_launch_is_always_mpirun(self):
        argv = AB.runner.argv(AB.translate(mno2b_abinit()).command + ("x.abi",))
        self.assertEqual(argv[:3], ["mpirun", "-np", "1"])
        self.assertTrue(argv[3].endswith("abinit"))
        self.assertNotIn("--force-mpirun", argv, "the test-suite driver's flag: abinit itself exits on it")


@unittest.skipUnless((NIO / "nio_ucrpa_op.abo").exists(), "the banked ABINIT NiO run is absent")
class ParseABankedRun(unittest.TestCase):
    def test_tail_and_parser(self):
        seen = []
        t = AbinitTail(seen.append)
        for line in (NIO / "run_op6_serial.log").read_text().splitlines(keepends=True):
            t.feed(line)
        self.assertAlmostEqual(seen[0].energy, -367.74355891699)
        p = parse_output((NIO / "nio_ucrpa_op.abo").read_text())
        self.assertAlmostEqual(p["energy"], -363.71080887, places=8)
        self.assertTrue(p["converged"])
        self.assertEqual(p["version"], "10.8.3")

    def test_each_stopping_criterion_s_converged_line(self):
        for line in (" At SCF step   31       nres2   =  4.16E-14 < tolvrs=  1.00E-13 =>converged.",
                     " At SCF step   38, etot is converged : "):          # toldfe prints no arrow
            self.assertTrue(parse_output(line + "\n")["converged"], line)
        self.assertFalse(parse_output(" At SCF step   38, etot is converged : \n  nstep=  38 was not enough SCF cycles to converge;\n")["converged"])


@unittest.skipUnless(os.environ.get("QVIZ_LONG") and AB.capabilities().installed and gth.database().exists(),
                     "QVIZ_LONG=1, abinit and qchem's GTH database needed (minutes; take a heavy slot)")
class RunCrossCode(unittest.TestCase):
    def test_mno2b_agrees_with_the_banked_qe_energy(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["RUNDIR_ROOT"] = d
            try:
                steps = []
                r = AB.run(AB.translate(mno2b_abinit()), steps.append)
            finally:
                del os.environ["RUNDIR_ROOT"]
        self.assertTrue(r.converged)
        print(f"\nABINIT {r.energy_Ha:.10f} Ha vs QE {QE_BANKED_HA:.10f} Ha: {1e6 * (r.energy_Ha - QE_BANKED_HA):+.1f} μHa")
        self.assertAlmostEqual(r.energy_Ha, QE_BANKED_HA, delta=1e-4)
        self.assertEqual(steps, r.scf)


if __name__ == "__main__":
    unittest.main()
