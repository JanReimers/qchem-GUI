"""The CP2K back end: the translator reproduces qchem's BANKED oracle decks (IntegrationTests/CP2K/), the tailer and
parser read CP2K's output, the launch is always `mpirun`, and a real run reproduces a banked energy
(doc/Records/CP2Kresults.md): FCC Si, Γ, SIPP_SR, 80 Ry -> -7.11506 Ha.

    .venv/bin/python -m unittest tests.test_cp2k
"""
import os
import pathlib
import re
import tempfile
import unittest

from qviz.backends.cp2k import Cp2kBackend, Cp2kTail, parse_output
from qviz.spec import RunSpec

ORACLES = pathlib.Path(os.environ.get("QCHEM_SRC", pathlib.Path.home() / "Code/qchem-main")) / "IntegrationTests/CP2K"
CP2K = Cp2kBackend()


def si_oracle() -> RunSpec:
    """si_fcc_gpw.inp, as a spec: everything the banked deck pins is STATED for CP2K."""
    return RunSpec.from_dict({
        "label": "Si fcc Γ oracle",
        "structure": {"kind": "crystal", "lattice_bohr": [[0, 5.13, 5.13], [5.13, 0, 5.13], [5.13, 5.13, 0]],
                      "sites": [{"element": "Si", "frac": [0, 0, 0]}, {"element": "Si", "frac": [0.25, 0.25, 0.25]}],
                      "source": {"materials": "Si_fcc"}},
        "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential", "spin": {"polarized": False}},
        "accuracy": {"epsilon_Ha_per_atom": 1e-5, "scf": {"max_iter": 60},
                     "representation": {"cp2k": {"basis_file": str(ORACLES / "SIPP-SR-BASIS"),
                                                 "kinds": {"Si": {"basis": "SIPP-SR", "potential": "GTH-PADE-q4"}},
                                                 "cutoff_Ry": 80, "rel_cutoff_Ry": 60, "eps_default": 1e-12,
                                                 "added_mos": 4}}},
    }).validate()


def mno_oracle() -> RunSpec:
    """mno_afm2_gpw_va_plusU.inp, as a spec.  A μHa oracle, so ε 1e-6: the translator's own rule (EPS_SCF = ε/100) then
    writes the deck's EPS_SCF 1e-8 (re-banked there by lane H: at 1e-6 it had stopped 83 μHa low)."""
    va = {"basis": "VALENCE-LOWQ-VA", "potential": "GTH-PADE-q7"}
    return RunSpec.from_dict({
        "label": "MnO AFM-II +U oracle",
        "structure": {"kind": "crystal", "lattice_bohr": [[8.4, 4.2, 4.2], [4.2, 8.4, 4.2], [4.2, 4.2, 8.4]],
                      "sites": [{"element": "Mn", "frac": [0, 0, 0], "tag": "Mn1"},
                                {"element": "Mn", "frac": [0.5, 0.5, 0.5], "tag": "Mn2"},
                                {"element": "O", "frac": [0.25, 0.25, 0.25]}, {"element": "O", "frac": [0.75, 0.75, 0.75]}],
                      "source": {"materials": "MnO_AFM2"}},
        "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential",
                    "spin": {"polarized": True, "site_moments": [5, -5, 0, 0]},
                    "hubbard": [{"sites": ["Mn1", "Mn2"], "l": 2, "U_eV": 4.0, "projector": "lowdin"}],
                    "occupations": {"smearing": "fermi-dirac", "kT_Ha": 5e-3}},
        "accuracy": {"epsilon_Ha_per_atom": 1e-6, "scf": {"max_iter": 500},
                     "representation": {"cp2k": {"basis_file": str(ORACLES / "VALENCE-LOWQ-BASIS"),
                                                 "potential_file": "/home/janr/Code/cp2k/data/GTH_POTENTIALS",
                                                 "kinds": {"Mn1": va, "Mn2": va,
                                                           "O": {"basis": "VALENCE-LOWQ-VB", "potential": "GTH-PADE-q6"}},
                                                 "cutoff_Ry": 144, "rel_cutoff_Ry": 60,
                                                 "eps_default": 1e-14, "eps_pgf_orb": 1e-12, "added_mos": 20}}},
    }).validate()


def sections(deck: str) -> dict:
    """A CP2K input as {section path: [(KEYWORD, value), ...]} -- comments dropped, the oracle's relative paths ignored."""
    out, path = {}, []
    for raw in deck.splitlines():
        line = re.split(r"\s[#!]|^[#!]", raw)[0].strip()
        if not line:
            continue
        if line.upper().startswith("&END"):
            path.pop(); continue
        if line.startswith("&"):
            head = line[1:].split(None, 1)
            path.append(head[0].upper() + (" " + head[1] if len(head) > 1 else ""))
            out.setdefault("/".join(path), [])
            continue
        key, _, val = line.partition(" ")
        out.setdefault("/".join(path), []).append((key.upper(), " ".join(val.split())))
    return out


def same_numbers(a: str, b: str) -> bool:
    """Two keyword values agree token by token, numerically where both are numbers."""
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        try:
            if abs(float(x) - float(y)) > 1e-12 * max(1.0, abs(float(y))):
                return False
        except ValueError:
            if x.upper() != y.upper():
                return False
    return True


class TranslateMatchesTheBankedDecks(unittest.TestCase):
    #: keywords an oracle deck states that the translator deliberately does differently, with the reason
    ALLOWED = {
        "PROJECT": "named per run",
        "BASIS_SET_FILE_NAME": "an absolute path to the same file",
        "EPS_PGF_ORB": "always stated (CP2K's default once made a 3.5 Ha 'oracle')",
        "IGNORE_CONVERGENCE_FAILURE": "never: a capped SCF fails",
        "ALGORITHM": "CP2K's default diagonalization",
        "PRINT_LEVEL": "",
    }
    DROPPED_SECTIONS = ("DIAGONALIZATION", "PRINT", "SMEAR OFF")   # defaults the oracle spells out

    def compare(self, spec, oracle_file):
        ours = sections(CP2K.translate(spec).files["deck"].decode())
        theirs = sections((ORACLES / oracle_file).read_text())
        for path, kws in theirs.items():
            if any(seg.startswith(self.DROPPED_SECTIONS) for seg in path.split("/")):
                continue
            self.assertIn(path, ours, f"section {path} missing")
            got = dict(ours[path]) if not path.endswith("COORD") else None
            for k, v in kws:
                if k in self.ALLOWED:
                    continue
                if path.endswith("COORD"):
                    self.assertTrue(any(kk == k and same_numbers(vv, v) for kk, vv in ours[path]), f"{path}: {k} {v}")
                else:
                    self.assertIn(k, got, f"{path}/{k} missing")
                    self.assertTrue(same_numbers(got[k], v), f"{path}/{k}: ours {got[k]!r}, oracle {v!r}")
        return ours

    @unittest.skipUnless((ORACLES / "si_fcc_gpw.inp").exists(), "qchem's oracle decks not found (QCHEM_SRC)")
    def test_si_gamma(self):
        ours = self.compare(si_oracle(), "si_fcc_gpw.inp")
        self.assertNotIn("UKS", dict(ours["FORCE_EVAL/DFT"]), "an unpolarized spec is RKS")

    @unittest.skipUnless((ORACLES / "mno_afm2_gpw_va_plusU.inp").exists(), "qchem's oracle decks not found (QCHEM_SRC)")
    def test_mno_afm2_plus_u(self):
        ours = self.compare(mno_oracle(), "mno_afm2_gpw_va_plusU.inp")
        self.assertIn(("NEL", "-5"), ours["FORCE_EVAL/SUBSYS/KIND Mn2/BS/ALPHA"], "the -m sublattice flips")

    def test_no_basis_stated_is_refused_never_defaulted(self):
        """The user, 2026-11-11: a CP2K-only fallback (DZVP-MOLOPT-SR-GTH) let a picked crystal run on CP2K while SOLID² refused
        it, on a basis the spec never named.  The basis is chosen once, in the spec; CP2K says so like SOLID² does."""
        from qviz.spec import RunSpec
        nio = RunSpec.from_dict({"label": "NiO", "structure": {"kind": "crystal",
                                 "lattice_bohr": [[0, 3.94, 3.94], [3.94, 0, 3.94], [3.94, 3.94, 0]],
                                 "sites": [{"element": "Ni", "frac": [0, 0, 0]}, {"element": "O", "frac": [0.5, 0.5, 0.5]}]},
                                 "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential"},
                                 "accuracy": {"kmesh": [1, 1, 1]}}).validate()
        why = "; ".join(CP2K.capabilities().can_run(nio).reasons)
        if not CP2K.capabilities().installed:
            self.skipTest(why)
        self.assertIn("choose a basis on the Basis tab", why)
        self.assertIn("Ni, O", why)
        stated = RunSpec.from_dict({**nio.to_dict(), "accuracy": {"kmesh": [1, 1, 1], "representation": {"cp2k": {
            "kinds": {"Ni": {"basis": "DZVP-MOLOPT-SR-GTH"}, "O": {"basis": "DZVP-MOLOPT-SR-GTH"}}}}}}).validate()
        self.assertNotIn("choose a basis", "; ".join(CP2K.capabilities().can_run(stated).reasons), "a CP2K kinds basis is a stated one")

    def test_the_launch_is_always_mpirun(self):
        job = CP2K.translate(si_oracle())
        argv = CP2K.runner.argv(job.command + ("-i", "x.inp"))
        self.assertEqual(argv[:3], ["mpirun", "-np", "1"])
        self.assertTrue(argv[3].endswith("cp2k.psmp"))


OUTPUT = """ CP2K| version string:                                       CP2K version 2026.1
 SCF WAVEFUNCTION OPTIMIZATION

  Step     Update method      Time    Convergence         Total energy    Change
  ------------------------------------------------------------------------------
     1 P_Mix/Diag. 0.40E+00    0.2     0.75000000        -7.0500000000 -7.05E+00
     2 P_Mix/Diag. 0.40E+00    0.3     0.01000000        -7.1140000000 -6.40E-02
     3 P_Mix/Diag. 0.40E+00    0.4     0.00000009        -7.1150580000 -1.06E-03

  *** SCF run converged in     3 steps ***

 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]               -7.115058123456789
"""


class ParseTheOutput(unittest.TestCase):
    def test_the_tail_streams_the_table_and_the_parser_reads_the_energy(self):
        seen = []
        t = Cp2kTail(seen.append)
        for line in OUTPUT.splitlines(keepends=True):
            t.feed(line)
        self.assertEqual([s.iteration for s in seen], [1, 2, 3])
        self.assertAlmostEqual(seen[-1].energy, -7.115058)
        self.assertEqual(seen[1].titles, ("Update method", "Time", "Convergence", "Total energy", "Change"))   # CP2K's own header
        self.assertAlmostEqual(seen[1].value("Change"), -6.4e-2)
        self.assertEqual([c.scale for c in seen[1].cells], ["linear", "linear", "log", "linear", "log"])
        self.assertEqual(seen[1].cells[0].text, "P_Mix/Diag. 0.40E+00")
        p = parse_output(OUTPUT)
        self.assertTrue(p["converged"])
        self.assertAlmostEqual(p["energy"], -7.115058123456789)
        self.assertEqual(p["version"], "CP2K version 2026.1")

    def test_a_capped_scf_is_not_converged(self):
        self.assertFalse(parse_output(OUTPUT.replace("SCF run converged in     3 steps", "WARNING :: SCF run NOT converged"))["converged"])


@unittest.skipUnless(CP2K.capabilities().installed and (ORACLES / "SIPP-SR-BASIS").exists(), "cp2k / oracle files absent")
class RunTheOracle(unittest.TestCase):
    def test_si_gamma_reproduces_the_banked_energy(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["RUNDIR_ROOT"] = d
            try:
                steps = []
                r = CP2K.run(CP2K.translate(si_oracle()), steps.append)
            finally:
                del os.environ["RUNDIR_ROOT"]
            out = pathlib.Path(r.provenance["output"])
            self.assertTrue(str(out).startswith(d) and out.parent.name == "Si_fcc", out)
            self.assertTrue((out.parent / out.name.replace(".out", ".record.json")).exists(), "record written")
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.energy_Ha, -7.11506, delta=2e-5)   # doc/Records/CP2Kresults.md, si_fcc_gpw.inp
        self.assertGreater(len(steps), 2)
        self.assertEqual(steps, r.scf)


if __name__ == "__main__":
    unittest.main()
