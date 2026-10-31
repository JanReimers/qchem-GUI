"""The Quantum ESPRESSO back end: the translator reproduces qchem's BANKED pw.x oracle deck (IntegrationTests/QE/
mno2b.scf.in, β-MnO₂ at Γ), the grid rule reproduces pw.x's own minimal grid, the tailer and parser read the banked
output, the launch is always `mpirun`, and a real run reproduces the banked energy -186.93546944 Ry
(~/Code/qe-runs/MnO2/mno2b.scf.out) when run at the banked grid.

    .venv/bin/python -m unittest tests.test_qe
    QVIZ_LONG=1 ~/Code/qchem6-runs/heavy .venv/bin/python -m unittest tests.test_qe   # + the real run (~2 min)
"""
import os
import pathlib
import re
import tempfile
import unittest

from qviz.backends.qe import QeBackend, QeTail, fft_grid, parse_output
from qviz.spec import RunSpec

ORACLES = pathlib.Path(os.environ.get("QCHEM_SRC", pathlib.Path.home() / "Code/qchem-main")) / "IntegrationTests/QE"
UPF_DIR = pathlib.Path.home() / "Code/qchem6/IntegrationTests/QE"          # where the banked run's UPFs were written
BANKED_OUT = pathlib.Path.home() / "Code/qe-runs/MnO2/mno2b.scf.out"
QE = QeBackend()
A, C = 8.322543, 8.322543 * 0.653141


def mno2b(**qe) -> RunSpec:
    """mno2b.scf.in as a spec: rutile β-MnO₂, AFM Mn1/Mn2, U = 1e-8 eV (the hp.x reference state), Γ."""
    x = 0.3046
    return RunSpec.from_dict({
        "label": "beta-MnO2 Gamma",
        "structure": {"kind": "crystal", "lattice_bohr": [[A, 0, 0], [0, A, 0], [0, 0, C]],
                      "sites": [{"element": "Mn", "frac": [0, 0, 0], "tag": "Mn1"}, {"element": "Mn", "frac": [0.5, 0.5, 0.5], "tag": "Mn2"},
                                {"element": "O", "frac": [x, x, 0]}, {"element": "O", "frac": [0.5 + x, 0.5 - x, 0.5]},
                                {"element": "O", "frac": [0.5 - x, 0.5 + x, 0.5]}, {"element": "O", "frac": [1 - x, 1 - x, 0]}],
                      "source": {"materials": "MnO2"}},
        "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential",
                    "spin": {"polarized": True, "site_moments": [3, -3, 0, 0, 0, 0]},
                    "hubbard": [{"sites": ["Mn1", "Mn2"], "l": 2, "U_eV": 1e-8, "projector": "ortho-atomic"}],
                    "occupations": {"smearing": "gaussian", "kT_Ha": 0.001}},
        "accuracy": {"kmesh": [1, 1, 1], "representation": {"qe": {
            "pseudo_dir": str(UPF_DIR), "ecutwfc_Ry": 300.0, "ecutrho_Ry": 1200.0, "nr": [96, 96, 60],
            "starting_magnetization": {"Mn1": 0.8, "Mn2": -0.8}, "nbnd": 30, "conv_thr": 1e-10, "mixing_beta": 0.3} | qe}},
    }).validate()


def namelists(deck: str) -> dict:
    """The &namelist assignments of a pw.x input as {KEY: value}, numbers compared as numbers."""
    out = {}
    for m in re.finditer(r"(\w+(?:\(\d+\))?)\s*=\s*('[^']*'|[^,\s]+)", deck.split("CELL_PARAMETERS")[0].split("ATOMIC_SPECIES")[0]):
        k, v = m.group(1).lower(), m.group(2).strip("'").lower().replace("d", "e") if m.group(2)[0] != "'" else m.group(2).strip("'").lower()
        try:
            v = float(v)
        except ValueError:
            pass
        out[k] = v
    return out


def cards(deck: str, name: str) -> list[list[str]]:
    lines = deck.splitlines()
    i = next(n for n, l in enumerate(lines) if l.strip().upper().startswith(name))
    rows = []
    for l in lines[i + 1:]:
        t = l.split()
        if not t or t[0].isupper() and t[0] in ("ATOMIC_SPECIES", "ATOMIC_POSITIONS", "K_POINTS", "HUBBARD", "CELL_PARAMETERS"):
            break
        rows.append(t)
    return rows


@unittest.skipUnless((ORACLES / "mno2b.scf.in").exists() and (UPF_DIR / "Mn.pz-gth-q7.UPF").exists(), "qchem's QE decks / UPFs absent")
class TranslateMatchesTheBankedDeck(unittest.TestCase):
    #: what the banked deck states that the translator deliberately does differently, with the reason
    ALLOWED = {"ibrav": "the cell as vectors (ibrav 0) -- the same cell", "celldm(1)": "", "celldm(3)": "",
               "prefix": "named per run", "outdir": "", "pseudo_dir": "the same directory, with a trailing /"}

    def test_mno2b(self):
        ours, theirs = QE.translate(mno2b()).files["deck"].decode(), (ORACLES / "mno2b.scf.in").read_text()
        a, b = namelists(ours), namelists(theirs)
        for k, v in b.items():
            if k in self.ALLOWED:
                continue
            self.assertIn(k, a, f"{k} missing")
            self.assertEqual(a[k], v, f"{k}: ours {a[k]!r}, oracle {v!r}")
        self.assertEqual((a["nr1"], a["nr2"], a["nr3"]), (96, 96, 60), "the grid is ALWAYS stated")
        self.assertEqual([r[0] + r[2] for r in cards(ours, "ATOMIC_SPECIES")], [r[0] + r[2] for r in cards(theirs, "ATOMIC_SPECIES")])
        for o, t in zip(cards(ours, "ATOMIC_POSITIONS"), cards(theirs, "ATOMIC_POSITIONS"), strict=True):
            self.assertEqual(o[0], t[0])
            self.assertTrue(all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(o[1:], t[1:])), (o, t))
        self.assertEqual(cards(ours, "K_POINTS"), cards(theirs, "K_POINTS"))
        self.assertIn("HUBBARD {ortho-atomic}", ours)
        self.assertEqual([[r[0], r[1], float(r[2])] for r in cards(ours, "HUBBARD")],
                         [[r[0], r[1], float(r[2].replace("d", "e"))] for r in cards(theirs, "HUBBARD")])

    def test_the_grid_rule_is_pw_x_s_own_minimal_grid(self):
        self.assertEqual(fft_grid(mno2b().structure.lattice_bohr, 1200.0), [96, 96, 60])   # the banked run's FFT dimensions

    def test_an_unstated_grid_is_still_stated_in_the_deck(self):
        deck = QE.translate(RunSpec.from_dict(_without_nr(mno2b()))).files["deck"].decode()
        self.assertRegex(deck, r"nr1 = \d+")

    def test_the_launch_is_always_mpirun(self):
        argv = QE.runner.argv(QE.translate(mno2b()).command + ("-in", "x.in"))
        self.assertEqual(argv[:3], ["mpirun", "-np", "1"])
        self.assertTrue(argv[3].endswith("pw.x"))


def _without_nr(spec: RunSpec) -> dict:
    d = spec.to_dict()
    d["accuracy"]["representation"]["qe"].pop("nr")
    return d


class Refusals(unittest.TestCase):
    def test_what_qe_cannot_run_here(self):
        if not (UPF_DIR / "Mn.pz-gth-q7.UPF").exists():
            self.skipTest("UPFs absent")
        base = mno2b().to_dict()
        cases = {"Löwdin projector": ({"physics": base["physics"] | {"hubbard": [base["physics"]["hubbard"][0] | {"projector": "lowdin"}]}}, "projector"),
                 "a Gaussian basis": ({"accuracy": base["accuracy"] | {"representation": {"basis": "valence_lowq_va"}}}, "Gaussian"),
                 "no k-mesh": ({"accuracy": base["accuracy"] | {"kmesh": None}}, "kmesh"),
                 "a missing UPF": ({"accuracy": base["accuracy"] | {"representation": {"qe": {"pseudo_dir": "/nonexistent"}}}}, "gth2upf")}
        for what, (patch, word) in cases.items():
            with self.subTest(what):
                v = QE.capabilities().can_run(RunSpec.from_dict(base | patch))
                if not QE.capabilities().installed:
                    self.skipTest("pw.x absent")
                self.assertFalse(v.ok)
                self.assertIn(word, " ".join(v.reasons))


@unittest.skipUnless(BANKED_OUT.exists(), "the banked β-MnO₂ output is absent")
class ParseTheBankedOutput(unittest.TestCase):
    def test_tail_and_parser(self):
        text = BANKED_OUT.read_text()
        seen = []
        t = QeTail(seen.append)
        for line in text.splitlines(keepends=True):
            t.feed(line)
        self.assertEqual(len(seen), 25, "converged in 25 iterations")
        p = parse_output(text)
        self.assertTrue(p["converged"])
        self.assertAlmostEqual(p["energy"], -186.93546944 / 2, places=9)
        self.assertAlmostEqual(seen[-1].energy, p["energy"], places=6)
        self.assertEqual(seen[-1].titles, ("total energy", "estimated scf accuracy", "total magnetization", "absolute magnetization"))
        self.assertAlmostEqual(seen[0].value("estimated scf accuracy"), 2.86090945)       # pw.x's own number, in Ry
        self.assertAlmostEqual(seen[0].value("absolute magnetization"), 7.20)
        self.assertFalse(parse_output(text.replace("convergence has been achieved", "convergence NOT achieved"))["converged"])


@unittest.skipUnless(os.environ.get("QVIZ_LONG") and QE.capabilities().installed and (UPF_DIR / "Mn.pz-gth-q7.UPF").exists(),
                     "QVIZ_LONG=1, pw.x and the UPFs needed (~2 min; take a heavy slot)")
class RunTheOracle(unittest.TestCase):
    def test_mno2b_reproduces_the_banked_energy(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["RUNDIR_ROOT"] = d
            try:
                steps = []
                r = QE.run(QE.translate(mno2b()), steps.append)
            finally:
                del os.environ["RUNDIR_ROOT"]
            out = pathlib.Path(r.provenance["output"])
            self.assertEqual(out.parent.name, "MnO2")
            self.assertEqual(list((out.parent / "tmp").glob("*")), [], "QE scratch removed")
        self.assertTrue(r.converged)
        self.assertAlmostEqual(r.energy_Ha, -186.93546944 / 2, delta=1e-7)
        self.assertEqual(steps, r.scf)


if __name__ == "__main__":
    unittest.main()
