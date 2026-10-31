"""The RunSpec -> qchem deck translator (qviz.backends.solid2_deck; doc/RunSpecPlan.md §9 route (b)).

GOLDEN DECKS: a spec stating what qchem's banked bench decks state translates to exactly that deck; the decks it writes are
accepted by `rundeck --check`, and one runs end to end.  THE BRIDGE: the same deck through `qchem_py.SolidRun` (deck::Build)
lands on rundeck's energy to the last digit, and a refusal / an unconverged SCF comes back in the library's words.

    QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest tests.test_solid2_deck
"""
import json
import locale
import os
import pathlib
import subprocess
import tempfile
import unittest

from qviz.backends import solid2_deck as D
from qviz.spec import RunSpec

BENCH = D.qchem_src() / "decks/bench"
RUNDECK = pathlib.Path(os.environ.get("QCHEM_BUILD", pathlib.Path.home() / "Code/qchem-lane-g/build/Release")) / "CLIapps/rundeck"


def fcc(a: float):
    h = a / 2
    return [[0, h, h], [h, 0, h], [h, h, 0]]


def crystal(label, lattice, sites, materials=None, physics=None, accuracy=None, task=None) -> RunSpec:
    d = {"label": label,
         "structure": {"kind": "crystal", "lattice_bohr": lattice, "sites": sites,
                       **({"source": {"materials": materials}} if materials else {})},
         "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential", "spin": {"polarized": False}} | (physics or {}),
         "accuracy": {"kmesh": [1, 1, 1]} | (accuracy or {})}
    if task:
        d["task"] = task
    return RunSpec.from_dict(d).validate()


def bench(name: str) -> dict:
    return {k: v for k, v in json.loads((BENCH / name).read_text()).items() if not k.startswith("_")}


def from_bench(name: str, lattice, sites) -> RunSpec:
    """The bench deck as a spec: the geometry, k-mesh and basis in the shared spec; the rest of the recipe in the solid2 block."""
    b = bench(name)
    recipe = {k: v for k, v in b.items() if k not in ("structure", "kmesh", "basis")}
    return crystal(name, lattice, sites, materials=b["structure"],
                   accuracy={"kmesh": b["kmesh"], "representation": {"basis": b["basis"]["data"].lower(), "solid2": {"deck": recipe}}})


SI = (fcc(10.26), [{"element": "Si", "frac": [0, 0, 0]}, {"element": "Si", "frac": [0.25, 0.25, 0.25]}])
NAF = (fcc(8.73), [{"element": "Na", "frac": [0, 0, 0]}, {"element": "F", "frac": [0.5, 0.5, 0.5]}])


def mno_plus_u() -> RunSpec:
    return crystal("MnO AFM-II +U", [[8.4, 4.2, 4.2], [4.2, 8.4, 4.2], [4.2, 4.2, 8.4]],
                   [{"element": "Mn", "frac": [0, 0, 0], "tag": "Mn1"}, {"element": "Mn", "frac": [0.5, 0.5, 0.5], "tag": "Mn2"},
                    {"element": "O", "frac": [0.25, 0.25, 0.25]}, {"element": "O", "frac": [0.75, 0.75, 0.75]}],
                   materials="MnO_AFM2",
                   physics={"spin": {"polarized": True, "site_moments": [5, -5, 0, 0]},
                            "hubbard": [{"sites": ["Mn1", "Mn2"], "l": 2, "U_eV": 4.0, "projector": "lowdin"}],
                            "occupations": {"smearing": "fermi-dirac", "kT_Ha": 5e-3}},
                   accuracy={"representation": {"basis": "valence_lowq_va"}, "scf": {"max_iter": 200}})


@unittest.skipUnless((BENCH / "Si_Gamma_parity.json").exists(), "qchem's decks not found (QCHEM_SRC)")
class GoldenDecks(unittest.TestCase):
    def test_si_gamma_parity(self):
        self.assertEqual(D.to_deck(from_bench("Si_Gamma_parity.json", *SI)), bench("Si_Gamma_parity.json"))

    def test_naf_gamma_parity(self):
        self.assertEqual(D.to_deck(from_bench("NaF_SR2_Gamma_parity.json", *NAF)), bench("NaF_SR2_Gamma_parity.json"))

    def test_mno_plus_u_states_the_ordering_and_the_manifolds(self):
        d = D.to_deck(mno_plus_u())
        self.assertEqual(d["structure"], "MnO_AFM2")
        self.assertEqual(d["solid"]["multiplicity"], 1, "AFM: zero net moment -> the two-channel singlet")
        self.assertEqual(d["solid"]["siteSpins"], [1, -1, 0, 0])
        self.assertEqual(d["solid"]["hubbard"], [{"site": 0, "l": 2, "U_eV": 4.0, "projector": "siteShells"},
                                                 {"site": 1, "l": 2, "U_eV": 4.0, "projector": "siteShells"}])
        self.assertEqual(d["scf"], {"smearingkT": 5e-3, "NMaxIter": 200})

    def test_a_geometry_that_is_not_the_named_entry_goes_inline(self):
        s = crystal("Si strained", fcc(10.5), SI[1], materials="Si_diamond", accuracy={"representation": {"basis": "sipp_sr"}})
        st = D.to_deck(s)["structure"]
        self.assertIsInstance(st, dict, "the materials name is provenance; the geometry decides")
        self.assertEqual([a["el"] for a in st["atoms"]], ["Si", "Si"])

    def test_compute_u_becomes_a_post_scf_action(self):
        s = mno_plus_u()
        lr = RunSpec.from_dict(s.to_dict() | {"task": {"kind": "compute_U", "compute_U": {"method": "linear-response", "sites": ["Mn1", "Mn2"]}}})
        self.assertEqual(D.to_deck(lr)["postSCF"], [{"hubbardLinearResponse": {}}])
        loop = RunSpec.from_dict(s.to_dict() | {"task": {"kind": "compute_U", "compute_U": {"method": "ACBN0", "self_consistent": True}}})
        self.assertEqual(D.to_deck(loop)["postSCF"], [{"hubbardLoop": {}}])


class Refusals(unittest.TestCase):
    def test_what_the_deck_cannot_say(self):
        base = crystal("Si", *SI, materials="Si_diamond", accuracy={"representation": {"basis": "sipp_sr"}}).to_dict()
        cases = {"kmesh": ({"accuracy": base["accuracy"] | {"kmesh": None}}, "kmesh"),
                 "basis": ({"accuracy": base["accuracy"] | {"representation": None}}, "basis"),
                 "all-electron basis": ({"accuracy": base["accuracy"] | {"representation": {"basis": "dzvp"}}}, "valence"),
                 "smearing": ({"physics": base["physics"] | {"occupations": {"smearing": "gaussian", "kT_Ha": 1e-3}}}, "Fermi-Dirac"),
                 "molecular mesh": ({"accuracy": base["accuracy"] | {"representation": {"basis": "sipp_sr", "solid2": {"mesh_radial": 80}}}}, "solid2")}
        for what, (patch, word) in cases.items():
            with self.subTest(what):
                self.assertIn(word, D.why_not(RunSpec.from_dict(base | patch)) or "")


@unittest.skipUnless(RUNDECK.exists() and (BENCH / "Si_Gamma_parity.json").exists(), "rundeck / qchem's decks not found")
class RundeckReadsThem(unittest.TestCase):
    def rundeck(self, deck: dict, *args, timeout=120):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "gui.json"
            p.write_text(json.dumps(deck))
            r = subprocess.run([str(RUNDECK), str(p), "--out", d, *args], capture_output=True, text=True, timeout=timeout,
                               env=os.environ | {"QCHEM_OPENMP_THREADS": "1"})
            revs = sorted(pathlib.Path(d).glob("*.r*.json"))
            return r, (json.loads(revs[-1].read_text()) if revs else None)

    def test_every_golden_deck_vets(self):
        for spec in (from_bench("Si_Gamma_parity.json", *SI), from_bench("NaF_SR2_Gamma_parity.json", *NAF), mno_plus_u()):
            with self.subTest(spec.label):
                r, _ = self.rundeck(D.to_deck(spec), "--check")
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_si_gamma_runs_and_records_the_deck_it_was_given(self):
        deck = D.to_deck(from_bench("Si_Gamma_parity.json", *SI))
        r, rev = self.rundeck(deck)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("[rundeck] CONVERGED", r.stdout)

        def within(ours, record, path="run"):          # every value the GUI stated is the value the record says ran
            if isinstance(ours, dict):
                for k, v in ours.items():
                    within(v, record[k], f"{path}.{k}")
            else:
                self.assertEqual(ours, record, path)
        within(deck, rev["run"])
        energies = [float(l.split()[3]) for l in r.stdout.splitlines() if l.startswith("[rundeck] Etot =")]
        self.assertEqual(len(energies), 1, r.stdout[-2000:])
        self.assertAlmostEqual(energies[0], -7.11506, delta=2e-5)    # CP2K's si_fcc_gpw.inp (doc/Records/CP2Kresults.md); ran -7.1150674


def si_inline(**scf) -> RunSpec:
    """Si as a cell the user built (no materials.json name -> the inline structure), the template deck's recipe."""
    return crystal("Si inline", *SI, accuracy={"representation": {"basis": "sipp_sr", "solid2": {"deck": {"solid": {"densityEcut": 20.0}}}},
                                               **({"scf": scf} if scf else {})})


def solid2():
    from qviz.backends.solid2 import Solid2Backend
    return Solid2Backend()


@unittest.skipUnless(solid2()._qc is not None and RUNDECK.exists(), "qchem_py / rundeck not built (QCHEM_BUILD)")
class TheBridgeRunsThem(unittest.TestCase):
    def test_the_inline_cell_lands_on_rundeck_s_energy(self):
        be, spec = solid2(), si_inline()
        self.assertTrue(be.capabilities().can_run(spec))
        job = be.translate(spec)
        self.assertIsInstance(json.loads(job.native["deck"])["structure"], dict, "a built cell goes inline")
        rows, host = [], locale.setlocale(locale.LC_ALL)
        res = be.run(job, rows.append)
        self.assertEqual(locale.setlocale(locale.LC_ALL), host, "the lib's console tables reset LC_ALL to C; the bridge puts it back")
        self.assertTrue(res.converged, res.failure)
        self.assertIsNone(res.failure)
        self.assertEqual(len(rows), res.provenance["iterations"])
        self.assertEqual(rows[0].titles[:2], ("Etotal", "[F,D]"), "the periodic console's own columns")
        self.assertEqual(res.provenance["deck"]["solid"]["Nelec"], 8, "the resolved deck derives Nelec")
        r, _ = RundeckReadsThem.rundeck(self, json.loads(job.native["deck"]))
        etot = [float(l.split()[3]) for l in r.stdout.splitlines() if l.startswith("[rundeck] Etot =")]
        self.assertEqual(len(etot), 1, r.stdout[-2000:])
        self.assertAlmostEqual(res.energy_Ha, etot[0], delta=1e-9, msg="one deck, one path: the bridge IS rundeck")

    def test_a_capped_scf_says_why_and_hands_back_no_energy(self):
        be = solid2()
        res = be.run(be.translate(si_inline(max_iter=2)))
        self.assertFalse(res.converged)
        self.assertIsNone(res.energy_Ha)
        self.assertIn("iteration limit", res.failure)

    def test_a_deck_the_library_refuses_raises_its_text(self):
        be, spec = solid2(), si_inline()
        job = be.translate(spec)
        deck = json.loads(job.native["deck"]); deck["kmseh"] = [1, 1, 1]
        job.native["deck"] = json.dumps(deck)
        with self.assertRaisesRegex(RuntimeError, "unknown key 'run.kmseh'"):
            be.run(job)

    def test_compute_u_is_rundeck_s(self):
        s = mno_plus_u()
        v = solid2().capabilities().can_run(RunSpec.from_dict(s.to_dict() | {"task": {"kind": "compute_U", "compute_U": {"method": "ACBN0"}}}))
        self.assertFalse(v)
        self.assertTrue(any("deck::Build stops at the SCF" in r for r in v.reasons), v.reasons)


if __name__ == "__main__":
    unittest.main()
