"""The RunSpec -> qchem deck translator (qviz.backends.solid2_deck; doc/RunSpecPlan.md §9 route (b)).

GOLDEN DECKS: a spec stating what qchem's banked bench decks state translates to exactly that deck; the decks it writes are
accepted by `rundeck --check`, and one runs end to end.  THE BRIDGE: the same deck through `qchem_py.SolidRun` (deck::Build)
lands on rundeck's energy to the last digit, and a refusal / an unconverged SCF comes back in the library's words.

    QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest tests.test_solid2_deck
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
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


class Continuation(unittest.TestCase):
    """The SCF knobs a continuation changes reach the deck, and the start rides the deck's own state block."""
    def test_the_stated_accelerator_reaches_the_deck(self):
        self.assertEqual(D.to_deck(si_inline(accelerator="GDM"))["solid"]["accelerator"], "GDM")
        self.assertNotIn("accelerator", D.to_deck(si_inline()).get("solid", {}), "unstated: the facade's Auto policy")

    def test_continued_round_trips_and_keeps_the_recipe(self):
        s = D.continued(si_inline(accelerator="GDM"), "/x/Si.h5")
        self.assertEqual(D.continues_from(s), "/x/Si.h5")
        deck = D.to_deck(s)
        self.assertEqual(deck["state"], {"restartFrom": "/x/Si.h5"})
        self.assertEqual(deck["solid"]["densityEcut"], 20.0, "the stated recipe stays")
        fresh = D.continued(s, None)
        self.assertIsNone(D.continues_from(fresh))
        self.assertEqual(fresh, si_inline(accelerator="GDM"), "start-from-scratch is the spec it came from")

    def test_every_crystal_deck_saves_to_its_own_file(self):
        from qviz import states
        a, b = (json.loads(solid2().translate(s).native["deck"]) for s in (si_inline(), si_inline(accelerator="GDM")))
        self.assertTrue(a["state"]["save"].startswith(str(states.state_dir())))
        self.assertNotEqual(a["state"]["save"], b["state"]["save"], "another deck, another file")
        self.assertEqual(a["state"]["save"], json.loads(solid2().translate(si_inline()).native["deck"])["state"]["save"],
                         "the same deck again overwrites its own file")


@unittest.skipUnless(solid2()._qc is not None and RUNDECK.exists(), "qchem_py / rundeck not built (QCHEM_BUILD)")
class SavedStates(unittest.TestCase):
    def test_a_run_saves_its_state_and_a_continuation_under_gdm_resumes_from_it(self):
        be = solid2()
        first = be.run(be.translate(si_inline()))          # the facade's Auto (stated DIIS on Si Γ: 100 iterations, no convergence)
        self.assertTrue(first.converged, first.failure)
        self.assertTrue(first.state_file and os.path.exists(first.state_file))
        rows = []
        then = be.run(be.translate(D.continued(si_inline(accelerator="GDM"), first.state_file)), rows.append)
        self.assertTrue(then.converged, then.failure)
        self.assertLessEqual(len(rows), 4, "a continuation starts at the answer, not from scratch")
        self.assertAlmostEqual(rows[0].energy, first.energy_Ha, delta=1e-6, msg="its first iterate is the saved state")
        # the first run stopped at [F,D] ~5e-6 (its gate); the resume, on the saved D, reads [F,D] ~1e-14 in one iteration -- the
        # energies agree to the FIRST run's convergence (measured 1.9e-7 Ha), not tighter
        self.assertAlmostEqual(then.energy_Ha, first.energy_Ha, delta=1e-6)
        self.assertNotEqual(then.state_file, first.state_file, "the continuation banks its own state")

    def test_a_state_of_another_crystal_is_refused_in_the_library_s_words(self):
        be = solid2()
        si = be.run(be.translate(si_inline()))
        strained = crystal("Si strained", fcc(10.5), SI[1],
                           accuracy={"representation": {"basis": "sipp_sr", "solid2": {"deck": {"solid": {"densityEcut": 20.0}}}}})
        with self.assertRaisesRegex(RuntimeError, "refused"):
            be.run(be.translate(D.continued(strained, si.state_file)))

    def test_a_molecule_cannot_continue_yet(self):
        from qviz.spec import Accuracy, molecule
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107],
                     accuracy=Accuracy(representation={"solid2": {"deck": {"state": {"restartFrom": "/x.h5"}}}}))
        v = solid2().capabilities().can_run(w)
        self.assertFalse(v)
        self.assertTrue(any("no saved state" in r for r in v.reasons), v.reasons)


class States(unittest.TestCase):
    def test_usage_counts_a_shared_file_once_and_release_keeps_it_while_shared(self):
        from types import SimpleNamespace as NS
        from qviz import states
        d = pathlib.Path(tempfile.mkdtemp())
        f = d / "a.h5"; f.write_bytes(b"x" * 1000)
        r1, r2 = (NS(result=NS(state_file=str(f))) for _ in range(2))
        self.assertEqual(states.usage([r1, r2]), (1, 1000))
        self.assertIsNone(states.release(r1, [r1, r2]), "r2 still points at it")
        self.assertTrue(f.exists())
        self.assertEqual(states.release(r1, [r1]), str(f))
        self.assertFalse(f.exists())
        self.assertIsNone(r1.result.state_file)
        self.assertEqual(states.usage([r1, r2]), (0, 0))


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

    def test_the_stated_mixer_reaches_the_deck_and_the_library_takes_it(self):
        be = solid2()
        for mixer in ("Linear", "Kerker", "Pulay"):
            with self.subTest(mixer):
                deck = json.loads(be.translate(si_inline(mixer=mixer)).native["deck"])
                self.assertEqual(deck["scf"]["mixer"], mixer)
                r, _ = RundeckReadsThem.rundeck(self, deck, "--check")
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_the_linear_alphas_reach_the_deck(self):
        deck = json.loads(solid2().translate(si_inline(mixer="Linear", mixing_alpha=0.3, mixing_alpha_max=0.5)).native["deck"])
        self.assertEqual((deck["scf"]["startingRelaxRo"], deck["scf"]["maxRelaxRo"]), (0.3, 0.5))
        r, _ = RundeckReadsThem.rundeck(self, deck, "--check")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_the_accelerator_knobs_reach_a_deck_the_library_takes(self):
        deck = json.loads(solid2().translate(si_inline(accelerator="Ladder", diis_depth=6, diis_engage_fd=0.1,
                                                       ladder_switch_at=1e-5, mixer="Linear", mixing_alpha_min=0.05)).native["deck"])
        r, _ = RundeckReadsThem.rundeck(self, deck, "--check")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_the_gates_a_run_states_are_the_gates_it_ran_and_the_row_carries(self):
        from qviz.spec import Accuracy, molecule
        be = solid2()
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107], theory="DFT", functional="LDA",
                     accuracy=Accuracy(scf={"gate_de": 1e-8, "gate_drho": "off"}))
        rows = []
        res = be.run(be.translate(w), rows.append)
        self.assertTrue(res.converged, res.failure)
        std = next(r["scf"]["standard"] for k, r in res.record.items() if (r.get("scf") or {}).get("standard"))
        self.assertEqual((std["minDE"], std["minDrho"], std["minFD"]), (1e-8, "off", 1e-5))
        tol = {c.title: c.tolerance for c in rows[-1].cells}
        self.assertEqual((tol["ΔE/E"], tol["Δρ"], tol["[F,D]"]), (1e-8, None, 1e-5))
        deck = json.loads(be.translate(si_inline(gate_de=1e-7, gate_dfd="off")).native["deck"])
        r, _ = RundeckReadsThem.rundeck(self, deck, "--check")
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_a_molecular_ladder_hands_over_to_gdm(self):
        from qviz.spec import Accuracy, molecule
        be = solid2()
        for switch, hands in ((None, True), (0.0, False)):
            with self.subTest(switch_at=switch):
                w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107], theory="DFT", functional="LDA",
                             accuracy=Accuracy(scf={"accelerator": "Ladder", **({} if switch is None else {"ladder_switch_at": switch})}))
                rows = []
                res = be.run(be.translate(w), rows.append)
                self.assertTrue(res.converged, res.failure)
                accel = [c.text for r in rows for c in r.cells if c.title == "accel"]
                self.assertEqual(any("GDM" in a for a in accel), hands, accel)

    def test_a_molecule_takes_no_g_space_mixer(self):
        from qviz.spec import Accuracy, molecule
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107], theory="DFT", functional="LDA",
                     accuracy=Accuracy(scf={"mixer": "Kerker"}))
        v = solid2().capabilities().can_run(w)
        self.assertFalse(v)
        self.assertTrue(any("mix in G space" in r for r in v.reasons), v.reasons)

    def test_compute_u_is_rundeck_s(self):
        s = mno_plus_u()
        v = solid2().capabilities().can_run(RunSpec.from_dict(s.to_dict() | {"task": {"kind": "compute_U", "compute_U": {"method": "ACBN0"}}}))
        self.assertFalse(v)
        self.assertTrue(any("deck::Build stops at the SCF" in r for r in v.reasons), v.reasons)


if __name__ == "__main__":
    unittest.main()
