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

    def test_a_continued_crystal_says_it_resumed_exactly(self):
        be = solid2()
        first = be.run(be.translate(si_inline()))
        then = be.run(be.translate(D.continued(si_inline(accelerator="GDM"), first.state_file)))
        how = then.provenance["restartedFrom"]
        self.assertTrue(how["exact"], how)
        self.assertEqual(how["path"], first.state_file)
        self.assertTrue(then.resolved["starts from"].startswith("exact resume"), then.resolved["starts from"])
        self.assertIsNone(first.provenance["restartedFrom"], "a fresh run restarted from nothing")


def water(xyz=(0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107), basis="dzvp", **scf):
    from qviz.spec import Accuracy, molecule
    return molecule("water", [8, 1, 1], list(xyz), accuracy=Accuracy(representation={"basis": basis}, scf=scf or None))


@unittest.skipUnless(solid2()._qc is not None, "qchem_py not built (QCHEM_BUILD)")
class MolecularSavedStates(unittest.TestCase):
    """V-FACADE-STATE through the binding: a molecule saves after every run and continues through the same spec key a crystal
    uses (the solid2 deck block's state.restartFrom), judged by the library."""

    def test_a_molecule_saves_and_a_continuation_under_gdm_resumes_exactly(self):
        be = solid2()
        first = be.run(be.translate(water()))
        self.assertTrue(first.converged, first.failure)
        self.assertTrue(first.state_file and os.path.exists(first.state_file))
        rows = []
        then = be.run(be.translate(D.continued(water(accelerator="GDM"), first.state_file)), rows.append)
        self.assertTrue(then.converged, then.failure)
        how = then.provenance["restartedFrom"]
        self.assertTrue(how["exact"], how)
        self.assertAlmostEqual(how["electrons"], 10.0, places=8)
        self.assertLessEqual(len(rows), 4, "a continuation starts at the answer, not from scratch")
        self.assertAlmostEqual(rows[0].energy, first.energy_Ha, delta=1e-6, msg="its first iterate is the saved state")
        self.assertAlmostEqual(then.energy_Ha, first.energy_Ha, delta=1e-6)
        self.assertTrue(then.resolved["starts from"].startswith("exact resume"), then.resolved["starts from"])
        self.assertNotEqual(then.state_file, first.state_file, "the continuation banks its own state")

    def test_moved_atoms_are_a_warm_start_that_names_the_move(self):
        be = solid2()
        first = be.run(be.translate(water()))
        moved = water(xyz=(0, 0, 0, 0, 1.431, 1.171, 0, -1.431, 1.107))      # one H 0.064 bohr
        then = be.run(be.translate(D.continued(moved, first.state_file)))
        self.assertTrue(then.converged, then.failure)
        how = then.provenance["restartedFrom"]
        self.assertFalse(how["exact"], how)
        self.assertAlmostEqual(how["maxDisplacement"], 0.064, places=6)
        self.assertIn("atoms moved", then.resolved["starts from"])

    def test_a_state_of_another_basis_is_refused_in_the_library_s_words(self):
        be = solid2()
        first = be.run(be.translate(water()))
        with self.assertRaisesRegex(RuntimeError, "restart refused \\(mismatch\\)"):
            be.run(be.translate(D.continued(water(basis="tzvp"), first.state_file)))

    def test_judge_state_answers_without_running(self):
        be = solid2()
        first = be.run(be.translate(water()))
        n = be.translate(water()).native
        calc = be._qc.Calculator(n["numbers"], n["positions"], basis="dzvp", model="HF")
        self.assertEqual(calc.judge_state(first.state_file), {"exact": True, "warm": []})
        self.assertEqual(calc.judge_state("/nonexistent.h5")["refused"]["why"], "unreadable")
        self.assertIsNone(calc.restarted_from())

    def test_a_molecular_deck_has_no_keys_but_restart_from(self):
        from qviz.spec import Accuracy, molecule
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107],
                     accuracy=Accuracy(representation={"solid2": {"deck": {"solid": {"densityEcut": 20.0}}}}))
        v = solid2().capabilities().can_run(w)
        self.assertFalse(v)
        self.assertTrue(any("V-FACADE-OPTS" in r for r in v.reasons), v.reasons)
        ok = D.continued(water(), "/x.h5")
        self.assertTrue(solid2().capabilities().can_run(ok), solid2().capabilities().can_run(ok).reasons)


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
class HelperRuns(unittest.TestCase):
    """A +U crystal's record files its pseudo-atom SCFs first: what the Setup panel states is the crystal's own run."""
    RECORD = {"pseudo-atom Si (+U projector)@1": {"scf": {"standard": {"minFD": 1e-7, "nMaxIter": 120}},
                                                  "basis": {"perIrrep": [{"irrep": "s", "nFunctions": 9, "lambdaMin": 0.5, "cond": 2.0}]}},
              "inline build@2": {"basis": {"perIrrep": [{"irrep": "Γ", "nFunctions": 26, "lambdaMin": 0.1, "cond": 9.0}]}},
              "gridAudit reference@3": {"scf": {"standard": {"minFD": 1.0}}},
              "inline@4": {"scf": {"standard": {"minFD": 1e-5, "nMaxIter": 100}}}}

    def test_the_gates_are_the_run_s_own_not_a_pseudo_atom_s(self):
        from qviz.backends.solid2 import _gates_run
        self.assertEqual(_gates_run(self.RECORD), "run: [F,D] 1e-05; at most 100 iterations")

    def test_the_conditioning_is_the_crystal_s_not_a_pseudo_atom_s(self):
        from qviz import gridinfo
        lines = gridinfo.conditioning(self.RECORD)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("Γ: 26 functions", lines[0])


class Conditioning(unittest.TestCase):
    def test_the_verdict_and_its_floor_are_the_library_s(self):
        from qviz import gridinfo
        rows = [{"irrep": "A1 ", "nFunctions": 12, "lambdaMin": 5e-4, "cond": 9e3}, {"irrep": "B1", "nFunctions": 5, "lambdaMin": 0.05, "cond": 48.0}]
        whole = {"lambdaMin": 5e-4, "lambdaMinAt": "A1 ", "cond": 9e3, "condAt": "A1 ", "floor": 1e-3, "wouldDrop": 2}
        worst, a1, b1, verdict = gridinfo.conditioning({"m@1": {"basis": {"perIrrep": rows, "conditioning": whole}}})
        self.assertEqual(verdict, "⚠ near-singular: pick a trim (Basis tab: Trim near-dependent shells) and Build again")
        self.assertEqual(worst, "worst: S min eigenvalue 5.000e-04 at A1, condition number 9e+03 at A1; 2 function(s) would drop at floor 0.001")
        self.assertIn("⚠", a1, "under the library's floor, whatever that floor is")
        self.assertNotIn("⚠", b1)


class TheTrimIsReported(unittest.TestCase):
    """GUI item 27 (2026-11-11): "what was removed?  What is min eigenvalue, condition after the removal?  I can't tell if I am good
    to go" -- the vet stage's own record (basisTrim) and the redundant functions a pivoted Cholesky would drop (basis.redundant)."""

    def test_the_trim_says_what_went_and_s_before_and_after_and_the_verdict(self):
        from qviz import gridinfo
        whole = {"lambdaMin": 1.09e-6, "lambdaMinAt": "(0,0,0)", "cond": 9.8e6, "condAt": "(0,0,0)", "floor": 1e-6, "wouldDrop": 0}
        trim = {"orthoTol": 1e-6, "lambdaMinBefore": -5.13e-8, "lambdaMinAfter": 1.09e-6, "nShellsRemoved": 2,
                "passes": [{"removes": "Ni l=0 {0.06}"}, {"removes": "Ni l=0 {0.167104}"}]}
        lines = gridinfo.conditioning({"nio build@1": {"basisTrim": trim, "basis": {"nFunctions": 98, "conditioning": whole,
                                       "perIrrep": [{"irrep": "(0,0,0)", "nFunctions": 98, "lambdaMin": 1.09e-6, "cond": 9.8e6}]}}})
        self.assertIn("trimmed at orthoTol 1e-06: Ni l=0 {0.06}; Ni l=0 {0.167104} -- S min eigenvalue -5.130e-08 → 1.090e-06", lines)
        self.assertEqual(lines[-1], "✓ conditioning OK: no function near-dependent at the floor")

    def test_redundant_functions_are_a_report_never_called_removed(self):
        from qviz import gridinfo
        whole = {"lambdaMin": -5e-8, "lambdaMinAt": "(0,0,0)", "cond": 4e8, "condAt": "(0,0,0)", "floor": 1e-6, "wouldDrop": 5}
        lines = gridinfo.conditioning({"nio build@1": {"basis": {"conditioning": whole, "redundant": [{"irrep": "(0,0,0)", "index": 7}],
                                       "perIrrep": [{"irrep": "(0,0,0)", "nFunctions": 104, "lambdaMin": -5e-8, "cond": 4e8}]}}})
        self.assertTrue(any("redundant (a pivoted Cholesky would drop them; nothing removed): (0,0,0) #7" in l for l in lines), lines)
        self.assertFalse(any("removed" in l and "nothing removed" not in l for l in lines), lines)
        self.assertTrue(lines[-1].startswith("⚠ near-singular: pick a trim"))


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
        for mixer, knobs in (("Linear", {}), ("Kerker", {"kerker_g0": 0.8}), ("Pulay", {"pulay_depth": 6, "pulay_start": 3})):
            with self.subTest(mixer):
                deck = json.loads(be.translate(si_inline(mixer=mixer, **knobs)).native["deck"])
                self.assertEqual(deck["scf"]["mixer"], mixer)
                for ours, deck_key, _ in D.MIXER_KNOBS:
                    self.assertEqual(deck["scf"].get(deck_key), knobs.get(ours))
                r, _ = RundeckReadsThem.rundeck(self, deck, "--check")
                self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_a_stated_kerker_without_its_g0_is_refused_in_the_library_s_words(self):
        deck = json.loads(solid2().translate(si_inline(mixer="Kerker")).native["deck"])
        r, _ = RundeckReadsThem.rundeck(self, deck, "--check")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("KerkerG0 > 0", r.stderr + r.stdout)

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

    def test_epsilon_reaches_the_crystal_as_the_run_s_accuracy(self):
        d = si_inline().to_dict()
        d["accuracy"]["epsilon_Ha_per_atom"] = 3e-6
        spec = RunSpec.from_dict(d)
        be = solid2()
        deck = be.translate(spec).native["deck"]
        self.assertEqual(json.loads(deck)["accuracy"] if isinstance(deck, str) else deck["accuracy"], 3e-6)
        self.assertEqual(json.loads(be._qc.SolidRun(deck).resolved())["accuracy"], 3e-6)
        plain = be.translate(si_inline()).native["deck"]                    # at the default the deck states nothing ...
        self.assertNotIn("accuracy", json.loads(plain) if isinstance(plain, str) else plain)
        self.assertEqual(json.loads(be._qc.SolidRun(plain).resolved())["accuracy"], D.LIB_EPS)   # ... and runs the same ε

    def test_the_panel_s_gate_defaults_are_the_deck_s(self):
        from qviz.setup_panel import GATE_DEFAULTS
        be = solid2()
        scf = json.loads(be._qc.SolidRun(be.translate(si_inline()).native["deck"]).resolved())["scf"]
        for key, deck_key in D.DECK_GATES:
            self.assertEqual(scf[deck_key], GATE_DEFAULTS[key], deck_key)

    def test_the_panel_s_diis_defaults_are_what_the_library_runs(self):
        from qviz import molecules
        from qviz.setup_panel import accelerator_defaults
        be = solid2()
        diis = json.loads(be._qc.SolidRun(be.translate(si_inline()).native["deck"]).resolved())["acceleratorOptions"]["diis"]
        d = accelerator_defaults(si_inline(), mol=False)
        self.assertEqual((diis["nProj"], diis["svTol"], diis["engageBelowFD"]), (d["depth"], d["sv"], float(d["engage"])))
        spec = molecules.preset_spec("Methane (CH4)")                      # HF all-electron: the Z-scaled engage gate
        rec = next(iter(be.run(be.translate(spec)).record.values()))["accelerators"][0]
        d = accelerator_defaults(spec, mol=True)
        self.assertEqual((rec["nProj"], rec["svTol"]), (d["depth"], d["sv"]))
        self.assertEqual(f"{rec['engageBelowFD']:.3g}, Z-scaled", d["engage"])

    def test_levels_are_numbered_within_their_irrep(self):
        from qviz import molecules
        from qviz.spec import RunSpec
        d = molecules.preset_spec("Methane (CH4)").to_dict()
        d["accuracy"]["scf"] = {"impose_symmetry": True}
        res = solid2().run(solid2().translate(RunSpec.from_dict(d)))
        occ = [f"{L['n']}{L['label']}" for L in res.levels if L["occ"] > 0.5]
        self.assertEqual(sorted(occ), ["1A1", "1B1", "1B2", "2A1", "3A1"], "C2v blocking: t2 splits into 3A1 + 1B1 + 1B2")

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

    def test_the_library_refuses_a_molecule_s_g_space_mixer_and_runs_its_smearing(self):
        from qviz.backends.base import Refused
        from qviz.spec import Accuracy, molecule
        be = solid2()
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107], theory="DFT", functional="LDA",
                     accuracy=Accuracy(scf={"mixer": "Kerker"}))
        self.assertTrue(be.capabilities().can_run(w), "the GUI no longer pre-empts the library")
        with self.assertRaises(Refused) as got:
            be.run(be.translate(w))
        self.assertIn("periodic (G-space) run", str(got.exception))
        smeared = RunSpec.from_dict(w.to_dict() | {"accuracy": {"epsilon_Ha_per_atom": 1e-5},
                                                   "physics": w.to_dict()["physics"] | {"occupations": {"smearing": "fermi-dirac", "kT_Ha": 0.01}}})
        res = be.run(be.translate(smeared))
        self.assertTrue(res.converged, res.failure)

    def test_compute_u_is_rundeck_s(self):
        s = mno_plus_u()
        v = solid2().capabilities().can_run(RunSpec.from_dict(s.to_dict() | {"task": {"kind": "compute_U", "compute_U": {"method": "ACBN0"}}}))
        self.assertFalse(v)
        self.assertTrue(any("deck::Build stops at the SCF" in r for r in v.reasons), v.reasons)


if __name__ == "__main__":
    unittest.main()
