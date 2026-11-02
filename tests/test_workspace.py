"""The Workspace/Run model on the Backend seam: a spec runs on the first back end that can run it, a refused spec
FAILS with the strip's reasons, the trace streams once and is kept, the project round-trips specs and back ends,
and SOLID² (when the qchem extension is built) reproduces the water anchor.

    QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest discover -s tests
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from qviz import project
from qviz.backends import BackendRegistry, FakeBackend, Solid2Backend
from qviz.spec import Accuracy, RunSpec, molecule
from qviz.workspace import RunStatus, Workspace

WATER = ([8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107])
ROOT = pathlib.Path(__file__).resolve().parents[1]
SOLID2 = Solid2Backend()


def water(**kw):
    return molecule("water", *WATER, **kw)


class WorkspaceOnFake(unittest.TestCase):
    def ws(self):
        return Workspace("t", registry=BackendRegistry([FakeBackend()]))

    def test_a_run_streams_its_trace_once_and_keeps_it(self):
        steps = []
        run = self.ws().add_run(water(), compute=False)
        run.compute(on_step=steps.append)
        self.assertIs(run.status, RunStatus.READY, run.error)
        self.assertEqual(len(steps), len(run.scf))
        self.assertEqual(run.scf[0].iteration, 1)
        self.assertAlmostEqual(run.energy, run.scf[-1].energy)
        self.assertEqual(run.density(16).values.shape, (16, 16, 16))
        self.assertIsNone(run.spin_density(16), "no spin field offered")
        rep = run.report("orbitals")
        self.assertEqual(rep["num_occupied"], 5)

    def test_a_spec_no_back_end_can_run_fails_with_the_reasons(self):
        bad = RunSpec.from_dict(json.loads(water().to_json()) | {"task": {"kind": "relax"}})
        run = self.ws().add_run(bad)
        self.assertIs(run.status, RunStatus.FAILED)
        self.assertIn("single-point energy only", run.error)

    def test_the_project_round_trips_specs_and_back_ends(self):
        ws = self.ws()
        ws.add_run(water())
        ws.add_run(water(theory="DFT", functional="LDA", multiplicity=3))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.qproj.h5")
            project.save_workspace(path, ws)
            back = project.load_workspace(path, ws.registry, compute=False)
        self.assertEqual([r.spec for r in back.runs], [r.spec for r in ws.runs])
        self.assertEqual([r.backend_name for r in back.runs], ["Fake", "Fake"])

    def test_a_finished_run_comes_back_with_its_result_and_no_compute(self):
        ws = self.ws()
        done = ws.add_run(water())
        done.result.record = {"water@t": {"grids": {"xcQuadrature": {"points": 100}}}}
        quit_mid_run = ws.add_run(water(theory="DFT", functional="LDA"), compute=False)   # still pending at quit
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.qproj.h5")
            project.save_workspace(path, ws)
            self.assertEqual(ws.path, path)
            back = project.load_workspace(path, ws.registry, compute=False)
        a, b = back.runs
        self.assertIs(a.status, RunStatus.READY)
        self.assertTrue(a.restored, "no compute: the result came from the file")
        self.assertEqual(a.energy, done.energy)
        self.assertEqual([(c.title, c.text, c.tolerance) for c in a.scf[-1].cells],
                         [(c.title, c.text, c.tolerance) for c in done.scf[-1].cells], "printed text and gates survive")
        self.assertEqual(a.result.record, done.result.record)
        self.assertEqual(a.report("orbitals")["num_occupied"], done.report("orbitals")["num_occupied"])
        self.assertFalse(a.offers("density"), "fields need the live calculation")
        self.assertIs(b.status, RunStatus.FAILED)
        self.assertIn("quit", b.error)
        self.assertEqual(back.path, path)
        a.compute()
        self.assertFalse(a.restored)
        self.assertTrue(a.offers("density"), "a recompute brings the fields back")

    def test_a_v1_workspace_is_refused_and_the_converter_makes_a_v2_one(self):
        with tempfile.TemporaryDirectory() as d:
            v1, v2 = os.path.join(d, "v1.h5"), os.path.join(d, "v2.h5")
            with h5py.File(v1, "w") as h:
                h.attrs.update(format="qviz-workspace/1", title="old", selected=0)
                rg = h.create_group("runs").create_group("run000")
                rg.attrs.update(label="water", method="LDA", basis="dzvp", symmetry=True, multiplicity=0,
                                energy=np.nan, status="ready")
                rg.create_dataset("numbers", data=np.asarray(WATER[0]))
                rg.create_dataset("positions", data=np.asarray(WATER[1], float))
            with self.assertRaisesRegex(ValueError, "convert_workspace_v1"):
                project.load_workspace(v1, BackendRegistry([FakeBackend()]), compute=False)
            subprocess.run([sys.executable, str(ROOT / "scripts/convert_workspace_v1.py"), v1, v2],
                           check=True, capture_output=True)
            ws = project.load_workspace(v2, BackendRegistry([FakeBackend()]), compute=False)
        s = ws.runs[0].spec
        self.assertEqual((s.physics.theory, s.physics.functional), ("DFT", "LDA"))
        self.assertEqual(s.accuracy.representation, {"basis": "dzvp"})
        self.assertTrue(s.accuracy.scf["impose_symmetry"])


@unittest.skipUnless(SOLID2.capabilities().installed, "the qchem extension is not built (set QCHEM_BUILD)")
class Solid2EndToEnd(unittest.TestCase):
    def ws(self):
        return Workspace("t", registry=BackendRegistry([SOLID2, FakeBackend()]))

    def test_water_HF_dzvp_reproduces_the_anchor_and_streams_from_iteration_1(self):
        steps = []
        run = self.ws().add_run(water(accuracy=Accuracy(representation={"basis": "dzvp"})), compute=False)
        self.assertEqual(run.backend_name, "SOLID²", "SOLID² is the default when it can run the spec")
        run.compute(on_step=steps.append)
        self.assertIs(run.status, RunStatus.READY, run.error)
        self.assertAlmostEqual(run.energy, -76.0229032, places=6)     # M_HF_U water anchor
        self.assertEqual(steps[0].iteration, 1)
        self.assertEqual(len(steps), len(run.scf))
        rep = run.report("orbitals")
        self.assertEqual(rep["num_occupied"], 5)
        self.assertGreater(rep["num_orbitals"], 5, "virtuals are listed")

    def test_stop_ends_the_scf_at_the_next_iteration(self):
        steps = []
        run = self.ws().add_run(water(accuracy=Accuracy(representation={"basis": "dzvp"})), compute=False)

        def watch(row):
            steps.append(row)
            if row.iteration == 2:
                run.stop()                                     # the Stop button, from the GUI thread
        run.compute(on_step=watch)
        self.assertIs(run.status, RunStatus.FAILED)
        self.assertEqual(run.error, "stopped by the user")
        self.assertEqual([r.iteration for r in steps], [1, 2], "iteration 3 is where the stop lands")

    def test_alpha_max_caps_the_linear_mixing_step(self):
        steps = []
        spec = water(theory="DFT", functional="LDA",
                     accuracy=Accuracy(representation={"basis": "dzvp"}, scf={"mixing_alpha": 0.2, "mixing_alpha_max": 0.3}))
        run = self.ws().add_run(spec, compute=False)
        run.compute(on_step=steps.append)
        self.assertIs(run.status, RunStatus.READY, run.error)
        alphas = [float(c.text.split()[1]) for r in steps for c in r.cells if c.title == "ρ_mix" and c.text.strip().startswith("Lin")]
        self.assertTrue(alphas, "a linear-mixing run prints its step")
        self.assertAlmostEqual(alphas[0], 0.2)
        self.assertLessEqual(max(alphas), 0.3)

    def test_the_row_is_the_console_s_cells_and_a_capped_run_says_why(self):
        steps = []
        run = self.ws().add_run(water(accuracy=Accuracy(representation={"basis": "dzvp"})), compute=False)
        run.compute(on_step=steps.append)
        self.assertEqual(steps[-1].titles[:4], ("Etotal", "[F,D]", "ΔE/E", "Δρ"), "the molecular console's columns")
        fd = steps[-1].cells[1]
        self.assertEqual((fd.scale, fd.tolerance), ("log", 1e-5))
        self.assertAlmostEqual(steps[-1].value("Etotal"), steps[-1].energy)
        state = {c.title: c for c in steps[-1].cells if c.title in ("ρ_mix", "accel", "cfg")}
        self.assertEqual(set(state), {"ρ_mix", "accel", "cfg"})
        self.assertTrue(all(math.isnan(c.value) for c in state.values()), "solver STATE is table-only: no number to plot")
        self.assertTrue(state["ρ_mix"].text.strip() and state["accel"].text.strip(), "the printed text is kept")
        capped = self.ws().add_run(water(accuracy=Accuracy(representation={"basis": "dzvp"}, scf={"max_iter": 2})))
        self.assertIs(capped.status, RunStatus.FAILED)
        self.assertIn("not converged after 2 iterations", capped.error, "the SCFFailure's reason, through the bridge")
        self.assertIsNone(capped.result.energy_Ha, "a failed SCF hands back no plausible number")

    def test_the_O2_triplet_offers_a_spin_density(self):
        run = self.ws().add_run(molecule("O2", [8, 8], [0, 0, 1.141, 0, 0, -1.141], multiplicity=3,
                                         accuracy=Accuracy(representation={"basis": "dzvp"})))
        self.assertIs(run.status, RunStatus.READY, run.error)
        m = run.spin_density(20)
        self.assertIsNotNone(m)
        self.assertGreater(m.values.max(), 1e-3)

    def test_a_polarized_closed_shell_runs_spin_native_and_has_no_moment(self):
        polarized = RunSpec.from_dict(json.loads(water(accuracy=Accuracy(representation={"basis": "dzvp"})).to_json()))
        d = polarized.to_dict(); d["physics"]["spin"]["polarized"] = True
        run = self.ws().add_run(RunSpec.from_dict(d).validate())
        self.assertIs(run.status, RunStatus.READY, run.error)
        self.assertAlmostEqual(run.energy, -76.0229032, places=6)
        self.assertTrue(run.result.polarized, "both spin channels are listed")
        self.assertLess(np.abs(run.spin_density(16).values).max(), 1e-8)

    def test_a_built_crystal_runs_in_process_through_the_deck(self):
        h = 5.13
        spec = RunSpec.from_dict({
            "label": "Si cell", "structure": {"kind": "crystal", "lattice_bohr": [[0, h, h], [h, 0, h], [h, h, 0]],
                                              "sites": [{"element": "Si", "frac": [0, 0, 0]}, {"element": "Si", "frac": [0.25, 0.25, 0.25]}]},
            "physics": {"theory": "DFT", "functional": "LDA", "core": "pseudopotential", "spin": {"polarized": False}},
            "accuracy": {"kmesh": [1, 1, 1], "representation": {"basis": "sipp_sr", "solid2": {"deck": {"solid": {"densityEcut": 20.0}}}}}})
        steps = []
        run = self.ws().add_run(spec, compute=False)
        self.assertEqual(run.backend_name, "SOLID²")
        run.compute(on_step=steps.append)
        self.assertIs(run.status, RunStatus.READY, run.error)
        self.assertAlmostEqual(run.energy, -7.1150676382, places=8)     # rundeck decks/Si_diamond_inline.json, same deck
        self.assertEqual(steps[0].titles[:3], ("Etotal", "[F,D]", "ΔE/E"), "the periodic console's columns")
        # a cell whose stated electron count is not its valence sum: the setup refuses it, in its own words, before any SCF
        bad = RunSpec.from_dict(spec.to_dict() | {"accuracy": {"kmesh": [1, 1, 1], "representation": {
            "basis": "sipp_sr", "solid2": {"deck": {"solid": {"densityEcut": 20.0, "Nelec": 7}}}}}})
        refused, steps = self.ws().add_run(bad, compute=False), []
        refused.compute(on_step=steps.append)
        self.assertIs(refused.status, RunStatus.FAILED)
        self.assertIn("Nelec 7 disagrees with the species", refused.error)
        self.assertEqual(steps, [], "refused at setup: no iteration ran")

    def test_a_crystal_the_deck_cannot_say_is_refused_by_capability_not_by_a_crash(self):
        v = SOLID2.capabilities().can_run(RunSpec.from_dict({
            "label": "Si", "structure": {"kind": "crystal", "lattice_bohr": [[0, 5.13, 5.13], [5.13, 0, 5.13], [5.13, 5.13, 0]],
                                         "sites": [{"element": "Si", "frac": [0, 0, 0]}, {"element": "Si", "frac": [0.25, 0.25, 0.25]}]}}))
        self.assertFalse(v)
        self.assertTrue(any("accuracy.kmesh" in r for r in v.reasons), v.reasons)   # the deck translator's own reason



class Solid2BasisCoverage(unittest.TestCase):
    def test_a_basis_without_a_block_for_an_element_is_refused_by_name(self):
        caps = Solid2Backend().capabilities()
        pp = molecule("water", *WATER, theory="DFT", functional="LDA", core="pseudopotential",
                      accuracy=Accuracy(representation={"basis": "valence_lowq_vb"}))
        self.assertIn("basis valence_lowq_vb has no block for H", caps.can_run(pp).reasons)
        ae = molecule("water", *WATER, theory="DFT", functional="LDA", accuracy=Accuracy(representation={"basis": "dzvp"}))
        self.assertTrue(caps.can_run(ae).ok)


class TheRunnerStops(unittest.TestCase):
    def test_a_stopped_stream_terminates_the_process(self):
        import time
        from qviz.runner import MpiRunner
        from qviz.workspace import RunStopped

        class Direct(MpiRunner):                               # no mpirun: the stream, not the launcher, is under test
            def argv(self, command):
                return list(command)

        def on_line(line):
            raise RunStopped()
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as d, self.assertRaises(RunStopped):
            Direct().run((sys.executable, "-u", "-c", "import time\nwhile True: print('x'); time.sleep(0.05)"),
                         pathlib.Path(d), "p", {}, {}, on_line)
        self.assertLess(time.monotonic() - t0, 5, "terminated, not waited out")

if __name__ == "__main__":
    unittest.main()
