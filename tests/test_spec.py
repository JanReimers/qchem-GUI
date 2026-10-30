"""qviz.spec + qviz.capability: the neutral run spec reads strictly, validates every defect at once,
round-trips through JSON, and filters back ends by what the spec needs (option 2.2).

    .venv/bin/python -m unittest discover -s tests
"""
import json
import unittest

from qviz.spec import RunSpec, SpecError, molecule
from qviz.capability import Capability, Capabilities, filter_backends

WATER = ((8, 1, 1), (0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107))

# qchem's banked CP2K oracle cell, MnO AFM-II (IntegrationTests/CP2K/mno_afm2_gpw_va_plusU.inp)
MNO = {
    "schema": "solid2-runspec/1",
    "label": "MnO AFM-II +U 4 eV",
    "_doc": "a comment key: ignored, not kept",
    "structure": {
        "kind": "crystal",
        "lattice_bohr": [[8.4, 4.2, 4.2], [4.2, 8.4, 4.2], [4.2, 4.2, 8.4]],
        "sites": [{"element": "Mn", "frac": [0, 0, 0], "tag": "Mn1"},
                  {"element": "Mn", "frac": [0.5, 0.5, 0.5], "tag": "Mn2"},
                  {"element": "O", "frac": [0.25, 0.25, 0.25]},
                  {"element": "O", "frac": [0.75, 0.75, 0.75]}],
        "source": {"materials": "MnO_AFM2"},
    },
    "physics": {
        "theory": "DFT", "functional": "LDA", "core": "pseudopotential",
        "spin": {"polarized": True, "site_moments": [5, -5, 0, 0]},
        "hubbard": [{"sites": ["Mn1", "Mn2"], "l": 2, "U_eV": 4.0}],
        "occupations": {"smearing": "fermi-dirac", "kT_Ha": 5e-3},
    },
    "accuracy": {"epsilon_Ha_per_atom": 1e-5, "kmesh": [1, 1, 1]},
}


class SpecRead(unittest.TestCase):
    def test_a_crystal_reads_validates_and_round_trips(self):
        s = RunSpec.from_dict(MNO).validate()
        self.assertEqual(s.structure.sites[1].tag, "Mn2")
        self.assertEqual(s.physics.spin.site_moments, (5, -5, 0, 0))
        self.assertEqual(s.num_electrons, 2 * 25 + 2 * 8)
        again = RunSpec.from_json(s.to_json())
        self.assertEqual(again, s)
        self.assertEqual(hash(again), hash(s))
        self.assertNotIn("_doc", s.to_json())

    def test_an_unknown_key_is_refused_naming_the_path_and_the_legal_keys(self):
        bad = json.loads(json.dumps(MNO)); bad["physics"]["spin"]["moments"] = [5, -5, 0, 0]
        with self.assertRaisesRegex(SpecError, r"physics\.spin\.moments.*site_moments"):
            RunSpec.from_dict(bad)

    def test_a_missing_key_keeps_its_default(self):
        d = {k: v for k, v in MNO.items() if k != "task"}
        self.assertEqual(RunSpec.from_dict(d).task.kind, "energy")

    def test_validation_lists_every_defect_at_once(self):
        bad = json.loads(json.dumps(MNO))
        bad["physics"]["spin"]["site_moments"] = [5, -5]                 # wrong length
        bad["physics"]["hubbard"][0]["sites"] = ["Mn3"]                   # unknown tag
        bad["physics"]["occupations"] = {"smearing": "fermi-dirac"}       # kT missing
        with self.assertRaises(SpecError) as e:
            RunSpec.from_dict(bad).validate()
        msg = str(e.exception)
        for needle in ("site_moments needs one entry per site", "unknown tags", "kT_Ha"):
            self.assertIn(needle, msg)

    def test_a_U_left_unset_must_be_computed(self):
        bad = json.loads(json.dumps(MNO)); del bad["physics"]["hubbard"][0]["U_eV"]
        with self.assertRaisesRegex(SpecError, "U_eV is unset"):
            RunSpec.from_dict(bad).validate()
        bad["task"] = {"kind": "compute_U", "compute_U": {"method": "linear-response", "sites": ["Mn1"]}}
        RunSpec.from_dict(bad).validate()


class SpecMolecule(unittest.TestCase):
    def test_water_from_the_picker_shape(self):
        s = molecule("water", *WATER)
        self.assertEqual(s.structure.kind, "molecule")
        self.assertFalse(s.physics.spin.polarized)
        self.assertEqual(s.num_electrons, 10)
        self.assertEqual(RunSpec.from_json(s.to_json()), s)

    def test_an_open_shell_is_polarized_and_its_parity_is_checked(self):
        self.assertTrue(molecule("water triplet", *WATER, multiplicity=3).physics.spin.polarized)
        with self.assertRaisesRegex(SpecError, "parity"):
            molecule("water doublet", *WATER, multiplicity=2)


class CapabilityFilter(unittest.TestCase):
    def caps(self):
        no_orbital_U = Capability("physics.hubbard.form=Dudarev",
                                  lambda s: all(h.form == "Dudarev" for h in s.physics.hubbard), "Dudarev U only")
        crystals_only = Capability("structure.kind=crystal",
                                   lambda s: s.structure.kind == "crystal", "crystals only")
        no_compute_U = Capability("task.kind!=compute_U", lambda s: s.task.kind != "compute_U", "cannot compute U")
        return [Capabilities("SOLID²", []),
                Capabilities("CP2K", [no_orbital_U, no_compute_U]),
                Capabilities("PWonly", [crystals_only, no_compute_U]),
                Capabilities("VASP", [], installed=False, why_absent="no licensed VASP on this box")]

    def test_a_spec_enables_the_back_ends_that_can_run_it_and_says_why_not_for_the_rest(self):
        v = filter_backends(molecule("water", *WATER), self.caps())
        self.assertTrue(v["SOLID²"]); self.assertTrue(v["CP2K"])
        self.assertFalse(v["PWonly"]); self.assertEqual(v["PWonly"].reasons, ("crystals only",))
        self.assertFalse(v["VASP"]); self.assertIn("licensed", v["VASP"].reasons[0])

    def test_every_reason_is_collected_not_the_first(self):
        d = json.loads(json.dumps(MNO))
        del d["physics"]["hubbard"][0]["U_eV"]; d["physics"]["hubbard"][0]["form"] = "Liechtenstein"
        d["task"] = {"kind": "compute_U", "compute_U": {"sites": ["Mn1"]}}
        v = filter_backends(RunSpec.from_dict(d).validate(), self.caps())
        self.assertEqual(v["CP2K"].reasons, ("Dudarev U only", "cannot compute U"))
        self.assertTrue(v["SOLID²"])


if __name__ == "__main__":
    unittest.main()
