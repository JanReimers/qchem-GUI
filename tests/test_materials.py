"""Crystal import / export and the materials picker (OpenWork §2, ruled 2026-11-09): the LIBRARY translates a file into one
materials.json entry and lists its layers (shipped, then the user's own); the GUI turns an entry into a crystal draft and back.

    QCHEM_BUILD=~/Code/qchem-lane-g/build/Release .venv/bin/python -m unittest tests.test_materials
"""
import os as _os, tempfile as _tempfile
_os.environ.setdefault("QCHEM_GUI_STATES", _tempfile.mkdtemp(prefix="qchem-gui-states-"))   # saved SCF states: never ~
import json
import os
import pathlib
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets                                     # noqa: E402

from qviz import materials                                        # noqa: E402
materials.USER_LIBRARY = os.path.join(_tempfile.mkdtemp(prefix="qchem-gui-materials-"), "materials.json")   # never ~/.solid2
from qviz.backends import solid2_deck                             # noqa: E402
from qviz.backends.base import BackendRegistry                    # noqa: E402
from qviz.backends import FakeBackend                             # noqa: E402
from qviz.draft import SpecDraft                                  # noqa: E402
from qviz.setup_panel import SetupPanel                           # noqa: E402
import tempfile as _bse_tmp; from qviz import bse as _bse; _bse.BASES_DIR = _bse_tmp.mkdtemp(prefix="qchem-gui-bases-")   # never ~/.solid2/bases

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

ROCKSALT = {"comment": "an import", "lattice": {"vectors_bohr": [[0, 4, 4], [4, 0, 4], [4, 4, 0]]},
            "atoms": [{"el": "Mn", "frac": [0, 0, 0], "spin": 1}, {"el": "O", "frac": [0.5, 0.5, 0.5]}],
            "provenance": {"source": "x.cif", "format": "CIF", "imported": "2026-11-09"}}


def live() -> bool:
    try:
        return hasattr(materials._qc(), "materials")
    except materials.MaterialsError:
        return False


def cod_periclase() -> pathlib.Path | None:
    for root in (solid2_deck.qchem_src(), pathlib.Path.home() / "Code/qchem-lane-g"):
        p = root / "src/Structure/tests/data/cod_9006799_periclase.cif"
        if p.exists():
            return p
    return None


class TheEntryAndTheSpec(unittest.TestCase):
    """Pure translation, no library: an entry is a crystal spec, and the spec is that entry again."""

    def test_an_entry_is_a_crystal_draft_and_back(self):
        spec = materials.spec_of("MnO", ROCKSALT, ROCKSALT["lattice"]["vectors_bohr"], {"import": ROCKSALT["provenance"], "comment": "an import"})
        self.assertEqual(spec.structure.kind, "crystal")
        self.assertEqual(spec.physics.spin.site_moments, (1.0, 0.0), "the entry's spin decoration is the site moments")
        self.assertTrue(spec.physics.spin.polarized)
        self.assertEqual(materials.entry_of(spec), ROCKSALT, "the spec is the entry again, provenance and spins included")

    def test_site_spins_are_one_sign_per_site(self):
        self.assertEqual(materials.parse_spins("+ - 0 0", 4), (1.0, -1.0, 0.0, 0.0))
        self.assertEqual(materials.parse_spins("+1, -1", 2), (1.0, -1.0))
        self.assertIsNone(materials.parse_spins("0 0", 2), "all zero is no order")
        self.assertIsNone(materials.parse_spins("", 2))
        with self.assertRaisesRegex(ValueError, "3 spins for 2 sites"):
            materials.parse_spins("+ - 0", 2)
        with self.assertRaisesRegex(ValueError, "not up"):
            materials.parse_spins("up down", 2)

    def test_an_imported_cell_keeps_its_provenance_in_the_deck(self):
        spec = materials.spec_of("MnO", ROCKSALT, ROCKSALT["lattice"]["vectors_bohr"], {"import": ROCKSALT["provenance"]})
        entry = solid2_deck.structure_entry(spec)
        self.assertEqual(entry["provenance"], ROCKSALT["provenance"], "the inline structure says where it came from")


@unittest.skipUnless(live(), "qchem_py with the materials face not built (QCHEM_BUILD)")
class TheLibraryTranslates(unittest.TestCase):
    def test_the_picker_lists_the_shipped_materials_with_their_cells(self):
        ms = {m["name"]: m for m in materials.listing()}
        self.assertEqual(ms["Si_diamond"]["layer"], "shipped")
        spec = materials.pick(ms["NaF_rocksalt"])
        self.assertEqual(spec.structure.source["materials"], "NaF_rocksalt")
        self.assertEqual(solid2_deck.structure_entry(spec), "NaF_rocksalt", "a shipped material reaches the deck by name")
        self.assertEqual(materials.pick(ms["MnO_AFM2"]).physics.spin.site_moments, (1.0, -1.0, 0.0, 0.0))

    @unittest.skipUnless(cod_periclase(), "the COD periclase CIF is not in this qchem tree")
    def test_a_cod_cif_imports_as_its_primitive_cell(self):
        spec, warnings = materials.import_file(str(cod_periclase()))
        self.assertEqual(warnings, [])
        self.assertEqual([s.element for s in spec.structure.sites], ["Mg", "O"])
        self.assertEqual(spec.structure.source["import"]["spaceGroup"]["stated"], "F m 3 m")
        cif = materials.export_text(spec, "cif")
        self.assertIn("_cell_length_a", cif)
        self.assertIn("Mg1 Mg", cif)

    def test_a_refused_import_says_why_in_the_library_s_words(self):
        d = pathlib.Path(_tempfile.mkdtemp())
        p = d / "half.cif"
        p.write_text("data_x\n_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4\n_cell_angle_alpha 90\n_cell_angle_beta 90\n"
                     "_cell_angle_gamma 90\nloop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
                     "_atom_site_occupancy\nNa1 0 0 0 0.5\n")
        with self.assertRaisesRegex(materials.MaterialsError, "site Na1 has occupancy 0.5"):
            materials.import_file(str(p))

    def test_save_to_my_materials_lists_it_as_mine_and_never_takes_a_shipped_name(self):
        spec = materials.spec_of("MnO mine", ROCKSALT, ROCKSALT["lattice"]["vectors_bohr"], {"import": ROCKSALT["provenance"]})
        path = materials.save("MnO_mine_test", spec, replace=True)
        self.assertEqual(path, materials.USER_LIBRARY)
        mine = [m for m in materials.listing() if m["name"] == "MnO_mine_test"]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]["layer"], "user")
        self.assertEqual(mine[0]["entry"]["provenance"], ROCKSALT["provenance"])
        back = materials.pick(mine[0])
        self.assertEqual(back.physics.spin.site_moments, (1.0, 0.0))
        self.assertIsInstance(solid2_deck.structure_entry(back), dict, "a user material goes INLINE: the deck does not depend on ~")
        with self.assertRaisesRegex(materials.MaterialsError, "already in the shipped materials"):
            materials.save("Si_diamond", spec)
        with self.assertRaisesRegex(materials.MaterialsError, "replace it"):
            materials.save("MnO_mine_test", spec)


@unittest.skipUnless(live(), "qchem_py with the materials face not built (QCHEM_BUILD)")
class TheSetupPanel(unittest.TestCase):
    def setUp(self):
        reg = BackendRegistry([FakeBackend()])
        self.draft = SpecDraft()
        self.panel = SetupPanel(self.draft, reg)

    def test_picking_a_material_makes_the_draft_that_crystal_and_spins_are_stated_after(self):
        self.panel.refresh_materials()
        i = self.panel.material_box.findText("NaF_rocksalt")
        self.assertGreater(i, 0)
        self.panel._pick_material(i)
        s = self.draft.spec
        self.assertEqual(s.structure.source["materials"], "NaF_rocksalt")
        self.assertEqual(self.panel.material_box.currentText(), "NaF_rocksalt")
        self.panel.spins_edit.setText("+ -")
        self.panel._on_spins()
        self.assertEqual(self.draft.spec.physics.spin.site_moments, (1.0, -1.0))
        self.assertTrue(self.draft.spec.physics.spin.polarized)
        self.panel.spins_edit.setText("+ - 0")
        self.panel._on_spins()
        self.assertIn("3 spins for 2 sites", self.panel.reason.text())
        self.assertEqual(self.draft.spec.physics.spin.site_moments, (1.0, -1.0), "a refused line leaves the draft")


@unittest.skipUnless(live(), "qchem_py with the materials face not built (QCHEM_BUILD)")
class TheWorkspaceLibrary(unittest.TestCase):
    """A workspace may add its own materials library on top of the user's (OpenWork §2, ruled 2026-11-09): listed after it,
    saved into on request, stored with the workspace, and one name is one structure across them."""

    def setUp(self):
        self.ws_lib = os.path.join(_tempfile.mkdtemp(prefix="qchem-gui-ws-materials-"), "materials.json")
        self.spec = materials.spec_of("MnO", ROCKSALT, ROCKSALT["lattice"]["vectors_bohr"], {"import": ROCKSALT["provenance"]})

    def test_a_workspace_library_is_listed_after_the_user_s_and_saved_into(self):
        materials.save("InTheWorkspace", self.spec, workspace=self.ws_lib, into_workspace=True)
        ms = materials.listing(self.ws_lib)
        self.assertEqual(ms[-1]["name"], "InTheWorkspace")
        self.assertEqual(ms[-1]["layer"], "workspace")
        self.assertNotIn("InTheWorkspace", [m["name"] for m in materials.listing()], "without the workspace it is not listed")
        materials.save("MineToo", self.spec, replace=True, workspace=self.ws_lib)        # the default target is the user's library
        with self.assertRaisesRegex(materials.MaterialsError, "already in"):
            materials.save("MineToo", self.spec, workspace=self.ws_lib, into_workspace=True)   # one name, one structure
        with self.assertRaisesRegex(materials.MaterialsError, "no materials library"):
            materials.save("X", self.spec, into_workspace=True)

    def test_the_workspace_keeps_its_library_across_save_and_open(self):
        from qviz import project
        from qviz.workspace import Workspace
        reg = BackendRegistry([FakeBackend()])
        ws = Workspace("t", registry=reg); ws.materials = self.ws_lib
        path = os.path.join(_tempfile.mkdtemp(), "w.qproj.h5")
        project.save_workspace(path, ws)
        self.assertEqual(project.load_workspace(path, reg, compute=False).materials, self.ws_lib)

    def test_the_setup_panel_lists_it_and_says_so(self):
        materials.save("PanelWs", self.spec, workspace=self.ws_lib, into_workspace=True, replace=True)
        panel = SetupPanel(SpecDraft(), BackendRegistry([FakeBackend()]))
        seen = []
        panel.workspace_library_changed.connect(seen.append)
        panel._set_ws_library(self.ws_lib)
        self.assertEqual(seen, [self.ws_lib], "the app hears the choice, to store it with the workspace")
        self.assertGreater(panel.material_box.findText("PanelWs  (workspace)"), 0)
        panel._set_ws_library(None)
        self.assertLess(panel.material_box.findText("PanelWs  (workspace)"), 0)
        self.assertEqual(panel.ws_library_label.text(), "none")
