"""qviz.bse: the Basis Set Exchange picker, download and local store -- with the network replaced by canned answers.

The canned basis is the Exchange's own gaussian94 text for cc-pVDZ H and O (fetched 2026-11-10); the translation is the
library's (qchem_py.import_basis), so these tests need the extension (QCHEM_BUILD).
"""
import json
import os
import pathlib
import tempfile
import unittest

from qviz import bse
from qviz.backends.solid2 import _import_qchem_py
from qviz.spec import molecule

QC, WHY = _import_qchem_py()

CC_HO = """!----------------------------------------------------------------------
! Basis Set Exchange
! Version 0.12
! https://www.basissetexchange.org
!----------------------------------------------------------------------
!   Basis set: cc-pVDZ
! Description: cc-pVDZ
!        Role: orbital
!     Version: 1  (Data from ccRepo/Grant Hill)
!----------------------------------------------------------------------


H     0
S    4   1.00
      1.301000D+01           1.968500D-02
      1.962000D+00           1.379770D-01
      4.446000D-01           4.781480D-01
      1.220000D-01           5.012400D-01
S    1   1.00
      1.220000D-01           1.000000D+00
P    1   1.00
      7.270000D-01           1.0000000
****
O     0
S    9   1.00
      1.172000D+04           7.100000D-04
      1.759000D+03           5.470000D-03
      4.008000D+02           2.783700D-02
      1.137000D+02           1.048000D-01
      3.703000D+01           2.830620D-01
      1.327000D+01           4.487190D-01
      5.025000D+00           2.709520D-01
      1.013000D+00           1.545800D-02
      3.023000D-01          -2.585000D-03
S    9   1.00
      1.172000D+04          -1.600000D-04
      1.759000D+03          -1.263000D-03
      4.008000D+02          -6.267000D-03
      1.137000D+02          -2.571600D-02
      3.703000D+01          -7.092400D-02
      1.327000D+01          -1.654110D-01
      5.025000D+00          -1.169550D-01
      1.013000D+00           5.573680D-01
      3.023000D-01           5.727590D-01
S    1   1.00
      3.023000D-01           1.000000D+00
P    4   1.00
      1.770000D+01           4.301800D-02
      3.854000D+00           2.289130D-01
      1.046000D+00           5.087280D-01
      2.753000D-01           4.605310D-01
P    1   1.00
      2.753000D-01           1.000000D+00
D    1   1.00
      1.185000D+00           1.0000000
****

"""

META = {
    "cc-pvdz": {"display_name": "cc-pVDZ", "description": "cc-pVDZ", "role": "orbital", "family": "dunning",
                "function_types": ["gto", "gto_spherical"], "latest_version": "1",
                "versions": {"1": {"elements": ["1", "8"]}}},
    "def2-svp": {"display_name": "def2-SVP", "description": "def2-SVP", "role": "orbital", "family": "ahlrichs",
                 "function_types": ["gto", "scalar_ecp"], "latest_version": "1",
                 "versions": {"1": {"elements": ["1", "8", "26"]}}},
    "cc-pvdz-rifit": {"display_name": "cc-pVDZ-RIFIT", "description": "RI fit", "role": "rifit", "family": "dunning",
                      "function_types": ["gto"], "latest_version": "1", "versions": {"1": {"elements": ["1", "8"]}}},
    "h-only": {"display_name": "H only", "description": "hydrogen", "role": "orbital", "family": "x",
               "function_types": ["gto"], "latest_version": "2", "versions": {"2": {"elements": ["1"]}}},
}


class BSE(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="qchem-gui-bases-")
        self._saved = (bse.BASES_DIR, bse.fetch, bse._metadata)
        bse.BASES_DIR, bse._metadata = self.dir, None
        self.urls = []

        def fake(url):
            self.urls.append(url)
            if url.endswith("/metadata/"):
                return json.dumps(META).encode()
            if "/basis/cc-pvdz/format/gaussian94/" in url:
                return CC_HO.encode()
            if "/references/" in url:
                return b"pritchard2019a\n    A New Basis Set Exchange\n"
            raise AssertionError(f"unexpected fetch {url}")
        bse.fetch = fake

    def tearDown(self):
        bse.BASES_DIR, bse.fetch, bse._metadata = self._saved

    def test_the_picker_offers_orbital_sets_that_cover_the_structure_and_no_ecp_set(self):
        got = [c["key"] for c in bse.candidates({1, 8})]
        self.assertEqual(got, ["cc-pvdz"], "def2-SVP carries ECPs, the RI fit is not an orbital set, H-only misses O")
        self.assertEqual([c["key"] for c in bse.candidates({1})], ["cc-pvdz", "h-only"])
        self.assertEqual([c["key"] for c in bse.candidates({1}, "hydrogen")], ["h-only"])
        self.assertTrue(bse.candidates({1, 8})[0]["spherical"])

    def test_the_catalogue_is_kept_for_offline_use(self):
        bse.metadata()
        bse._metadata = None
        bse.fetch = lambda url: (_ for _ in ()).throw(OSError("offline"))
        self.assertIn("cc-pvdz", bse.metadata(), "the last catalogue seen")

    @unittest.skipUnless(QC is not None, WHY or "")
    def test_a_download_is_translated_and_saved_with_its_version_and_citations(self):
        side = bse.download("cc-pvdz", QC)
        self.assertTrue(all(u.startswith("https://www.basissetexchange.org/api/") for u in self.urls), self.urls)
        path = pathlib.Path(side["path"])
        self.assertEqual(path.parent, pathlib.Path(self.dir))
        self.assertEqual(path.name, "cc-pvdz.v1.bsd")
        text = path.read_text()
        self.assertIn('BASIS="cc-pVDZ"', text)
        self.assertIn("version 1", text)
        self.assertIn("pritchard2019a", text, "the citations ride in the file's preamble")
        self.assertNotIn("D+", text.split("BASIS=")[1], "the library translated it")
        self.assertEqual(side["elements"], ["H", "O"])
        self.assertEqual([s["key"] for s in bse.installed()], ["cc-pvdz"])
        self.assertEqual([p.name for p in pathlib.Path(self.dir).iterdir() if p.name.startswith(".incoming")], [],
                         "the download's own directory is gone")
        self.assertTrue(any("spherical" in n for n in bse.notes(side)))

    @unittest.skipUnless(QC is not None, WHY or "")
    def test_an_ecp_set_and_an_unknown_one_are_refused_before_any_download(self):
        with self.assertRaisesRegex(bse.BSEError, "effective core potentials"):
            bse.download("def2-svp", QC)
        with self.assertRaisesRegex(bse.BSEError, "not in the Basis Set Exchange catalogue"):
            bse.download("nonesuch", QC)
        self.assertFalse(any("/basis/" in u for u in self.urls))

    @unittest.skipUnless(QC is not None, WHY or "")
    def test_a_set_the_library_cannot_represent_is_refused_in_its_words_and_nothing_is_saved(self):
        bse.fetch = lambda url: (json.dumps(META).encode() if url.endswith("/metadata/") else
                                 b"FE     0\nFE-ECP     3     10\n" if "/basis/" in url else b"")
        with self.assertRaisesRegex(bse.BSEError, "ECP"):
            bse.download("cc-pvdz", QC)
        self.assertEqual(bse.installed(), [])

    def test_a_non_https_url_is_refused(self):
        with self.assertRaisesRegex(bse.BSEError, "non-https"):
            bse._get("http://example.com/x")

    @unittest.skipUnless(QC is not None, WHY or "")
    def test_water_runs_on_a_downloaded_cc_pvdz_picked_in_the_basis_tab(self):
        from qviz.backends import BackendRegistry, FakeBackend
        from qviz.backends.solid2 import Solid2Backend
        from qviz.draft import SpecDraft
        from qviz.setup_panel import SetupPanel
        from qviz import basisinfo
        from PySide6 import QtWidgets
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        side = bse.download("cc-pvdz", QC)
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107])
        draft = SpecDraft(w)
        be = Solid2Backend()
        panel = SetupPanel(draft, BackendRegistry([FakeBackend(), be]), qc=QC)
        i = panel.basis_box.findData(side["path"])
        self.assertGreaterEqual(i, 0, "the download is listed in the Basis box")
        panel.basis_box.setCurrentIndex(i); panel.basis_box.activated.emit(i)
        s = draft.spec
        self.assertEqual(basisinfo.basis_file(s), side["path"])
        self.assertEqual(s.accuracy.representation["basis"], "cc-pvdz")
        self.assertIn("Basis Set Exchange", panel.bse_note.text())
        self.assertTrue(be.capabilities().can_run(s), be.capabilities().can_run(s).reasons)
        self.assertEqual(basisinfo.element_rows(s), [("O", 1, "3s 2p 1d", 15), ("H", 2, "2s 1p", 5)], "read from the download")
        job = be.translate(s)
        self.assertIn("downloaded", job.resolved["basis"])
        res = be.run(job)
        self.assertTrue(res.converged, res.failure)
        # ORACLE: PySCF 2.14.0 (~/Code/pyscf-env) RHF on the SAME downloaded text (CC_HO, split per element), cart=True (25
        # functions), this geometry in bohr, conv_tol 1e-12 -- its built-in cc-pVDZ gives the identical number:
        #     from pyscf import gto, scf; from pyscf.gto.basis import parse_gaussian
        #     basis = {el: parse_gaussian.parse(<the CC_HO block of el>) for el in ("H", "O")}
        #     mol = gto.M(atom="O 0 0 0; H 0 1.431 1.107; H 0 -1.431 1.107", unit="Bohr", basis=basis, cart=True)
        #     mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
        #     -> nao 25, converged, E_HF -76.0271296557   (built-in basis="cc-pvdz", cart=True: -76.0271296557)
        # Ours, 2026-11-10: -76.0271296557.  Gated at 1e-8 Ha: the default [F,D] 1e-5 stop puts the energy within ~1e-10.
        self.assertAlmostEqual(res.energy_Ha, -76.0271296557, delta=1e-8)
        basis = next(r["basis"] for k, r in res.record.items() if "basis" in r and "file" in r["basis"])
        self.assertEqual(basis["file"], side["path"])
        self.assertEqual(basis["nFunctions"], 25)
        self.assertTrue(basis["checksum"])
        panel.basis_box.setCurrentIndex(0); panel.basis_box.activated.emit(0)
        self.assertIsNone(basisinfo.basis_file(draft.spec), "a shipped basis drops the downloaded file")

    def test_a_crystal_cannot_take_a_downloaded_basis(self):
        from qviz.backends.solid2 import Solid2Backend
        from tests.test_solid2_deck import SI, crystal
        si = crystal("Si", *SI, accuracy={"representation": {"basis": "sipp_sr", "solid2": {"basis_file": __file__}}})
        v = Solid2Backend().capabilities().can_run(si)
        self.assertFalse(v)
        self.assertTrue(any("molecular" in r for r in v.reasons), v.reasons)

    def test_fit_gaps_name_the_elements_a1_coul_lacks(self):
        w = molecule("water", [8, 1, 1], [0, 0, 0, 0, 1.431, 1.107, 0, -1.431, 1.107])
        self.assertEqual(bse.fit_gaps(w), [])


if __name__ == "__main__":
    unittest.main()
