"""The SCF view: a plot tab per numeric column under the code's own title, a log axis for convergence measures, the
table alongside; and the SCF rows' project round trip.

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest tests.test_scfview
"""
import math
import os
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets  # noqa: E402

from qviz import project  # noqa: E402
from qviz.backends.cp2k import Cp2kTail  # noqa: E402
from qviz.data import SCFCell, SCFRow, Structure  # noqa: E402
from qviz.scfview import SCFView, column_xy, export_column  # noqa: E402
from tests.test_cp2k import OUTPUT  # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def cp2k_rows():
    t = Cp2kTail()
    for line in OUTPUT.splitlines(keepends=True):
        t.feed(line)
    return t.steps


class TheView(unittest.TestCase):
    def test_a_gated_column_shows_its_gate_and_turns_green_once_met(self):
        from qviz.scfview import PASS_GREEN
        v = SCFView()
        for i, fd in enumerate((3e-3, 2e-5, 4e-6), start=1):
            v.add_row(SCFRow(i, -1.0, (SCFCell("[F,D]", fd, "log", f"{fd:.2e}", 1e-5),
                                       SCFCell("ΔE/E", 1e-9, "log", "1.00e-09", None))))
        heads = [v.table.horizontalHeaderItem(k).text() for k in range(v.table.columnCount())]
        self.assertEqual(heads, ["#", "[F,D]\n(< 1e-05)", "ΔE/E"], "an ungated (off) column shows no gate")
        green = [v.table.item(r, 1).foreground().color().name() == PASS_GREEN for r in range(3)]
        self.assertEqual(green, [False, False, True], "white above the gate, green below")
        self.assertNotEqual(v.table.item(2, 2).foreground().color().name(), PASS_GREEN, "no gate, never green")
        self.assertEqual(v.table.item(2, 1).text(), "4.00e-06", "the gate in the header does not break the row lookup")

    def test_a_tab_per_numeric_column_under_the_code_s_title(self):
        v = SCFView()
        v.set_rows(cp2k_rows())
        tabs = [v.tabs.tabText(i) for i in range(v.tabs.count())]
        self.assertEqual(tabs, ["Time", "Convergence", "Total energy", "Change"], "Update method is text: table only")
        self.assertEqual([v.table.horizontalHeaderItem(k).text() for k in range(v.table.columnCount())],
                         ["#", "Update method", "Time", "Convergence", "Total energy", "Change"])
        self.assertEqual(v.table.rowCount(), 3)
        self.assertEqual(v.table.item(1, 5).text(), "-6.40E-02", "the cell as CP2K printed it")

    def test_log_for_convergence_measures_linear_for_energies(self):
        v = SCFView()
        v.set_rows(cp2k_rows())
        logy = {v.tabs.tabText(i): v.tabs.widget(i).getPlotItem().ctrl.logYCheck.isChecked() for i in range(v.tabs.count())}
        self.assertEqual(logy, {"Time": False, "Convergence": True, "Total energy": False, "Change": True})

    def test_a_log_column_plots_magnitudes_and_drops_what_a_log_axis_cannot_draw(self):
        rows = [SCFRow(i, -1.0, (SCFCell("ΔE/E", x, "log"),)) for i, x in enumerate((-1e-2, 0.0, math.nan, 3e-5), 1)]
        self.assertEqual(column_xy(rows, "ΔE/E", log=True), ([1, 4], [1e-2, 3e-5]))

    def test_the_first_convergence_measure_is_shown_until_the_user_picks(self):
        v = SCFView()
        v.set_rows(cp2k_rows())
        self.assertEqual(v.current_title, "Convergence")

    def test_the_picked_tab_survives_a_new_run_with_that_column(self):
        v = SCFView()
        v.set_rows(cp2k_rows())
        v.tabs.setCurrentIndex(3)                                   # the user picks "Change"
        v.set_rows(cp2k_rows())
        self.assertEqual(v.current_title, "Change")

    def test_a_column_that_first_reads_dashes_gets_its_tab_when_a_number_arrives(self):
        v = SCFView()
        v.add_row(SCFRow(1, -1.0, (SCFCell("Etotal", -1.0), SCFCell("Δρ", math.nan))))   # the console's "----"
        self.assertEqual(v.tabs.count(), 1)
        self.assertEqual(v.table.item(0, 2).text(), "----")
        v.add_row(SCFRow(2, -1.1, (SCFCell("Etotal", -1.1), SCFCell("Δρ", 1e-3, "log"))))
        self.assertEqual([v.tabs.tabText(i) for i in range(v.tabs.count())], ["Etotal", "Δρ"])

    def test_export_writes_the_picked_column(self):
        with tempfile.TemporaryDirectory() as d:
            export_column(f"{d}/c.png", cp2k_rows(), "Convergence", "Si")
            self.assertGreater(os.path.getsize(f"{d}/c.png"), 1000)


class TheProjectFile(unittest.TestCase):
    def test_rows_round_trip_by_title(self):
        rows = cp2k_rows()
        with tempfile.TemporaryDirectory() as d:
            project.save(f"{d}/p.h5", title="t", backend="CP2K", scf=rows,
                         structure=Structure(symbols=["Si"], positions=np.zeros((1, 3)), numbers=np.array([14])))
            back = project.load(f"{d}/p.h5")["scf"]
        self.assertEqual([r.titles for r in back], [r.titles for r in rows])
        self.assertEqual([c.scale for c in back[1].cells], [c.scale for c in rows[1].cells])
        self.assertAlmostEqual(back[2].value("Convergence"), 9e-8)
        self.assertAlmostEqual(back[2].energy, rows[2].energy)


if __name__ == "__main__":
    unittest.main()
