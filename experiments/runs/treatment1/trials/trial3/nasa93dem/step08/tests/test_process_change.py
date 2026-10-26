import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from process_change import load_rows, recommend, risk


class ProcessChangeTests(unittest.TestCase):
    def test_risk_is_defects_per_kloc(self):
        self.assertAlmostEqual(risk({"Defects-": "20", "Kloc+": "10"}), 2.0)

    def test_recommendation_has_positive_reduction(self):
        rows = [
            {"team": "l", "pmat": "l", "tool": "l", "site": "l", "sced": "l", "Kloc+": "10", "Defects-": "20", "Effort-": "100"},
            {"team": "h", "pmat": "h", "tool": "h", "site": "h", "sced": "h", "Kloc+": "20", "Defects-": "10", "Effort-": "200"},
        ]
        result = recommend(rows)
        self.assertIsNotNone(result["recommendation"])
        self.assertGreater(result["recommendation"]["risk_reduction"], 0)

    def test_load_rows_reads_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text("team,pmat,tool,site,sced,Kloc+,Defects-,Effort-\nl,l,l,l,l,1,2,3\n")
            self.assertEqual(len(load_rows(path)), 1)


if __name__ == "__main__":
    unittest.main()
