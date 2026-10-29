import csv
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "source"))
import summarize_projects as sp

DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class SummaryTests(unittest.TestCase):
    def test_row_count_and_outcomes(self):
        result = sp.summarize(sp.read_rows(DATA))
        self.assertEqual(result["projects"], 93)
        self.assertEqual(set(result["outcomes"]), {"kloc", "effort", "defects", "months"})
        self.assertGreater(result["outcomes"]["months"]["max"], result["outcomes"]["months"]["min"])

    def test_year_groups_and_model(self):
        result = sp.summarize(sp.read_rows(DATA))
        self.assertEqual(sum(x["projects"] for x in result["by_year"]), 93)
        self.assertEqual(len(result["schedule_model"]["standardized_coefficients"]), len(sp.PREDICTORS))

    def test_render_is_readable(self):
        text = sp.render(sp.summarize(sp.read_rows(DATA)))
        self.assertIn("Historical NASA93 project summary", text)
        self.assertIn("Delivery outcomes", text)


if __name__ == "__main__":
    unittest.main()
