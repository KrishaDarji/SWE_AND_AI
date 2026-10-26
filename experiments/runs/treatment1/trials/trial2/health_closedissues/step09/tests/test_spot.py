import csv
import pathlib
import tempfile
import unittest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import spot


class SpotTests(unittest.TestCase):
    def test_detects_range_and_discordance(self):
        rows = [{"MRE-": 0.2, "ACC+": 1.1, "PRED40+": 0, "N_estimators": "1",
                 "criterion": "x", "Min_sample_leaves": "1", "Min_impurity_decrease": "0",
                 "Max_depth": "1", "_line": 2}]
        kinds = {item["kind"] for item in spot.inspect(rows)}
        self.assertIn("range_error", kinds)
        self.assertIn("discordant_quality", kinds)

    def test_read_rows_and_outlier(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as handle:
            writer = csv.writer(handle)
            writer.writerow(list(spot.QUALITY + spot.CONFIG))
            for value in [0.1] * 9 + [9.0]:
                writer.writerow([value, .5, .5, 1, "x", 1, 0, 1])
            path = handle.name
        rows = spot.read_rows(path)
        self.assertEqual(len(rows), 10)
        self.assertTrue(any(x["kind"] == "range_error" for x in spot.inspect(rows)))


if __name__ == "__main__":
    unittest.main()
