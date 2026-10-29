import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np

from source.nasa93_summary import race_compare


DATA = Path("/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class RaceTests(unittest.TestCase):
    def test_reproducibility_groups_and_arithmetic(self):
        first = race_compare(DATA)
        second = race_compare(DATA)
        self.assertEqual(first, second)
        self.assertEqual(first["split_count"], 10)
        for row in first["splits"]:
            self.assertTrue(row["group_separation"])
            self.assertAlmostEqual(row["mae_difference_model_minus_baseline"], row["model_mae"] - row["baseline_mae"])

    def test_training_mean_baseline_and_fresh_predictions(self):
        result = race_compare(DATA)
        for row in result["splits"]:
            self.assertTrue(np.isfinite(row["training_mean"]))
            self.assertEqual(len(row["predictions"]), len(row["observed_efforts"]))
            self.assertEqual(row["train_count"] + row["test_count"], 93)

    def test_held_out_outcomes_do_not_change_fit_or_mean(self):
        with open(DATA, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        original = race_compare(DATA)
        effort = rows[0].index("Effort-")
        for index in original["splits"][0]["test_indices"]:
            rows[index + 1][effort] = "999999"
        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False) as handle:
            csv.writer(handle).writerows(rows)
            altered_path = Path(handle.name)
        self.addCleanup(lambda: altered_path.unlink(missing_ok=True))
        altered = race_compare(altered_path)
        np.testing.assert_allclose(original["splits"][0]["predictions"], altered["splits"][0]["predictions"])
        self.assertEqual(original["splits"][0]["training_mean"], altered["splits"][0]["training_mean"])

    def test_summary_ties(self):
        result = race_compare(DATA, tie_tolerance=1e9)
        self.assertEqual(result["summary"]["ties"], 10)


if __name__ == "__main__":
    unittest.main()
