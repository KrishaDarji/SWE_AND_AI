import csv
import os
import tempfile
import unittest

from source.blame import EXPECTED, analyze, load_dataset, quantile


DATA = os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv")


class BlameTests(unittest.TestCase):
    def test_load_and_validate(self):
        rows = load_dataset(DATA)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(set(rows[0]), set(EXPECTED))

    def test_quantile(self):
        self.assertEqual(quantile([1, 2, 3, 4], .75), 3.25)

    def test_analysis_is_ranked_and_balanced(self):
        result = analyze(DATA)
        self.assertEqual(result["target"], "MRE-")
        self.assertGreater(result["threshold"], 0)
        self.assertEqual(len(result["ranked_hyperparameters"]), 7)
        self.assertGreater(result["high_error_rate"], .2)
        self.assertLess(result["high_error_rate"], .3)

    def test_bad_value_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPECTED)
            writer.writeheader()
            row = {key: "0" for key in EXPECTED}
            row["criterion"] = "squared_error"
            row["MRE-"] = "2"
            writer.writerow(row)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                load_dataset(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
