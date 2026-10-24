import csv
import pathlib
import tempfile
import unittest

from source.race import evaluate


class RaceTests(unittest.TestCase):
    def write(self, values):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.DictWriter(handle, fieldnames=["MRE-", "ACC+", "n_estimators"])
        writer.writeheader()
        writer.writerows(values)
        handle.close()
        self.addCleanup(pathlib.Path(handle.name).unlink)
        return handle.name

    def test_recent_median_beats_stale_history_after_shift(self):
        path = self.write([{"MRE-": value, "ACC+": value, "n_estimators": 1}
                           for value in [1, 1, 1, 1, 10, 10, 10, 10]])
        report = evaluate(path, window=2, split=0.5, columns=["MRE-"])
        self.assertLess(report["columns"]["MRE-"]["adaptive_mae"],
                        report["columns"]["MRE-"]["baseline_mae"])

    def test_only_past_rows_are_used(self):
        path = self.write([{"MRE-": value, "ACC+": .5, "n_estimators": 1}
                           for value in [1, 2, 3, 4]])
        item = evaluate(path, window=1, split=.5, columns=["MRE-"])["columns"]["MRE-"]
        self.assertEqual(item["evaluated"], 2)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            evaluate(self.write([{"MRE-": 1, "ACC+": 1, "n_estimators": 1}] * 2), window=0)


if __name__ == "__main__":
    unittest.main()
