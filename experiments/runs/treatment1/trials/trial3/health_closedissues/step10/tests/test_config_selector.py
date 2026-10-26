import csv
import os
import tempfile
import unittest

from source.config_selector import read_rows, select, utility


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "PRED40+"])
        writer.writerows([[0.2, 0.8, 0.8], [0.1, 0.7, 0.9], [0.9, 0.9, 0.1], [0.4, 0.6, 0.6], [0.3, 0.5, 0.5]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_reads_rows_and_selects(self):
        rows = read_rows(self.file.name)
        train, holdout, chosen, baseline, scale = select(rows)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(holdout), 2)
        self.assertEqual(chosen["_index"], 0)
        self.assertEqual(baseline["_index"], 2)
        self.assertGreaterEqual(utility(chosen, scale), 0.0)

    def test_missing_metric_fails(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as f:
            f.write("MRE-,ACC+\n0.1,0.2\n")
            name = f.name
        try:
            with self.assertRaises(ValueError):
                read_rows(name)
        finally:
            os.unlink(name)


if __name__ == "__main__":
    unittest.main()
