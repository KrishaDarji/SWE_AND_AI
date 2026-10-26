import csv
import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(__file__)
SOURCE = os.path.join(HERE, "..", "source", "blame.py")
spec = importlib.util.spec_from_file_location("blame", SOURCE)
blame = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blame)


class BlameTests(unittest.TestCase):
    def test_rank_correlation_detects_increasing_relation(self):
        self.assertAlmostEqual(blame.correlation([1, 2, 3], [10, 20, 30]), 1.0)

    def test_analyze_marks_top_quartile(self):
        rows = []
        for i in range(4):
            rows.append({"MRE-": float(i), "N_estimators": float(i),
                         "Min_sample_leaves": 1.0, "Min_impurity_decrease": 0.0,
                         "Max_depth": 5.0, "criterion": "x"})
        cutoff, count, numeric, categories = blame.analyze(rows)
        self.assertEqual(cutoff, 2.0)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(numeric[0][4], 1.0)

    def test_read_rows_rejects_missing_columns(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            f.write("MRE-\n1\n")
            name = f.name
        try:
            with self.assertRaises(ValueError):
                blame.read_rows(name)
        finally:
            os.unlink(name)


if __name__ == "__main__":
    unittest.main()
