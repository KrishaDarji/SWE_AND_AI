import csv
import pathlib
import tempfile
import unittest
from importlib.util import spec_from_file_location, module_from_spec

ROOT = pathlib.Path(__file__).parents[1]
spec = spec_from_file_location("blame", ROOT / "source" / "blame.py")
blame = module_from_spec(spec)
spec.loader.exec_module(blame)

class BlameTests(unittest.TestCase):
    def test_numeric_and_empty_data(self):
        self.assertEqual(blame.numeric(" 3.5 "), 3.5)
        self.assertIsNone(blame.numeric("x"))
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("idX,Effort-\n")
            name = f.name
        with self.assertRaises(ValueError):
            blame.render(name)

    def test_synthetic_factor_is_ranked(self):
        rows = [{"idX": str(i), "factor": "bad" if i < 3 else "good", "Effort-": str(100 if i < 3 else 10), "Kloc+": "1", "Defects-": "1", "MonthsX": "1"} for i in range(6)]
        threshold, count, results = blame.summarize(rows)
        self.assertEqual(count, 3)
        self.assertEqual(results[0][1], "factor")
        self.assertEqual(results[0][2][0][0], "bad")
        self.assertGreater(threshold, 10)

if __name__ == "__main__":
    unittest.main()
