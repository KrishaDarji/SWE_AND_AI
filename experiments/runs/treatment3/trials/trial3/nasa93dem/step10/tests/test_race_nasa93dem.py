import pathlib
import sys
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from race_nasa93dem import evaluate, load_dataset

DATASET = pathlib.Path(__file__).resolve().parents[8] / "moot/optimize/process/nasa93dem.csv"


class RaceTests(unittest.TestCase):
    def test_load_and_validate(self):
        fields, rows = load_dataset(DATASET)
        self.assertEqual(len(rows), 93)
        self.assertIn("Effort-", fields)

    def test_reproducible_comparison(self):
        first = evaluate(DATASET)
        second = evaluate(DATASET)
        self.assertEqual(first["scores"], second["scores"])
        self.assertLess(first["scores"]["treatment3"]["mae"], 2000)
        self.assertGreater(first["scores"]["historical_baseline"]["mae"], 0)


if __name__ == "__main__":
    unittest.main()
