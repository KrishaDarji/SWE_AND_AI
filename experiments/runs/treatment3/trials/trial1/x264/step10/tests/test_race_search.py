import unittest
from pathlib import Path

from source.race_search import load_measurements, run


class RaceSearchTests(unittest.TestCase):
    def test_loads_expected_shape(self):
        path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
        x, y, names = load_measurements(path)
        self.assertEqual((len(x), len(y), len(names)), (1152, 1152, 16))
        self.assertTrue(all(len(row) == 16 for row in x))

    def test_run_is_deterministic_and_respects_budget(self):
        path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
        one = run(path, seed=3, budget=20)
        two = run(path, seed=3, budget=20)
        self.assertEqual(one, two)
        self.assertEqual(one["race"]["evaluated"], 20)
        self.assertEqual(one["baseline"]["evaluated"], 20)


if __name__ == "__main__":
    unittest.main()
