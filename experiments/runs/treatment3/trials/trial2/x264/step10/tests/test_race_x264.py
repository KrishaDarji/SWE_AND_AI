import csv, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / 'source'))
from race_x264 import load_dataset, race, random_baseline, FEATURES

DATA = Path(__file__).parents[8] / 'moot/optimize/config/X264_AllMeasurements.csv'

class RaceTests(unittest.TestCase):
    def test_load_and_validate(self):
        d = load_dataset(DATA); self.assertEqual(len(d), 1152); self.assertEqual(len(d[0][0]), 16)
    def test_reproducible_and_budgeted(self):
        d = load_dataset(DATA); self.assertEqual(race(d, 4), race(d, 4)); self.assertEqual(race(d, 4)[2], 32)
    def test_baseline_budget(self):
        d = load_dataset(DATA); self.assertEqual(random_baseline(d, 4)[2], 32)

if __name__ == '__main__': unittest.main()
