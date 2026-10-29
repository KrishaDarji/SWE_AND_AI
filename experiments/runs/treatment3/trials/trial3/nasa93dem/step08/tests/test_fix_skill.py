import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from fix_skill import load_dataset, main, risk


DATASET = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"


class FixSkillTests(unittest.TestCase):
    def test_dataset_validates_and_has_expected_size(self):
        rows = load_dataset(DATASET.resolve())
        self.assertEqual(len(rows), 93)
        self.assertGreaterEqual(risk(rows[0]), 0)

    def test_recommendation_is_minimal_and_measurable(self):
        result = main(DATASET.resolve())
        self.assertGreater(result["candidate_count"], 0)
        recommendation = result["recommendation"]
        if recommendation:
            self.assertEqual(recommendation["cost"], 1.0)
            self.assertGreater(recommendation["reduction"], 0)
            self.assertNotEqual(recommendation["from"], recommendation["to"])


if __name__ == "__main__":
    unittest.main()
