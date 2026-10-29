import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "../source"))
import blame_nasa93dem


class BlameTests(unittest.TestCase):
    def test_validation_and_target(self):
        rows, fields, target, efforts = blame_nasa93dem.load_validate(blame_nasa93dem.DATASET)
        self.assertEqual(len(rows), 93)
        self.assertEqual(target, "Effort-")
        self.assertEqual(len(efforts), 93)

    def test_analysis_has_ranked_factors_and_high_group(self):
        result = blame_nasa93dem.analyze()
        self.assertGreater(result["high_count"], 0)
        self.assertEqual(len(result["ranked"]), len(result["features"]))
        self.assertTrue(all(len(item) == 5 for item in result["ranked"]))


if __name__ == "__main__":
    unittest.main()
