import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import spot_quality_anomalies


class SpotTests(unittest.TestCase):
    def test_flags_range_outlier_and_duplicate_inconsistency(self):
        rows = [
            {"MRE-": "0.1", "ACC+": "0.8", "depth": "2"},
            {"MRE-": "0.2", "ACC+": "0.7", "depth": "3"},
            {"MRE-": "0.15", "ACC+": "1.4", "depth": "4"},
            {"MRE-": "0.9", "ACC+": "0.6", "depth": "2"},
        ]
        result = spot_quality_anomalies(rows, ["MRE-", "ACC+"], z_threshold=2)
        self.assertEqual(result["flagged_rows"][0]["index"], 2)
        self.assertEqual(result["duplicate_inconsistency_count"], 1)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            spot_quality_anomalies([], ["ACC+"])
        with self.assertRaises(ValueError):
            spot_quality_anomalies([{"ACC+": "0.5"}], ["missing"])


if __name__ == "__main__":
    unittest.main()
